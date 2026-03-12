"""
Training des KI-Modells für Tankpreis-Vorhersage.

Lädt historische Tankpreise aus InfluxDB 3, erstellt Zeitreihen-Features
und trainiert ein Modell für Vorhersagen. Bei Anfrage mit Input
„Aktuelle Tankpreise“ kann das gespeicherte Modell dann Prognosen liefern.

Täglicher Ablauf: Einmal am Tag ausführen (z. B. per Cron). Es wird jedes Mal
auf der vollen Influx-Historie (inkl. der neuen Daten von fetch_petrol_data)
neu trainiert; ein bestehendes Modell wird überschrieben. Es wird kein
Weitertrainieren eines geladenen Modells durchgeführt.
"""

from __future__ import annotations

import os
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# InfluxDB 3 – optional mit certifi für Windows (gRPC TLS)
try:
    from influxdb_client_3 import InfluxDBClient3, flight_client_options
except ImportError:
    InfluxDBClient3 = None
    flight_client_options = None

_FLIGHT_OPTS = None
if flight_client_options is not None:
    try:
        import certifi
        with open(certifi.where(), "r", encoding="utf-8") as f:
            _FLIGHT_OPTS = flight_client_options(tls_root_certs=f.read())
    except Exception:
        pass

# Konfiguration aus Umgebungsvariablen
INFLUX_HOST = os.getenv("INFLUX_HOST", "localhost:8181")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "")
INFLUX_DATABASE = os.getenv("INFLUX_DATABASE", "tankpreise")
INFLUX_TABLE = os.getenv("INFLUX_TABLE", "tankpreise")
INFLUX_TIME_COL = os.getenv("INFLUX_TIME_COL", "time")
INFLUX_PRICE_COL = os.getenv("INFLUX_PRICE_COL", "price")
INFLUX_STATION_ID_COL = os.getenv("INFLUX_STATION_ID_COL", "station_id")
INFLUX_FUEL_TYPE_COL = os.getenv("INFLUX_FUEL_TYPE_COL", "fuel_type")  # z. B. "ARAL Ultimate 102"
STATIONS_BASE_DIR = Path(os.getenv("STATIONS_BASE_DIR", "stations"))
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "168"))  # 7 Tage
PREDICT_HORIZON_HOURS = int(os.getenv("PREDICT_HORIZON_HOURS", "24"))
# InfluxDB 3 Core: Limit an Parquet-Files pro Query. 0/leer = nicht setzen.
INFLUX_QUERY_FILE_LIMIT = int(os.getenv("INFLUX_QUERY_FILE_LIMIT", "0") or "0")
# 10-Minuten-Vorhersage: Lookback = letzte 24 h als Eingabe (144 × 10 min). Das Modell lernt
# pro Schritt nur den *nächsten* 10-min-Preis; predict_next_144_steps ruft das 144× auf
# und ergibt so 144 verschiedene Preise = Vorhersagelinie für die nächsten 24 h.
PREDICT_INTERVAL_MINUTES = int(os.getenv("PREDICT_INTERVAL_MINUTES", "10"))
LOOKBACK_PERIODS_10MIN = 24 * 60 // PREDICT_INTERVAL_MINUTES  # 144 = Anzahl 10-min-Slots in 24 h (Eingabefenster)
# Standard-Historie für Training (Stunden). None = so viel wie nötig; 2 Jahre = 17520
DEFAULT_TRAINING_HOURS = int(os.getenv("DEFAULT_TRAINING_HOURS", "17520"))

_WARNED_QUERY_FILE_LIMIT_UNSUPPORTED = False


def _query_dataframe_safe(client, sql: str, *, query_file_limit: int | None = None) -> pd.DataFrame:
    """
    Führt eine SQL-Query aus und liefert ein DataFrame.

    InfluxDB 3 Core hat serverseitig ein `query-file-limit` (Parquet-File Scan Limit).
    Viele Setups zeigen an, man solle das Limit mit `--query-file-limit` erhöhen – das ist
    primär ein *Server* (serve) Flag.

    Manche Client-Versionen unterstützen das Weiterreichen entsprechender Optionen nicht
    (PyArrow `FlightCallOptions` akzeptiert dann kein `query_file_limit`). Deshalb:
    - wir versuchen es (falls gesetzt)
    - bei TypeError fallen wir zurück und warnen einmalig mit Hinweis auf Server-Config.
    """
    global _WARNED_QUERY_FILE_LIMIT_UNSUPPORTED
    if query_file_limit:
        try:
            return client.query_dataframe(sql, language="sql", query_file_limit=query_file_limit)
        except TypeError:
            if not _WARNED_QUERY_FILE_LIMIT_UNSUPPORTED:
                _WARNED_QUERY_FILE_LIMIT_UNSUPPORTED = True
                print(
                    "Hinweis: Dein influxdb3-python/PyArrow unterstützt das Weiterreichen von "
                    "`query_file_limit` nicht (TypeError in FlightCallOptions). "
                    "Bitte setze das Limit serverseitig beim Start von InfluxDB 3 Core, z. B.: "
                    "`influxdb3 serve --query-file-limit 2000` (oder ENV `INFLUXDB3_QUERY_FILE_LIMIT`)."
                )
    return client.query_dataframe(sql, language="sql")


def _query_table_safe(client, sql: str, *, query_file_limit: int | None = None):
    """Wie `_query_dataframe_safe`, aber nutzt `client.query()` (Arrow Table)."""
    global _WARNED_QUERY_FILE_LIMIT_UNSUPPORTED
    if query_file_limit:
        try:
            return client.query(query=sql, language="sql", query_file_limit=query_file_limit)
        except TypeError:
            if not _WARNED_QUERY_FILE_LIMIT_UNSUPPORTED:
                _WARNED_QUERY_FILE_LIMIT_UNSUPPORTED = True
                print(
                    "Hinweis: Dein influxdb3-python/PyArrow unterstützt das Weiterreichen von "
                    "`query_file_limit` nicht (TypeError in FlightCallOptions). "
                    "Bitte setze das Limit serverseitig beim Start von InfluxDB 3 Core, z. B.: "
                    "`influxdb3 serve --query-file-limit 2000` (oder ENV `INFLUXDB3_QUERY_FILE_LIMIT`)."
                )
    return client.query(query=sql, language="sql")


def _parse_influx_host():
    """
    Liefert (host für Client, port_overwrite oder None).
    Bei http:// wird die komplette URL durchgereicht → Client nutzt grpc+tcp (ohne TLS).
    """
    raw = INFLUX_HOST.strip()
    if raw.lower().startswith("http://"):
        return raw, None
    if raw.startswith(("https://", "grpc+", "grpc+tcp")):
        raw = raw.split("://", 1)[1].split("/")[0]
    if ":" in raw:
        hostname, port_str = raw.rsplit(":", 1)
        try:
            port = int(port_str)
            return hostname.strip(), port if port != 443 else None
        except ValueError:
            return raw, None
    return raw, None


def get_influx_client():
    """InfluxDB-3-Client mit optionalem TLS für Windows."""
    if InfluxDBClient3 is None:
        raise ImportError("influxdb3-python ist nicht installiert: pip install influxdb3-python pandas")
    if not INFLUX_TOKEN:
        raise ValueError(
            "INFLUX_TOKEN fehlt. Bitte in .env setzen oder exportieren."
        )
    host_arg, port = _parse_influx_host()
    kwargs = {
        "token": INFLUX_TOKEN,
        "host": host_arg,
        "database": INFLUX_DATABASE,
    }
    if port is not None:
        kwargs["query_port_overwrite"] = port
        kwargs["write_port_overwrite"] = port
    if _FLIGHT_OPTS is not None and not host_arg.lower().startswith("http://"):
        kwargs["flight_client_options"] = _FLIGHT_OPTS
    return InfluxDBClient3(**kwargs)


def get_station_fuel_combinations(
    hours: int | None = None,
    query_file_limit: int | None = None,
) -> list[tuple[int | str, str]]:
    """
    Ermittelt alle (station_id, fuel_type)-Kombinationen aus InfluxDB 3.

    Returns:
        Liste von (station_id, fuel_type), z. B. [(993, "ARAL Ultimate 102"), ...].
    """
    hours = hours if hours is not None else DEFAULT_TRAINING_HOURS
    if query_file_limit is None and INFLUX_QUERY_FILE_LIMIT > 0:
        query_file_limit = INFLUX_QUERY_FILE_LIMIT
    sql = f"""
    SELECT DISTINCT "{INFLUX_STATION_ID_COL}", "{INFLUX_FUEL_TYPE_COL}"
    FROM "{INFLUX_TABLE}"
    WHERE time > now() - INTERVAL '{hours} hours'
    ORDER BY 1, 2
    """
    client = get_influx_client()
    try:
        try:
            df = _query_dataframe_safe(client, sql, query_file_limit=query_file_limit)
        except AttributeError:
            table = _query_table_safe(client, sql, query_file_limit=query_file_limit)
            if hasattr(table, "read_all"):
                table = table.read_all()
            df = table.to_pandas()
    finally:
        client.close()

    sid_col = INFLUX_STATION_ID_COL
    ft_col = INFLUX_FUEL_TYPE_COL
    if sid_col not in df.columns or ft_col not in df.columns:
        raise ValueError(
            f"InfluxDB-Tabelle muss Spalten '{sid_col}' und '{ft_col}' haben."
        )
    return list(zip(df[sid_col].tolist(), df[ft_col].astype(str).tolist()))


def load_tankpreise_from_influx(
    station_id: int | str | None = None,
    fuel_type: str | None = None,
    hours: int | None = None,
    query_file_limit: int | None = None,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Lädt Tankpreis-Zeitreihe aus InfluxDB 3, optional gefiltert nach Tankstelle und Sprit.

    Erwartetes Schema: time, price, station_id, fuel_type (Sprit, z. B. "ARAL Ultimate 102").
    """
    hours = hours if hours is not None else DEFAULT_TRAINING_HOURS
    if query_file_limit is None and INFLUX_QUERY_FILE_LIMIT > 0:
        query_file_limit = INFLUX_QUERY_FILE_LIMIT
    cols = [INFLUX_TIME_COL, INFLUX_PRICE_COL, INFLUX_STATION_ID_COL, INFLUX_FUEL_TYPE_COL]
    if extra_columns:
        cols.extend(extra_columns)
    columns_sql = ", ".join(f'"{c}"' for c in cols)

    where_parts = [f"time > now() - INTERVAL '{hours} hours'"]
    if station_id is not None:
        if isinstance(station_id, (int, float)):
            where_parts.append(f'"{INFLUX_STATION_ID_COL}" = {int(station_id)}')
        else:
            where_parts.append(f"\"{INFLUX_STATION_ID_COL}\" = '{str(station_id).replace(chr(39), chr(39)+chr(39))}'")
    if fuel_type is not None:
        escaped = str(fuel_type).replace("'", "''")
        where_parts.append(f"\"{INFLUX_FUEL_TYPE_COL}\" = '{escaped}'")
    where_sql = " AND ".join(where_parts)

    sql = f"""
    SELECT {columns_sql}
    FROM "{INFLUX_TABLE}"
    WHERE {where_sql}
    ORDER BY {INFLUX_TIME_COL}
    """
    client = get_influx_client()
    try:
        try:
            df = _query_dataframe_safe(client, sql, query_file_limit=query_file_limit)
        except AttributeError:
            table = _query_table_safe(client, sql, query_file_limit=query_file_limit)
            if hasattr(table, "read_all"):
                table = table.read_all()
            df = table.to_pandas()
    finally:
        client.close()

    df[INFLUX_TIME_COL] = pd.to_datetime(df[INFLUX_TIME_COL], utc=True)
    df = df.sort_values(INFLUX_TIME_COL).drop_duplicates(subset=[INFLUX_TIME_COL])
    df = df.rename(columns={INFLUX_PRICE_COL: "price", INFLUX_TIME_COL: "time"})
    return df


def build_features(
    df: pd.DataFrame,
    lookback_hours: int = LOOKBACK_HOURS,
    horizon_hours: int = PREDICT_HORIZON_HOURS,
    resample_rule: str = "1h",
    lookback_periods: int | None = None,
    horizon_periods: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Erstellt Lag- und Rollings-Features aus der Preis-Zeitreihe.

    - Bei resample_rule="10min": Vorhersage des *nächsten* 10-Min-Preises (1 Schritt).
      Lookback = 144 Perioden = letzte 24 h als Eingabe. Die spätere 24-h-Linie entsteht,
      indem predict_next_144_steps das Modell 144× iterativ aufruft (jeweils nächste 10 min).
      Zusätzliche Features für plötzliche Niveauänderungen (z. B. Krisen): recent_mean_6,
      level_shift_6_24, level_shift_24_72, recent_max_24, recent_min_24 – damit die KI
      bei stark gestiegenen Preisen der letzten 24 h das neue Niveau vorhersagt.
    - Sonst: stündliches Resample, Lags + Rolling, Ziel = Ø-Preis über horizon_hours.
    """
    df = df.set_index("time").sort_index()
    if "price" not in df.columns:
        raise ValueError("DataFrame muss eine Spalte 'price' haben.")
    series = df["price"].resample(resample_rule).mean().ffill().bfill()

    use_10min = lookback_periods is not None and horizon_periods is not None
    if use_10min:
        lb, hor = lookback_periods, horizon_periods
        lags = list(range(1, lb + 1))
        X_list = []
        y_list = []
        index_list = []
        for i in range(lb, len(series) - hor):
            window = series.iloc[i - lb : i]
            row = {}
            for lag in lags:
                if lag <= len(window):
                    row[f"lag_{lag}"] = window.iloc[-lag]
            if len(window) >= 24:
                row["rolling_mean_24"] = window.tail(24).mean()
                row["rolling_std_24"] = window.tail(24).std()
            if len(window) >= 72:
                row["rolling_mean_72"] = window.tail(72).mean()
                row["rolling_std_72"] = window.tail(72).std()
            # Aktuelles Niveau und plötzliche Schocks (z. B. Krisen): letzte 1h stark gewichten
            if len(window) >= 6:
                row["recent_mean_6"] = window.tail(6).mean()
            if len(window) >= 24:
                # Niveauverschiebung: letzte 1h vs. 3h davor – erkennt plötzliche Sprünge
                row["level_shift_6_24"] = window.tail(6).mean() - window.iloc[-24:-6].mean()
            if len(window) >= 72:
                # Längerer Schock: letzte 4h vs. 8h davor – neues Niveau bleibt erhalten
                row["level_shift_24_72"] = window.tail(24).mean() - window.iloc[-72:-24].mean()
            if len(window) >= 24:
                row["recent_max_24"] = window.tail(24).max()
                row["recent_min_24"] = window.tail(24).min()
            ts = series.index[i]
            hour = getattr(ts, "hour", 0)
            dow = getattr(ts, "dayofweek", 0)
            row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            row["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            target = series.iloc[i]
            X_list.append(row)
            y_list.append(target)
            index_list.append(series.index[i])
        feature_df = pd.DataFrame(X_list).ffill().fillna(0)
        target_series = pd.Series(y_list, index=index_list)
        return feature_df, target_series

    lookback = lookback_hours
    horizon = horizon_hours
    lags = list(range(1, min(lookback + 1, 168)))
    X_list = []
    y_list = []
    index_list = []
    for i in range(lookback, len(series) - horizon):
        window = series.iloc[i - lookback : i]
        row = {}
        for lag in lags:
            if lag <= len(window):
                row[f"lag_{lag}"] = window.iloc[-lag]
        if len(window) >= 24:
            row["rolling_mean_24"] = window.tail(24).mean()
            row["rolling_std_24"] = window.tail(24).std()
        if len(window) >= 168:
            row["rolling_mean_168"] = window.tail(168).mean()
            row["rolling_std_168"] = window.tail(168).std()
        target = series.iloc[i : i + horizon].mean()
        X_list.append(row)
        y_list.append(target)
        index_list.append(series.index[i])
    feature_df = pd.DataFrame(X_list).ffill().fillna(0)
    target_series = pd.Series(y_list, index=index_list)
    return feature_df, target_series


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[GradientBoostingRegressor, dict]:
    """Trainiert GradientBoostingRegressor für präzise 10-Min-Vorhersage (nächste 24 h)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,
        min_samples_leaf=8,
        subsample=0.85,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    return model, metrics


def station_model_dir(station_id: int | str, fuel_type: str) -> Path:
    """Pfad zum Modell-Ordner: stations/<station_id>/<fuel_type>/ (z. B. stations/993/ARAL Ultimate 102/)."""
    return STATIONS_BASE_DIR / str(station_id) / fuel_type


def save_artifact(
    model: GradientBoostingRegressor,
    feature_names: list[str],
    metrics: dict,
    station_id: int | str,
    fuel_type: str,
    resample_rule: str,
    lookback_hours: int | None = None,
    horizon_hours: int | None = None,
    lookback_periods: int | None = None,
    horizon_periods: int | None = None,
) -> Path:
    """Speichert Modell und Metadaten unter stations/<station_id>/<fuel_type>/."""
    out_dir = station_model_dir(station_id, fuel_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "tankpreis_model.joblib"
    meta_path = out_dir / "tankpreis_meta.json"

    joblib.dump(model, model_path)
    meta = {
        "station_id": str(station_id),
        "fuel_type": fuel_type,
        "feature_names": feature_names,
        "resample_rule": resample_rule,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if lookback_periods is not None:
        meta["lookback_periods"] = lookback_periods
        meta["horizon_periods"] = horizon_periods or 1
        meta["interval_minutes"] = PREDICT_INTERVAL_MINUTES
    if lookback_hours is not None:
        meta["lookback_hours"] = lookback_hours
        meta["horizon_hours"] = horizon_hours or 24
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return model_path


def run_training(
    hours: int | None = None,
    test_size: float = 0.2,
    resample_rule: str = "10min",
    station_id: int | str | None = None,
    fuel_type: str | None = None,
    query_file_limit: int | None = None,
) -> None:
    """
    Trainiert pro Tankstelle und Sprit ein Modell und speichert unter stations/<id>/<Sprit>/.

    Es wird immer auf der vollen Historie aus Influx neu trainiert (kein Weitertrainieren
    eines geladenen Modells). Ein bestehendes Modell wird dabei überschrieben.
    Täglicher Ablauf: fetch_petrol_data schreibt neue Preise nach Influx → train.py
    lädt alle Daten (inkl. der neuen), trainiert neu, speichert. So ist das Modell
    stets mit den neuesten Daten trainiert.

    Ohne station_id/fuel_type: alle (station_id, fuel_type)-Kombinationen aus InfluxDB
    werden nacheinander trainiert.
    """
    if station_id is not None and fuel_type is not None:
        combinations = [(station_id, fuel_type)]
    else:
        print("Ermittle Tankstellen und Kraftstoffsorten aus InfluxDB 3 …")
        combinations = get_station_fuel_combinations(hours=hours, query_file_limit=query_file_limit)
        if not combinations:
            raise ValueError(
                "Keine (station_id, fuel_type)-Kombinationen in InfluxDB gefunden. "
                f"Tabelle '{INFLUX_TABLE}' mit Spalten '{INFLUX_STATION_ID_COL}', '{INFLUX_FUEL_TYPE_COL}' prüfen."
            )
        print(f"  {len(combinations)} Kombination(en) gefunden.")

    for sid, ftype in combinations:
        print(f"\n--- Station {sid} / {ftype} ---")
        out_dir = station_model_dir(sid, ftype)
        if (out_dir / "tankpreis_model.joblib").exists():
            print("  Bestehendes Modell gefunden → wird mit allen Daten aus Influx neu trainiert (inkl. neuer Punkte).")
        df = load_tankpreise_from_influx(
            station_id=sid,
            fuel_type=ftype,
            hours=hours,
            query_file_limit=query_file_limit,
        )
        use_10min = resample_rule.strip().lower() == "10min"
        min_points = (LOOKBACK_PERIODS_10MIN + 200) if use_10min else (LOOKBACK_HOURS + PREDICT_HORIZON_HOURS)
        if df.empty or len(df) < min_points:
            print(f"  Übersprungen: zu wenig Daten ({len(df)} Punkte).")
            continue

        print(f"  Geladene Daten: {len(df)} Punkte (Zeitraum: {hours} Stunden)")
        print("  Erstelle Features …")
        if use_10min:
            X, y = build_features(
                df,
                resample_rule=resample_rule,
                lookback_periods=LOOKBACK_PERIODS_10MIN,
                horizon_periods=1,
            )
        else:
            X, y = build_features(
                df,
                lookback_hours=LOOKBACK_HOURS,
                horizon_hours=PREDICT_HORIZON_HOURS,
                resample_rule=resample_rule,
            )
        if X.empty or len(X) < 20:
            print("  Übersprungen: zu wenig Samples nach Feature-Erstellung.")
            continue

        print(f"  Trainings-Samples: {len(X)}")
        print("  Training …")
        model, metrics = train_model(X, y, test_size=test_size)
        print(f"  MAE:  {metrics['mae']:.4f}  RMSE: {metrics['rmse']:.4f}")

        if use_10min:
            path = save_artifact(
                model,
                feature_names=list(X.columns),
                metrics=metrics,
                station_id=sid,
                fuel_type=ftype,
                resample_rule=resample_rule,
                lookback_periods=LOOKBACK_PERIODS_10MIN,
                horizon_periods=1,
            )
        else:
            path = save_artifact(
                model,
                feature_names=list(X.columns),
                metrics=metrics,
                station_id=sid,
                fuel_type=ftype,
                resample_rule=resample_rule,
                lookback_hours=LOOKBACK_HOURS,
                horizon_hours=PREDICT_HORIZON_HOURS,
            )
        print(f"  Modell gespeichert: {path.parent}")


def main():
    parser = argparse.ArgumentParser(description="KI-Modell für Tankpreis-Vorhersage trainieren (pro Tankstelle + Sprit)")
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help="Anzahl Stunden Historie (default: 2 Jahre = 17520, für volle Nutzung aller Daten)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Anteil Testdaten (default: 0.2)",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="10min",
        help="Resample-Regel: 10min = 144-Schritt-Vorhersage (nächste 24 h alle 10 min), 1h = 24h-Durchschnitt (default: 10min)",
    )
    parser.add_argument(
        "--station-id",
        type=str,
        default=None,
        help="Nur diese Tankstellen-ID trainieren (z. B. 993)",
    )
    parser.add_argument(
        "--fuel-type",
        type=str,
        default=None,
        help="Nur diesen Kraftstoff trainieren (z. B. 'ARAL Ultimate 102')",
    )
    parser.add_argument(
        "--query-file-limit",
        type=int,
        default=None,
        help="InfluxDB 3 Core: erlaubt Query-Planung über mehr Parquet-Files (z. B. 1000). Alternativ ENV INFLUX_QUERY_FILE_LIMIT.",
    )
    args = parser.parse_args()

    sid: int | str | None = int(args.station_id) if args.station_id and args.station_id.isdigit() else args.station_id
    run_training(
        hours=args.hours,
        test_size=args.test_size,
        resample_rule=args.resample,
        station_id=sid,
        fuel_type=args.fuel_type,
        query_file_limit=args.query_file_limit,
    )


def _build_feature_row(
    window: np.ndarray,
    feature_names: list[str],
    step_ts: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Baut eine Zeile Features aus einem Fenster (für Vorhersage). step_ts = Zeitpunkt des vorherzusagenden Schritts (für hour_sin/cos, dow_sin/cos)."""
    row = {}
    for name in feature_names:
        if name.startswith("lag_"):
            lag = int(name.split("_")[1])
            if lag <= len(window):
                row[name] = window[-lag]
        elif name == "rolling_mean_24" and len(window) >= 24:
            row[name] = np.mean(window[-24:])
        elif name == "rolling_std_24" and len(window) >= 24:
            row[name] = np.std(window[-24:]) or 0.0
        elif name == "rolling_mean_72" and len(window) >= 72:
            row[name] = np.mean(window[-72:])
        elif name == "rolling_std_72" and len(window) >= 72:
            row[name] = np.std(window[-72:]) or 0.0
        elif name == "rolling_mean_168" and len(window) >= 168:
            row[name] = np.mean(window[-168:])
        elif name == "rolling_std_168" and len(window) >= 168:
            row[name] = np.std(window[-168:]) or 0.0
        elif name == "recent_mean_6" and len(window) >= 6:
            row[name] = np.mean(window[-6:])
        elif name == "level_shift_6_24" and len(window) >= 24:
            row[name] = np.mean(window[-6:]) - np.mean(window[-24:-6])
        elif name == "level_shift_24_72" and len(window) >= 72:
            row[name] = np.mean(window[-24:]) - np.mean(window[-72:-24])
        elif name == "recent_max_24" and len(window) >= 24:
            row[name] = np.max(window[-24:])
        elif name == "recent_min_24" and len(window) >= 24:
            row[name] = np.min(window[-24:])
    if step_ts is not None:
        hour = getattr(step_ts, "hour", 0)
        dow = getattr(step_ts, "dayofweek", 0)
        if "hour_sin" in feature_names:
            row["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
        if "hour_cos" in feature_names:
            row["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
        if "dow_sin" in feature_names:
            row["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
        if "dow_cos" in feature_names:
            row["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))
    for name in feature_names:
        if name not in row:
            row[name] = 0.0
    return pd.DataFrame([row])[feature_names]


def predict_from_current_prices(
    aktuelle_tankpreise: list[float] | np.ndarray | pd.Series,
    station_id: int | str | None = None,
    fuel_type: str | None = None,
    model_dir: Path | None = None,
) -> float:
    """
    Vorhersage auf Basis der aktuellen Tankpreise (z. B. letzte Stunden/Tage).

    Lädt das Modell für die angegebene Tankstelle und Kraftstoffsorte aus
    stations/<station_id>/<fuel_type>/ und liefert die prognostizierte Preishöhe.
    Unterstützt sowohl 1h-Modell (lookback_hours) als auch 10min-Modell (lookback_periods).
    """
    if model_dir is not None:
        base_dir = Path(model_dir)
    elif station_id is not None and fuel_type is not None:
        base_dir = station_model_dir(station_id, fuel_type)
    else:
        raise ValueError("Entweder (station_id und fuel_type) oder model_dir angeben.")
    model_path = base_dir / "tankpreis_model.joblib"
    meta_path = base_dir / "tankpreis_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden in {base_dir}. Zuerst train.py für diese Station/Sprit ausführen."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = joblib.load(model_path)
    feature_names = meta["feature_names"]
    lookback = meta.get("lookback_periods") or meta.get("lookback_hours")

    prices = np.asarray(aktuelle_tankpreise).ravel()
    if len(prices) < lookback:
        raise ValueError(
            f"Mindestens {lookback} Preispunkte nötig, erhalten: {len(prices)}"
        )
    window = prices[-lookback:]
    X = _build_feature_row(window, feature_names)
    return float(model.predict(X)[0])


def predict_next_144_steps(
    history_prices: list[float] | np.ndarray | pd.Series,
    station_id: int | str,
    fuel_type: str,
    num_steps: int = 144,
    last_ts: datetime | pd.Timestamp | None = None,
) -> list[float]:
    """
    Erzeugt die Vorhersagelinie für die nächsten 24 h im 10-Minuten-Takt (144 Punkte).

    Das Modell sagt pro Aufruf nur den *nächsten* 10-Min-Preis vorher. Hier wird es
    144× nacheinander aufgerufen: Fenster (letzte 24 h) → Prediction → Fenster um
    einen Schritt weiterschieben (ältester raus, Prediction rein) → wieder Prediction.
    last_ts = Zeitstempel des letzten Punkts in history_prices; für jeden Schritt wird
    step_ts = last_ts + (k+1)*10min gesetzt (für Tageszeit-/Wochentag-Features).
    """
    base_dir = station_model_dir(station_id, fuel_type)
    model_path = base_dir / "tankpreis_model.joblib"
    meta_path = base_dir / "tankpreis_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden in {base_dir}. Mit --resample 10min trainieren."
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "lookback_periods" not in meta:
        raise ValueError(
            "Modell ist kein 10min-Modell. Bitte mit python train.py --resample 10min neu trainieren."
        )
    model = joblib.load(model_path)
    feature_names = meta["feature_names"]
    lookback = meta["lookback_periods"]
    prices = np.asarray(history_prices).ravel()
    if len(prices) < lookback:
        raise ValueError(
            f"Mindestens {lookback} Preispunkte (24 h in 10-min) nötig, erhalten: {len(prices)}"
        )
    if last_ts is not None and hasattr(last_ts, "tzinfo") and last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    if last_ts is None:
        last_ts = datetime.now(timezone.utc)
    interval_min = meta.get("interval_minutes", PREDICT_INTERVAL_MINUTES)
    window = list(prices[-lookback:])
    predictions = []
    for k in range(num_steps):
        step_ts = last_ts + timedelta(minutes=(k + 1) * interval_min)
        X = _build_feature_row(np.array(window), feature_names, step_ts=step_ts)
        pred = float(model.predict(X)[0])
        predictions.append(pred)
        window = window[1:] + [pred]
    return predictions


if __name__ == "__main__":
    main()
