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
from datetime import datetime, timezone

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
# Standard-Historie für Training (Stunden). None = so viel wie nötig; 2 Jahre = 17520
DEFAULT_TRAINING_HOURS = int(os.getenv("DEFAULT_TRAINING_HOURS", "17520"))


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


def get_station_fuel_combinations(hours: int | None = None) -> list[tuple[int | str, str]]:
    """
    Ermittelt alle (station_id, fuel_type)-Kombinationen aus InfluxDB 3.

    Returns:
        Liste von (station_id, fuel_type), z. B. [(993, "ARAL Ultimate 102"), ...].
    """
    hours = hours if hours is not None else DEFAULT_TRAINING_HOURS
    sql = f"""
    SELECT DISTINCT "{INFLUX_STATION_ID_COL}", "{INFLUX_FUEL_TYPE_COL}"
    FROM "{INFLUX_TABLE}"
    WHERE time > now() - INTERVAL '{hours} hours'
    ORDER BY 1, 2
    """
    client = get_influx_client()
    try:
        try:
            df = client.query_dataframe(sql, language="sql")
        except AttributeError:
            table = client.query(query=sql, language="sql")
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
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Lädt Tankpreis-Zeitreihe aus InfluxDB 3, optional gefiltert nach Tankstelle und Sprit.

    Erwartetes Schema: time, price, station_id, fuel_type (Sprit, z. B. "ARAL Ultimate 102").
    """
    hours = hours if hours is not None else DEFAULT_TRAINING_HOURS
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
            df = client.query_dataframe(sql, language="sql")
        except AttributeError:
            table = client.query(query=sql, language="sql")
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
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Erstellt Lag- und Rollings-Features aus der Preis-Zeitreihe.

    - Resample auf stündliche Werte (oder resample_rule).
    - Lags und Rolling-Metriken als Features, nächster Durchschnittspreis als Ziel.
    """
    df = df.set_index("time").sort_index()
    if "price" not in df.columns:
        raise ValueError("DataFrame muss eine Spalte 'price' haben.")
    series = df["price"].resample(resample_rule).mean().ffill().bfill()

    # Fenster in Perioden (z. B. Stunden)
    lookback = lookback_hours
    horizon = horizon_hours

    # Features: Lags 1..lookback (stündlich), dann Rolling Mean/Std
    lags = list(range(1, min(lookback + 1, 168)))  # max 168 Lags (1 Woche stündlich)
    X_list = []
    y_list = []
    index_list = []

    for i in range(lookback, len(series) - horizon):
        window = series.iloc[i - lookback : i]
        row = {}
        for lag in lags:
            if lag <= len(window):
                row[f"lag_{lag}"] = window.iloc[-lag]
        # Rolling-Statistiken (z. B. 24h, 168h)
        if len(window) >= 24:
            row["rolling_mean_24"] = window.tail(24).mean()
            row["rolling_std_24"] = window.tail(24).std()
        if len(window) >= 168:
            row["rolling_mean_168"] = window.tail(168).mean()
            row["rolling_std_168"] = window.tail(168).std()
        # Ziel: Durchschnittspreis über den nächsten horizon
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
    """Trainiert GradientBoostingRegressor und gibt Modell + Metriken zurück."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
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
    lookback_hours: int,
    horizon_hours: int,
    resample_rule: str,
    station_id: int | str,
    fuel_type: str,
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
        "lookback_hours": lookback_hours,
        "horizon_hours": horizon_hours,
        "resample_rule": resample_rule,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return model_path


def run_training(
    hours: int | None = None,
    test_size: float = 0.2,
    resample_rule: str = "1h",
    station_id: int | str | None = None,
    fuel_type: str | None = None,
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
        combinations = get_station_fuel_combinations(hours=hours)
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
        df = load_tankpreise_from_influx(station_id=sid, fuel_type=ftype, hours=hours)
        if df.empty or len(df) < LOOKBACK_HOURS + PREDICT_HORIZON_HOURS:
            print(f"  Übersprungen: zu wenig Daten ({len(df)} Punkte).")
            continue

        print(f"  Geladene Daten: {len(df)} Punkte (Zeitraum: {hours} Stunden)")
        print("  Erstelle Features …")
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

        path = save_artifact(
            model,
            feature_names=list(X.columns),
            metrics=metrics,
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=PREDICT_HORIZON_HOURS,
            resample_rule=resample_rule,
            station_id=sid,
            fuel_type=ftype,
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
        default="1h",
        help="Pandas-Resample-Regel für Zeitreihe (default: 1h)",
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
    args = parser.parse_args()

    sid: int | str | None = int(args.station_id) if args.station_id and args.station_id.isdigit() else args.station_id
    run_training(
        hours=args.hours,
        test_size=args.test_size,
        resample_rule=args.resample,
        station_id=sid,
        fuel_type=args.fuel_type,
    )


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

    Args:
        aktuelle_tankpreise: Zeitreihe der letzten Preise (ältester zuerst).
        station_id: Tankstellen-ID (z. B. 993).
        fuel_type: Kraftstoffsorte (z. B. "ARAL Ultimate 102").
        model_dir: Optional: Ordner mit tankpreis_model.joblib und tankpreis_meta.json
                   (überschreibt station_id/fuel_type).

    Returns:
        Vorhergesagter (Durchschnitts-)Preis für den konfigurierten Horizont.
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
    lookback = meta["lookback_hours"]

    prices = np.asarray(aktuelle_tankpreise).ravel()
    if len(prices) < lookback:
        raise ValueError(
            f"Mindestens {lookback} Preispunkte nötig, erhalten: {len(prices)}"
        )
    window = prices[-lookback:]

    row = {}
    for i, name in enumerate(feature_names):
        if name.startswith("lag_"):
            lag = int(name.split("_")[1])
            if lag <= len(window):
                row[name] = window[-lag]
        elif name == "rolling_mean_24" and len(window) >= 24:
            row[name] = np.mean(window[-24:])
        elif name == "rolling_std_24" and len(window) >= 24:
            row[name] = np.std(window[-24:]) or 0.0
        elif name == "rolling_mean_168" and len(window) >= 168:
            row[name] = np.mean(window[-168:])
        elif name == "rolling_std_168" and len(window) >= 168:
            row[name] = np.std(window[-168:]) or 0.0
    for name in feature_names:
        if name not in row:
            row[name] = 0.0
    X = pd.DataFrame([row])[feature_names]
    return float(model.predict(X)[0])


if __name__ == "__main__":
    main()
