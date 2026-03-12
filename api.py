"""
API und Frontend für Tankpreise & Vorhersage.

Endpoints:
  GET /api/stations       – verfügbare Stationen + Kraftstoffsorten (mit trainiertem Modell)
  GET /api/prices         – letzte N Stunden Preise (InfluxDB)
  GET /api/prediction     – Vorhersage (hours=24|72|168: nächste 24h, 3 oder 7 Tage)
  GET /api/best-time      – historisch günstigste Uhrzeit (7 Tage) + Prognose
  best_time_past ist in der Antwort von GET /api/best-time enthalten (günstigster Wochentag+Uhrzeit, letzte 2 Wochen)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Train-Logik wiederverwenden (Influx, Modell, Vorhersage)
from train import (
    STATIONS_BASE_DIR,
    LOOKBACK_HOURS,
    LOOKBACK_PERIODS_10MIN,
    load_tankpreise_from_influx,
    predict_from_current_prices,
    predict_next_144_steps,
    station_model_dir,
)

app = FastAPI(title="Clever Tanken – Preise & Vorhersage", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_available_stations() -> list[dict]:
    """Ermittelt alle (station_id, fuel_type), für die ein Modell existiert."""
    result = []
    base = Path(STATIONS_BASE_DIR)
    if not base.exists():
        return result
    for station_dir in base.iterdir():
        if not station_dir.is_dir():
            continue
        try:
            sid = int(station_dir.name)
        except ValueError:
            sid = station_dir.name
        for fuel_dir in station_dir.iterdir():
            if fuel_dir.is_dir() and (fuel_dir / "tankpreis_model.joblib").exists():
                result.append({"station_id": sid, "fuel_type": fuel_dir.name})
    return sorted(result, key=lambda x: (str(x["station_id"]), x["fuel_type"]))


@app.get("/api/stations")
def api_stations():
    """Liste aller Stationen + Kraftstoffsorten mit trainiertem Modell."""
    return {"stations": get_available_stations()}


WEEKDAY_DE = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
PAST_DAYS_HOURS = 14 * 24  # 2 Wochen


@app.get("/api/prices")
def api_prices(
    station_id: int = Query(..., description="Tankstellen-ID"),
    fuel_type: str = Query(..., description="Kraftstoffsorte"),
    hours: int = Query(24, ge=1, le=168, description="Letzte N Stunden"),
):
    """Preise der letzten N Stunden aus InfluxDB."""
    try:
        df = load_tankpreise_from_influx(
            station_id=station_id,
            fuel_type=fuel_type,
            hours=hours,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"InfluxDB: {str(e)}")
    if df.empty:
        return {"data": []}
    if "time" not in df.columns:
        df = df.reset_index()
    df = df.sort_values("time")
    data = [
        {"time": row["time"].isoformat(), "price": round(float(row["price"]), 3)}
        for _, row in df[["time", "price"]].iterrows()
    ]
    return {"data": data}


# Vorhersage im 10-Minuten-Takt: 24 h = 144 Punkte, 72 h = 432, 168 h = 1008
PREDICTION_INTERVAL_MINUTES = 10
ALLOWED_HORIZON_HOURS = (24, 72, 168)  # 24 h, 3 Tage, 7 Tage


def _prediction_num_steps(hours: int) -> int:
    """Anzahl 10-Minuten-Schritte für gegebene Stunden."""
    return hours * 60 // PREDICTION_INTERVAL_MINUTES


@app.get("/api/prediction")
def api_prediction(
    station_id: int = Query(..., description="Tankstellen-ID"),
    fuel_type: str = Query(..., description="Kraftstoffsorte"),
    horizon: int = Query(24, description="Vorhersage in Stunden: 24, 72 oder 168"),
):
    """
    Vorhersage für die nächsten N Stunden im 10-Minuten-Takt.
    horizon: 24 (24h), 72 (3 Tage) oder 168 (7 Tage). Query-Parameter: horizon=72
    """
    try:
        h = int(horizon)
    except (TypeError, ValueError):
        h = 24
    if h not in ALLOWED_HORIZON_HOURS:
        h = 24
    hours = h
    num_steps = _prediction_num_steps(hours)
    try:
        df = load_tankpreise_from_influx(
            station_id=station_id,
            fuel_type=fuel_type,
            hours=30,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"InfluxDB: {str(e)}")
    if df.empty or "price" not in df.columns:
        raise HTTPException(status_code=404, detail="Keine Preisdaten für diese Kombination.")
    df = df.sort_values("time")
    df = df.set_index("time")
    series_10min = df["price"].resample("10min").mean().ffill().bfill()
    history_10min = series_10min.tail(LOOKBACK_PERIODS_10MIN + 10).tolist()
    last_ts = series_10min.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    points = []
    try:
        preds = predict_next_144_steps(
            history_10min,
            station_id=station_id,
            fuel_type=fuel_type,
            num_steps=num_steps,
        )
        for i, price in enumerate(preds):
            t = last_ts + timedelta(minutes=(i + 1) * PREDICTION_INTERVAL_MINUTES)
            points.append({"time": t.isoformat(), "price": round(price, 3)})
        predicted = round(sum(preds) / len(preds), 3)
    except (FileNotFoundError, ValueError):
        try:
            df_h = load_tankpreise_from_influx(
                station_id=station_id,
                fuel_type=fuel_type,
                hours=LOOKBACK_HOURS + 24,
            )
            df_h = df_h.sort_values("time")
            prices_h = df_h["price"].tolist()
            predicted = predict_from_current_prices(
                prices_h,
                station_id=station_id,
                fuel_type=fuel_type,
            )
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        predicted = round(predicted, 3)
        for i in range(num_steps):
            t = last_ts + timedelta(minutes=(i + 1) * PREDICTION_INTERVAL_MINUTES)
            points.append({"time": t.isoformat(), "price": predicted})

    return {
        "predicted_price": predicted,
        "horizon_hours": hours,
        "interval_minutes": PREDICTION_INTERVAL_MINUTES,
        "points": points,
    }


@app.get("/api/best-time")
def api_best_time(
    station_id: int = Query(..., description="Tankstellen-ID"),
    fuel_type: str = Query(..., description="Kraftstoffsorte"),
):
    """
    Historisch günstigste Uhrzeit (letzte 7 Tage, nach Stunde gruppiert)
    plus Vorhersage für die nächsten 24h. Enthält außerdem best_time_past (günstigster Wochentag+Uhrzeit der letzten 2 Wochen).
    """
    try:
        df = load_tankpreise_from_influx(
            station_id=station_id,
            fuel_type=fuel_type,
            hours=168,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"InfluxDB: {str(e)}")
    if df.empty or "price" not in df.columns:
        raise HTTPException(status_code=404, detail="Keine Preisdaten.")
    df = df.sort_values("time")
    df["hour"] = df["time"].dt.hour
    by_hour = df.groupby("hour")["price"].agg(["mean", "count"]).reset_index()
    by_hour = by_hour[by_hour["count"] >= 3]
    if by_hour.empty:
        best_hour = None
        best_message = "Keine ausreichenden Daten für Empfehlung."
    else:
        best_row = by_hour.loc[by_hour["mean"].idxmin()]
        best_hour = int(best_row["hour"])
        best_message = f"Historisch günstigste Zeit: {best_hour}:00–{best_hour + 1}:00 Uhr (Ø letzte 7 Tage)."
    prices = df["price"].tolist()
    predicted = None
    try:
        predicted = predict_from_current_prices(prices, station_id=station_id, fuel_type=fuel_type)
    except (FileNotFoundError, ValueError):
        pass

    best_time_past = {"weekday": None, "hour": None, "avg_price": None, "message": None}
    try:
        df_14 = load_tankpreise_from_influx(
            station_id=station_id,
            fuel_type=fuel_type,
            hours=PAST_DAYS_HOURS,
        )
        if not df_14.empty and "time" in df_14.columns and "price" in df_14.columns:
            df_14 = df_14.sort_values("time")
            df_14["weekday"] = df_14["time"].dt.weekday
            df_14["hour"] = df_14["time"].dt.hour
            agg = df_14.groupby(["weekday", "hour"])["price"].agg(["mean", "count"]).reset_index()
            agg = agg[agg["count"] >= 2]
            if not agg.empty:
                best = agg.loc[agg["mean"].idxmin()]
                wd = int(best["weekday"])
                hr = int(best["hour"])
                avg = round(float(best["mean"]), 3)
                best_time_past = {
                    "weekday": WEEKDAY_DE[wd],
                    "weekday_num": wd,
                    "hour": hr,
                    "avg_price": avg,
                    "message": f"In den letzten 2 Wochen war {WEEKDAY_DE[wd]} {hr}:00–{hr + 1}:00 Uhr im Schnitt am günstigsten (Ø {avg:.2f} €/l).",
                }
    except Exception:
        pass

    return {
        "best_hour": best_hour,
        "message": best_message,
        "predicted_price_24h": round(predicted, 3) if predicted is not None else None,
        "hourly_averages": [
            {"hour": int(r["hour"]), "avg_price": round(float(r["mean"]), 3)}
            for _, r in by_hour.iterrows()
        ] if not by_hour.empty else [],
        "best_time_past": best_time_past,
    }


@app.get("/")
def index():
    """Frontend ausliefern."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Frontend: static/index.html anlegen und /static/index.html aufrufen."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
