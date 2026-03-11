"""
API und Frontend für Tankpreise & Vorhersage.

Endpoints:
  GET /api/stations     – verfügbare Stationen + Kraftstoffsorten (mit trainiertem Modell)
  GET /api/prices       – letzte N Stunden Preise (InfluxDB)
  GET /api/prediction   – Vorhersage für nächsten Horizont
  GET /api/best-time    – historisch günstigste Uhrzeit + Prognose
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


# Vorhersage im 10-Minuten-Takt: 24 h = 144 Punkte
PREDICTION_INTERVAL_MINUTES = 10
PREDICTION_POINTS_24H = 24 * 60 // PREDICTION_INTERVAL_MINUTES  # 144


@app.get("/api/prediction")
def api_prediction(
    station_id: int = Query(..., description="Tankstellen-ID"),
    fuel_type: str = Query(..., description="Kraftstoffsorte"),
):
    """
    Vorhersage für die nächsten 24 Stunden im 10-Minuten-Takt (144 Punkte).
    Nutzt das 10min-Modell (train.py --resample 10min) für echte 144 Werte;
    sonst Fallback: ein 24h-Durchschnitt 144× wiederholt.
    """
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
    # Auf 10-Minuten resamplen, damit wir 144 Punkte für das 10min-Modell haben
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
            num_steps=PREDICTION_POINTS_24H,
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
        for i in range(PREDICTION_POINTS_24H):
            t = last_ts + timedelta(minutes=(i + 1) * PREDICTION_INTERVAL_MINUTES)
            points.append({"time": t.isoformat(), "price": predicted})

    return {
        "predicted_price": predicted,
        "horizon_hours": 24,
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
    plus Vorhersage für die nächsten 24h.
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
    return {
        "best_hour": best_hour,
        "message": best_message,
        "predicted_price_24h": round(predicted, 3) if predicted is not None else None,
        "hourly_averages": [
            {"hour": int(r["hour"]), "avg_price": round(float(r["mean"]), 3)}
            for _, r in by_hour.iterrows()
        ] if not by_hour.empty else [],
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
