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
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db

# Train-Logik wiederverwenden (Influx, Modell, Vorhersage)
from train import (
    STATIONS_BASE_DIR,
    LOOKBACK_HOURS,
    LOOKBACK_PERIODS_10MIN,
    get_station_fuel_combinations,
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

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


def require_admin(x_admin_password: Optional[str] = Header(None, alias="X-Admin-Password")):
    """Dependency: erfordert gültiges Admin-Passwort aus der Umgebung."""
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Admin-Zugang nicht konfiguriert (ADMIN_PASSWORD).")
    if x_admin_password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Ungültiges Passwort.")


@app.on_event("startup")
def startup():
    """Beim Start: DB initialisieren und ggf. aus stations.json befüllen."""
    db.init_db()
    if not db.get_all_stations():
        json_path = Path(__file__).resolve().parent / "static" / "stations.json"
        if json_path.exists():
            n = db.seed_from_json_file(json_path)
            if n:
                print(f"DB: {n} Tankstellen aus static/stations.json übernommen.")


def _has_model(station_id: Union[int, str], fuel_type: str) -> bool:
    """Prüft, ob für (station_id, fuel_type) ein KI-Modell existiert."""
    path = station_model_dir(station_id, fuel_type) / "tankpreis_model.joblib"
    return path.exists()


def get_available_stations() -> list[dict]:
    """Ermittelt alle (station_id, fuel_type) mit Ist-Daten aus Influx; has_model=True wenn Modell existiert."""
    result = []
    try:
        # combinations = get_station_fuel_combinations(hours=24 * 365)  # alle mit Daten im letzten Jahr
        combinations = get_station_fuel_combinations(hours=24)  # alle mit Daten der letzten 24 Stunden
    except Exception:
        # Fallback: nur Kombinationen mit existierendem Modell (bisheriges Verhalten)
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
                    result.append({
                        "station_id": sid,
                        "fuel_type": fuel_dir.name,
                        "has_model": True,
                    })
        return sorted(result, key=lambda x: (str(x["station_id"]), x["fuel_type"]))
    for sid, fuel_type in combinations:
        result.append({
            "station_id": sid,
            "fuel_type": fuel_type,
            "has_model": _has_model(sid, fuel_type),
        })
    return sorted(result, key=lambda x: (str(x["station_id"]), x["fuel_type"]))


@app.get("/api/stations")
def api_stations():
    """Liste aller Stationen + Kraftstoffsorten (mit Ist-Daten); has_model zeigt, ob Vorhersage verfügbar ist."""
    return {"stations": get_available_stations()}


@app.get("/api/stations-info")
def api_stations_info():
    """Statische Tankstellendaten aus der SQLite-DB (Name, Adresse, PLZ, Ort usw.)."""
    return db.get_all_stations()


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
            last_ts=last_ts,
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


# --- Admin (Passwort aus env) ---

class StationCreate(BaseModel):
    id: int
    name: str = ""
    standort: str = ""
    adresse: str = ""
    plz: str = ""
    ort: str = ""


class StationUpdate(BaseModel):
    name: Optional[str] = None
    standort: Optional[str] = None
    adresse: Optional[str] = None
    plz: Optional[str] = None
    ort: Optional[str] = None


@app.get("/admin")
def admin_page():
    """Admin-Seite für Tankstellenverwaltung (Login auf der Seite)."""
    admin_path = STATIC_DIR / "admin.html"
    if admin_path.exists():
        return FileResponse(admin_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="admin.html nicht gefunden.")


@app.get("/api/admin/stations", dependencies=[Depends(require_admin)])
def admin_list_stations():
    """Alle Tankstellen (Admin)."""
    return db.get_all_stations()


@app.get("/api/admin/stations/{station_id}", dependencies=[Depends(require_admin)])
def admin_get_station(station_id: int):
    """Eine Tankstelle (Admin)."""
    station = db.get_station(station_id)
    if not station:
        raise HTTPException(status_code=404, detail="Station nicht gefunden.")
    return station


@app.post("/api/admin/stations", dependencies=[Depends(require_admin)])
def admin_create_station(body: StationCreate):
    """Tankstelle anlegen."""
    try:
        station = db.create_station(
            station_id=body.id,
            name=body.name,
            standort=body.standort,
            adresse=body.adresse,
            plz=body.plz,
            ort=body.ort,
        )
        return station
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail=f"Station mit ID {body.id} existiert bereits.")


@app.put("/api/admin/stations/{station_id}", dependencies=[Depends(require_admin)])
def admin_update_station(station_id: int, body: StationUpdate):
    """Tankstelle aktualisieren."""
    updated = db.update_station(
        station_id,
        name=body.name,
        standort=body.standort,
        adresse=body.adresse,
        plz=body.plz,
        ort=body.ort,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Station nicht gefunden.")
    return updated


@app.delete("/api/admin/stations/{station_id}", dependencies=[Depends(require_admin)])
def admin_delete_station(station_id: int):
    """Tankstelle löschen."""
    if not db.delete_station(station_id):
        raise HTTPException(status_code=404, detail="Station nicht gefunden.")
    return {"ok": True}


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
