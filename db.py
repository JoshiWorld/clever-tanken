"""
SQLite-Zugriff für statische Tankstellendaten.

Tabelle `stations`: id (PK), name, standort, adresse, plz, ort.
Die ID entspricht der Clever-Tanken-Stationsnummer.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Pfad zur SQLite-Datei (Standard: data/stations.db im Projektroot)
DEFAULT_DB_DIR = Path(__file__).resolve().parent / "data"
STATIONS_DB_ENV = os.getenv("STATIONS_DB", "")
if STATIONS_DB_ENV:
    DB_PATH = Path(STATIONS_DB_ENV)
else:
    DB_PATH = DEFAULT_DB_DIR / "stations.db"


def _get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Erstellt die Tabelle `stations`, falls sie nicht existiert."""
    sql = """
    CREATE TABLE IF NOT EXISTS stations (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL DEFAULT '',
        standort TEXT NOT NULL DEFAULT '',
        adresse TEXT NOT NULL DEFAULT '',
        plz TEXT NOT NULL DEFAULT '',
        ort TEXT NOT NULL DEFAULT ''
    )
    """
    with _get_connection() as conn:
        conn.execute(sql)


def get_all_stations() -> list[dict]:
    """Alle Tankstellen aus der DB (id, name, standort, adresse, plz, ort)."""
    init_db()
    with _get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name, standort, adresse, plz, ort FROM stations ORDER BY id"
        )
        return [dict(row) for row in cur.fetchall()]


def get_station_ids() -> list[int]:
    """Alle Tankstellen-IDs (für fetch_petrol_data)."""
    init_db()
    with _get_connection() as conn:
        cur = conn.execute("SELECT id FROM stations ORDER BY id")
        return [row[0] for row in cur.fetchall()]


def get_station(station_id: int) -> dict | None:
    """Eine Tankstelle anhand der ID laden."""
    init_db()
    with _get_connection() as conn:
        cur = conn.execute(
            "SELECT id, name, standort, adresse, plz, ort FROM stations WHERE id = ?",
            (station_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def create_station(
    station_id: int,
    name: str = "",
    standort: str = "",
    adresse: str = "",
    plz: str = "",
    ort: str = "",
) -> dict:
    """Tankstelle anlegen. Wirft bei doppelter ID."""
    init_db()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO stations (id, name, standort, adresse, plz, ort)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (station_id, name or "", standort or "", adresse or "", plz or "", ort or ""),
        )
    return get_station(station_id) or {}


def update_station(
    station_id: int,
    name: str | None = None,
    standort: str | None = None,
    adresse: str | None = None,
    plz: str | None = None,
    ort: str | None = None,
) -> dict | None:
    """Tankstelle aktualisieren. Nur übergebene Felder werden geändert."""
    init_db()
    station = get_station(station_id)
    if not station:
        return None
    updates = []
    params = []
    for key, val in [
        ("name", name),
        ("standort", standort),
        ("adresse", adresse),
        ("plz", plz),
        ("ort", ort),
    ]:
        if val is not None:
            updates.append(f"{key} = ?")
            params.append(val)
    if not updates:
        return station
    params.append(station_id)
    with _get_connection() as conn:
        conn.execute(
            f"UPDATE stations SET {', '.join(updates)} WHERE id = ?",
            params,
        )
    return get_station(station_id)


def delete_station(station_id: int) -> bool:
    """Tankstelle löschen. Returns True wenn ein Eintrag gelöscht wurde."""
    init_db()
    with _get_connection() as conn:
        cur = conn.execute("DELETE FROM stations WHERE id = ?", (station_id,))
        return cur.rowcount > 0


def seed_from_json_file(path: Path) -> int:
    """
    Tankstellen aus einer JSON-Datei (Format wie stations.json) in die DB einfügen.
    Überspringt IDs, die bereits existieren. Gibt die Anzahl eingefügter Zeilen zurück.
    """
    import json
    init_db()
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return 0
    count = 0
    with _get_connection() as conn:
        for row in data:
            sid = row.get("id")
            if sid is None:
                continue
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO stations (id, name, standort, adresse, plz, ort)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid,
                        row.get("name") or "",
                        row.get("standort") or "",
                        row.get("adresse") or "",
                        row.get("plz") or "",
                        row.get("ort") or "",
                    ),
                )
                if conn.total_changes > 0:
                    count += 1
            except Exception:
                pass
    return count
