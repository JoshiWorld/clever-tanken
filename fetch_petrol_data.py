"""
Tankpreise von Clever-Tanken abrufen und in InfluxDB 3 schreiben.

Für Cron: Einmal pro Aufruf ausführen (keine Endlosschleife).
Konfiguration über .env oder Umgebungsvariablen (siehe .env.example).
"""

from __future__ import annotations

import os
import re
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

# InfluxDB 3
try:
    from influxdb_client_3 import InfluxDBClient3, Point, flight_client_options
except ImportError:
    InfluxDBClient3 = None
    Point = None
    flight_client_options = None

_FLIGHT_OPTS = None
if flight_client_options is not None:
    try:
        import certifi
        with open(certifi.where(), "r", encoding="utf-8") as f:
            _FLIGHT_OPTS = flight_client_options(tls_root_certs=f.read())
    except Exception:
        pass

# Konfiguration (gleiche Namen wie train.py für InfluxDB)
INFLUX_HOST = os.getenv("INFLUX_HOST", "localhost:8181")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "")
INFLUX_DATABASE = os.getenv("INFLUX_DATABASE", "tankpreise")
INFLUX_TABLE = os.getenv("INFLUX_TABLE", "tankpreise")

CLEVER_BASE_URL = "https://www.clever-tanken.de/tankstelle_details"
PRICE_LINE_OFFSET = 10  # Abstand Kraftstoffzeile → Preiszeile im HTML


def substitute_special_characters(s: str) -> str:
    replacements = {
        "&amp;": "und",
        "ß": "ss",
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "@": "at",
    }
    for old, new in replacements.items():
        s = re.sub(re.escape(old), new, s, flags=re.IGNORECASE)
    return re.sub(r" +", " ", s)


def extract_fuel_type(line: str) -> str | None:
    m = re.search(r'price-type-name">(.*?)</div>', line)
    return m.group(1).strip() if m else None


def extract_fuel_price(line: str) -> float | None:
    m = re.search(r'current-price-\d+">(\d+\.\d+)</span>', line)
    return float(m.group(1)) if m else None


def extract_station_street(line: str) -> str | None:
    m = re.search(r'<span itemprop="streetAddress">(.*?)</span>', line)
    return substitute_special_characters(m.group(1)) if m else None


def extract_station_name(line: str) -> str | None:
    m = re.search(r'<span class="strong-title" itemprop="name">(.*?)</span>', line)
    return substitute_special_characters(m.group(1)) if m else None


def fetch_and_parse_station(station_id: int) -> dict | None:
    """
    Lädt die Clever-Tanken-Detailseite für eine Tankstelle und extrahiert
    Name, Adresse und Preise pro Kraftstoffsorte.

    Returns:
        {"station_id": int, "station_name": str, "station_street": str, "prices": {fuel_type: price}}
        oder None bei Fehler.
    """
    url = f"{CLEVER_BASE_URL}/{station_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Fehler beim Abruf von Station {station_id}: {e}", file=sys.stderr)
        return None

    lines = resp.text.split("\n")
    station_name = None
    station_street = None
    prices = {}

    for i, line in enumerate(lines):
        name = extract_station_name(line)
        if name:
            station_name = name
            continue
        street = extract_station_street(line)
        if street:
            station_street = street
            continue
        fuel_type = extract_fuel_type(line)
        if fuel_type:
            if i + PRICE_LINE_OFFSET < len(lines):
                price = extract_fuel_price(lines[i + PRICE_LINE_OFFSET])
                if price is not None:
                    prices[fuel_type] = price

    if not station_name:
        print(f"Station {station_id}: Stationsname nicht gefunden.", file=sys.stderr)
    if not prices:
        print(f"Station {station_id}: Keine Preise extrahiert.", file=sys.stderr)
        return None

    return {
        "station_id": station_id,
        "station_name": station_name or "",
        "station_street": station_street or "",
        "prices": prices,
    }


def _parse_influx_host():
    """
    Liefert (host für Client, port_overwrite oder None).
    Bei http:// wird die komplette URL durchgereicht → Client nutzt grpc+tcp (ohne TLS).
    """
    raw = INFLUX_HOST.strip()
    # Ohne TLS: http://... durchreichen, dann verwendet die Library grpc+tcp
    if raw.lower().startswith("http://"):
        return raw, None
    # Mit TLS oder nur host:port: Hostname und ggf. Port extrahieren
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
    if InfluxDBClient3 is None or Point is None:
        raise ImportError(
            "influxdb3-python fehlt. Bitte installieren: pip install influxdb3-python"
        )
    if not INFLUX_TOKEN:
        raise ValueError(
            "INFLUX_TOKEN fehlt. Setze die Variable in .env oder der Umgebung."
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
    # Bei http:// (ohne TLS) keine Zertifikatsprüfung – certifi weglassen
    if _FLIGHT_OPTS is not None and not host_arg.lower().startswith("http://"):
        kwargs["flight_client_options"] = _FLIGHT_OPTS
    return InfluxDBClient3(**kwargs)


def write_prices_to_influx(
    station_id: int,
    station_name: str,
    station_street: str,
    prices: dict[str, float],
    ts: datetime | None = None,
) -> int:
    """
    Schreibt einen Datensatz pro (station_id, fuel_type) in InfluxDB 3.
    Schema: time, station_id, fuel_type, price (Tags: station_id, fuel_type; Field: price).

    Returns:
        Anzahl geschriebener Punkte.
    """
    if ts is None:
        ts = datetime.now(timezone.utc)
    client = get_influx_client()
    written = 0
    try:
        for fuel_type, price in prices.items():
            point = (
                Point(INFLUX_TABLE)
                .tag("station_id", str(station_id))
                .tag("fuel_type", fuel_type)
                .field("price", float(price))
                .time(ts)
            )
            client.write(point)
            written += 1
    finally:
        client.close()
    return written


def main(station_ids: list[int]) -> int:
    """
    Fetcht für jede Station die Preise von Clever-Tanken und schreibt sie in InfluxDB 3.
    Returns: 0 bei Erfolg, 1 bei Fehlern (z. B. keine Stationen, Schreibfehler).
    """
    if not station_ids:
        print("Keine Station-IDs angegeben (--station oder Einträge in der SQLite-DB).", file=sys.stderr)
        return 1

    total_written = 0
    for sid in station_ids:
        data = fetch_and_parse_station(sid)
        if data is None:
            continue
        try:
            n = write_prices_to_influx(
                station_id=data["station_id"],
                station_name=data["station_name"],
                station_street=data["station_street"],
                prices=data["prices"],
            )
            total_written += n
            print(f"Station {sid}: {n} Preispunkte nach InfluxDB geschrieben.")
        except Exception as e:
            print(f"Station {sid}: Fehler beim Schreiben nach InfluxDB: {e}", file=sys.stderr)

    if total_written == 0:
        return 1
    print(f"Gesamt: {total_written} Punkte in InfluxDB geschrieben.")
    return 0


def get_station_ids_from_db() -> list[int]:
    """Tankstellen-IDs aus der SQLite-DB (data/stations.db)."""
    try:
        import db
        return db.get_station_ids()
    except Exception as e:
        print(f"Hinweis: Stationen konnten nicht aus der DB geladen werden: {e}", file=sys.stderr)
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tankpreise von Clever-Tanken holen und in InfluxDB 3 schreiben (cron-tauglich)."
    )
    parser.add_argument(
        "--station",
        type=int,
        action="append",
        dest="stations",
        metavar="ID",
        help="Tankstellen-ID (mehrfach möglich, z. B. --station 993 --station 1000). Ohne Angabe: alle IDs aus der SQLite-DB.",
    )
    args = parser.parse_args()

    station_ids = args.stations or get_station_ids_from_db()
    if not station_ids:
        print(
            "Keine Station-IDs: weder --station angegeben noch Einträge in der SQLite-DB (data/stations.db).",
            file=sys.stderr,
        )
    exit_code = main(station_ids) if station_ids else 1
    sys.exit(exit_code)
