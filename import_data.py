"""
Importiert Tankpreise aus einer CSV-Datei (ohne Header) in InfluxDB 3.

Format pro Zeile: "YYYY-MM-DD HH:MM:SS,preis"
Beispiel: 2024-06-21 22:41:29,2.11

Verwendung:
  python import_data.py
  python import_data.py --file prices.csv --station 993 --fuel "ARAL Ultimate 102"
  python import_data.py --file prices.csv --batch-size 5000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from influxdb_client_3 import Point

# Influx-Client und Konfiguration aus fetch_petrol_data wiederverwenden
from fetch_petrol_data import (
    INFLUX_TABLE,
    get_influx_client,
)


def parse_csv_line(line: str) -> tuple[datetime, float] | None:
    """Parst eine Zeile 'YYYY-MM-DD HH:MM:SS,preis' → (datetime, price)."""
    line = line.strip()
    if not line:
        return None
    parts = line.split(",", 1)
    if len(parts) != 2:
        return None
    try:
        ts = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M:%S")
        price = float(parts[1].strip())
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts, price


def import_csv_to_influx(
    csv_path: Path,
    station_id: int = 993,
    fuel_type: str = "ARAL Ultimate 102",
    batch_size: int = 5000,
) -> int:
    """
    Liest die CSV (ohne Header), erstellt pro Zeile einen Influx-Punkt
    und schreibt in Batches nach InfluxDB 3.

    Returns:
        Anzahl importierter Zeilen.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

    client = get_influx_client()
    total = 0
    batch: list[Point] = []

    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = parse_csv_line(line)
                if parsed is None:
                    continue
                ts, price = parsed
                point = (
                    Point(INFLUX_TABLE)
                    .tag("station_id", str(station_id))
                    .tag("fuel_type", fuel_type)
                    .field("price", price)
                    .time(ts)
                )
                batch.append(point)
                if len(batch) >= batch_size:
                    client.write(batch)
                    total += len(batch)
                    print(f"  {total} Zeilen geschrieben …")
                    batch = []

        if batch:
            client.write(batch)
            total += len(batch)
    finally:
        client.close()

    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tankpreise aus CSV (Zeitstempel,Preis pro Zeile) in InfluxDB 3 importieren."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=Path("prices.csv"),
        help="Pfad zur CSV-Datei (Standard: prices.csv)",
    )
    parser.add_argument(
        "--station",
        type=int,
        default=993,
        help="Tankstellen-ID (Standard: 993)",
    )
    parser.add_argument(
        "--fuel",
        type=str,
        default="ARAL Ultimate 102",
        help="Kraftstoffsorte (Standard: ARAL Ultimate 102)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Anzahl Zeilen pro Schreib-Batch (Standard: 5000)",
    )
    args = parser.parse_args()

    try:
        n = import_csv_to_influx(
            csv_path=args.file,
            station_id=args.station,
            fuel_type=args.fuel,
            batch_size=args.batch_size,
        )
        print(f"Fertig: {n} Preispunkte nach InfluxDB importiert.")
        return 0
    except FileNotFoundError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Fehler beim Import: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
