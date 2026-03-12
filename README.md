# Clever Tanken – KI-Vorhersage für Tankpreise

Dieses Projekt lädt Tankpreise von Clever-Tanken.de, speichert sie in **InfluxDB 3 Core** und trainiert pro Tankstelle und Kraftstoffsorte ein **KI-Modell**, das den Preisverlauf für die nächsten **24 Stunden im 10-Minuten-Takt** (144 Punkte) vorhersagt. Ein **Web-Frontend** zeigt Ist-Preise, Prognose und die „beste Uhrzeit zum Tanken“.

---

## Aufbau des Projekts

```
clever-tanken/
├── .env                    # Konfiguration (nicht im Git, siehe .env.example)
├── .env.example            # Vorlage für Umgebungsvariablen
├── requirements.txt        # Python-Abhängigkeiten
│
├── fetch_petrol_data.py    # Holt Preise von Clever-Tanken → schreibt in InfluxDB (für Cron)
├── import_data.py          # Import historischer CSV-Daten in InfluxDB
├── train.py                # Training: InfluxDB → Features → Modell pro Station/Sprit
├── api.py                  # FastAPI: Endpoints + Auslieferung des Frontends
│
├── static/
│   ├── index.html          # Web-Frontend (Diagramm, Dropdowns, Light/Dark)
│   └── stations.json       # Namen/Adressen der Stationen für die Anzeige
│
└── stations/               # Trainierte Modelle (nach dem ersten Training)
    └── <station_id>/
        └── <fuel_type>/
            ├── tankpreis_model.joblib
            └── tankpreis_meta.json
```

| Komponente | Aufgabe |
|------------|--------|
| **InfluxDB 3 Core** | Speichert Zeitreihen: `time`, `price`, `station_id`, `fuel_type`. |
| **fetch_petrol_data.py** | Einmal pro Aufruf (z. B. per Cron): Preise von Clever-Tanken abfragen und in InfluxDB schreiben. |
| **import_data.py** | Historische Preise aus einer CSV (`Datum Uhrzeit,Preis`) in InfluxDB importieren. |
| **train.py** | Liest Daten aus InfluxDB, baut Features (Lags, Rolling, Schock-Features), trainiert pro Station/Sprit ein Modell und speichert es unter `stations/<id>/<Sprit>/`. |
| **api.py** | REST-API und Frontend: Preise, 144-Punkte-Vorhersage, beste Uhrzeit; nutzt die trainierten Modelle. |

---

## Voraussetzungen

- **Python 3.10+** (oder 3.9 mit passenden Typ-Hints)
- **InfluxDB 3 Core** (läuft und erreichbar, z. B. `http://localhost:8181` oder deine URL)
- **Database Token** für InfluxDB mit Lese- und Schreibrechten

---

## Schritt-für-Schritt: Einrichtung

### 1. Repository klonen und Abhängigkeiten installieren

```bash
cd clever-tanken
pip install -r requirements.txt
```

### 2. Konfiguration anlegen

Kopiere die Beispiel-Konfiguration und passe sie an:

```bash
cp .env.example .env
```

In der `.env` mindestens setzen:

- **INFLUX_HOST** – z. B. `http://192.168.178.52:8181` (ohne TLS) oder `https://…` (mit TLS)
- **INFLUX_TOKEN** – dein InfluxDB-Datenbank-Token
- **INFLUX_DATABASE** – Name der Datenbank (z. B. `tankpreise`)
- **STATION_IDS** – Tankstellen-IDs für den Abruf, kommagetrennt (z. B. `993`)

Alle weiteren Optionen (Tabelle, Spaltennamen, Training) stehen in `.env.example` und können bei Bedarf angepasst werden.

### 3. Daten in InfluxDB bekommen

**Variante A – Live-Daten (empfohlen für Dauerbetrieb)**  
Script per Cron regelmäßig ausführen (z. B. alle 10 Minuten):

```bash
python fetch_petrol_data.py
```

**Variante B – Historische Daten aus CSV importieren**  
CSV-Format pro Zeile: `YYYY-MM-DD HH:MM:SS,preis` (ohne Header), z. B.:

```text
2024-06-21 22:41:29,2.11
2024-06-21 22:51:30,2.09
```

```bash
python import_data.py --file prices.csv --station 993 --fuel "ARAL Ultimate 102"
```

Weitere Optionen: `--batch-size`, siehe `python import_data.py --help`.

### 4. Modell trainieren

Training lädt die konfigurierte Historie aus InfluxDB und trainiert **pro Kombination aus Station und Kraftstoffsorte** ein Modell. Standard: 10-Minuten-Vorhersage (144 Punkte für 24 h).

```bash
python train.py
```

Optional:

- **Kürzerer Zeitraum:** `python train.py --hours 336` (z. B. 14 Tage)
- **Nur eine Station/Sprit:**  
  `python train.py --station-id 993 --fuel-type "ARAL Ultimate 102"`
- **1h-Modell (nur 24h-Durchschnitt):**  
  `python train.py --resample 1h`

Hinweis: Bei InfluxDB 3 Core kann bei großen Zeiträumen das Parquet-File-Limit greifen. Dann den Server mit `--query-file-limit` starten (z. B. `influxdb3 serve --query-file-limit 2000`) oder den Zeitraum mit `--hours` verkleinern.

### 5. API und Frontend starten

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Im Browser: **http://localhost:8000**

Dort: Station und Kraftstoffsorte wählen → Anzeige der letzten 24 h Ist-Preise, der 24h-Vorhersagelinie (10-Min-Takt) und der „besten Uhrzeit zum Tanken“.

---

## API-Endpunkte

| Methode | Pfad | Beschreibung |
|--------|------|--------------|
| GET | `/` | Frontend (`static/index.html`) |
| GET | `/api/stations` | Liste der Stationen + Kraftstoffsorten (mit trainiertem Modell) |
| GET | `/api/prices?station_id=…&fuel_type=…&hours=24` | Preise der letzten N Stunden aus InfluxDB |
| GET | `/api/prediction?station_id=…&fuel_type=…` | Vorhersage: 144 Punkte (nächste 24 h, 10-Min-Intervall) |
| GET | `/api/best-time?station_id=…&fuel_type=…` | Historisch günstigste Stunde + Hinweis zur Prognose |

---

## Optionale Anpassungen

### Stationen im Frontend mit Namen und Adresse

In `static/stations.json` können pro Station Anzeigename, Standort, Adresse, PLZ und Ort hinterlegt werden. Das Feld `id` muss der Clever-Tanken-Stationsnummer entsprechen. Fehlt ein Eintrag, erscheint nur „Station &lt;id&gt;“.

Beispiel:

```json
[
  {
    "id": 993,
    "name": "ARAL Tankstelle Musterstadt",
    "standort": "Zentrum",
    "adresse": "Hauptstraße 1",
    "plz": "12345",
    "ort": "Musterstadt"
  }
]
```

### InfluxDB 3 Core: Parquet-File-Limit

Wenn beim Training Fehlermeldungen wie „Query would scan … Parquet files, exceeding the file limit“ auftreten, das Limit serverseitig erhöhen, z. B.:

```bash
influxdb3 serve --query-file-limit 2000
```

oder per Umgebungsvariable auf dem Server: `INFLUXDB3_QUERY_FILE_LIMIT=2000`.

---

## FAQ

### Was passiert beim Training und bei der Vorhersage?

1. **Daten** werden aus InfluxDB geladen (Zeitstempel + Preis pro Station/Kraftstoff).
2. **Aufbereitung:** Resampling auf 10-Minuten-Intervalle; für jeden Zeitpunkt wird ein **Eingabefenster** aus den letzten **144 Zeitschritten** (24 h) gebaut, plus Zusatzfeatures (Lags, Rolling-Mittel/Std, Schock-Features, Tageszeit/Wochentag).
3. **Zielgröße:** Das Modell lernt den **nächsten** 10-Minuten-Preis (ein Schritt).
4. **Training:** Ein Regressionsmodell lernt „Fenster + Zeit-Features → nächster Preis“. Für die Prognose der nächsten 24 h wird es **144-mal nacheinander** aufgerufen (autoregressiv: jede Prognose wird ins Fenster geschoben, nächster Schritt vorhergesagt).
5. **Speicherung:** Ein Modell pro Kombination **Station + Kraftstoff** unter `stations/<station_id>/<fuel_type>/`.

### Welches ML-Modell wird verwendet und warum?

Es kommt **Gradient Boosting** (scikit-learn: `GradientBoostingRegressor`) zum Einsatz.

- **Vorteile:** Funktioniert gut mit tabellarischen Features (Lags, Rollings, Tageszeit), braucht keine riesigen Datenmengen (passend für z. B. 2 Wochen), keine Skalierung nötig, robust gegen Ausreißer, schnell trainierbar und vorhersagbar, kein GPU nötig, einfach deploybar (joblib).
- **Warum kein LSTM/Transformer?** Würden mehr Daten und Aufwand erfordern; bei 2 Wochen und 10-Min-Auflösung bringt Boosting oft mehr.
- **Warum kein Prophet?** Eher für starke Saisonalität auf Tages-/Wochenebene; hier geht es um kurzen Horizont (24 h) und 10-Min-Detail.
- **Warum keine lineare Regression / ARIMA?** Würden die nichtlinearen Muster (Tagesgang, Schocks) schlechter abbilden – die Vorhersage wirkt dann „zu linear“.

Kurz: Gradient Boosting ist ein guter Kompromiss aus Genauigkeit, Stabilität und Einfachheit bei begrenzten Daten.

### Ab wie vielen Datenpunkten wird ein Kraftstoff trainiert?

- **Rohdaten (Influx-Zeilen):** Es wird nur trainiert, wenn **mindestens 344 Punkte** vorhanden sind (`LOOKBACK_PERIODS_10MIN + 200` = 144 + 200).
- **Nach der Feature-Erstellung:** Zusätzlich müssen **mindestens 20 Trainings-Samples** entstehen (sonst wird die Station/Sprit-Kombination übersprungen).

Für stabiles Training mit 2 Wochen im 10-Min-Takt (ca. 2000+ Punkte) bist du damit deutlich über dem Minimum.

---

## Kurzreferenz: Wichtige Befehle

| Aktion | Befehl |
|--------|--------|
| Abhängigkeiten installieren | `pip install -r requirements.txt` |
| Preise von Clever-Tanken holen (Cron) | `python fetch_petrol_data.py` |
| CSV in InfluxDB importieren | `python import_data.py --file prices.csv --station 993 --fuel "ARAL Ultimate 102"` |
| Modell trainieren (alle Stationen/Sprit) | `python train.py` |
| Modell trainieren (14 Tage, eine Station) | `python train.py --hours 336 --station-id 993 --fuel-type "ARAL Ultimate 102"` |
| API + Frontend starten | `uvicorn api:app --reload --host 0.0.0.0 --port 8000` |

---

## Lizenz

Siehe `LICENSE` im Projektverzeichnis.
