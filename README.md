# clever-tanken
AI to predict petrol station prices

# Abhängigkeiten
pip install -r requirements.txt

# .env anlegen (aus .env.example) und INFLUX_* eintragen

# Training (liest aus InfluxDB 3). Standard: 10min-Modell für 144 Vorhersagepunkte (24 h, alle 10 min).
python train.py

# Optional: 1h-Modell (nur ein 24h-Durchschnitt pro Vorhersage)
python train.py --resample 1h

# API & Web-Frontend (Preise, Vorhersage, beste Uhrzeit)
uvicorn api:app --reload --host 0.0.0.0 --port 8000
# Dann im Browser: http://localhost:8000

# Stationen-Anzeige: In static/stations.json können pro Station Name, Standort, Adresse, PLZ und Ort hinterlegt werden (Feld "id" = Clever-Tanken-Stationsnummer). Fehlt ein Eintrag, erscheint im Dropdown nur "Station &lt;id&gt;".