# Clever Tanken – API & Frontend für Tankpreis-Vorhersage
FROM python:3.12-slim

WORKDIR /app

# Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Anwendung
COPY api.py train.py fetch_petrol_data.py import_data.py ./
COPY static/ ./static/

# Trainierte Modelle: zur Laufzeit als Volume mounten, z. B. -v $(pwd)/stations:/app/stations

EXPOSE 8000

# Umgebungsvariablen zur Laufzeit setzen (INFLUX_HOST, INFLUX_TOKEN, INFLUX_DATABASE, …)
# oder .env per Volume mounten: -v $(pwd)/.env:/app/.env
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
