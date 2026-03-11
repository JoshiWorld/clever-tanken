# clever-tanken
AI to predict petrol station prices

# Abhängigkeiten
pip install -r requirements.txt

# .env anlegen (aus .env.example) und INFLUX_* eintragen

# Training (liest aus InfluxDB 3)
python train.py

# Optional: weniger Historie, anderes Resample
python train.py --hours 720 --resample 1h