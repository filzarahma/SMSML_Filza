import requests
import random
import json
import time
import urllib3

# Nonaktifkan warning SSL karena menggunakan self-signed certificate
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "http://127.0.0.1:8000/predict"

COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]

def generate_valid_data():
    """Generate satu baris data valid secara acak"""
    return [[
        round(random.uniform(0.2, 0.8), 4),             # Age (normalized)
        random.choice([0, 1]),                          # Sex
        random.randint(0, 3),                           # ChestPainType
        round(random.uniform(0.5, 1.5), 2),             # RestingBP (normalized)
        round(random.uniform(0.3, 1.0), 4),             # Cholesterol (normalized)
        float(random.choice([0.0, 1.0])),               # FastingBS
        random.randint(0, 2),                           # RestingECG
        round(random.uniform(0.6, 1.0), 4),             # MaxHR (normalized)
        random.choice([0, 1]),                          # ExerciseAngina
        round(random.uniform(0.0, 1.0), 4),             # Oldpeak
        random.randint(0, 2)                            # ST_Slope
    ]]

def generate_invalid_data():
    """Generate request yang error (sintaks salah atau data kosong)"""
    options = [
        {},  # Empty payload
        {"wrong_key": "wrong_value"},  # Salah format
        {"dataframe_split": {"columns": COLUMNS}},  # Data hilang
        {"dataframe_split": {"data": [[1, 2, 3]]}},  # Kolom hilang
    ]
    return random.choice(options)

# Kirim 50 request
for i in range(100):
    if random.random() < 0.6:  # 80% request valid
        payload = {
            "dataframe_split": {
                "columns": COLUMNS,
                "data": generate_valid_data()
            }
        }
    else:  # 20% request error
        payload = generate_invalid_data()

    try:
        response = requests.post(URL, json=payload, timeout=3)
        print(f"[{i+1:02d}] Status: {response.status_code} | Response: {response.text}")
    except Exception as e:
        print(f"[{i+1:02d}] ERROR: {str(e)}")

    print(f"Request ke-{i+1}")
    delay = random.uniform(0.1, 30.0)  # jeda acak antara 0.1 hingga 30 detik
    print(f"Menunggu selama {delay:.2f} detik...\n")
    time.sleep(delay)
