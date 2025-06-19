import requests
import json
import time
import logging
 
# Konfigurasi logging
logging.basicConfig(filename="api_model_logs.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
 
# Endpoint API model
API_URL = "http://127.0.0.1:5005/invocations"
 
# Contoh input untuk model berdasarkan heart_preprocessing.csv
input_data = {"dataframe_split": {"columns": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
                                            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", 
                                            "Oldpeak", "ST_Slope"],
                                  "data": [[0.2448979591836734, 1, 1, 0.7000000000000001, 0.47927031509121065, 
                                           0.0, 1, 0.7887323943661971, 0, 0.29545454545454547, 2]]}}
 
# Konversi data ke JSON
headers = {"Content-Type": "application/json"}
payload = json.dumps(input_data)
 
# Mulai mencatat waktu eksekusi
start_time = time.time()
 
try:
    # Kirim request ke API
    response = requests.post(API_URL, headers=headers, data=payload)
   
    # Hitung response time
    response_time = time.time() - start_time
 
    if response.status_code == 200:
        prediction = response.json()  # Ambil hasil prediksi
 
        # Logging hasil request
        logging.info(f"Request: {input_data}, Response: {prediction}, Response Time: {response_time:.4f} sec")
 
        print(f"Prediction: {prediction}")
        print(f"Response Time: {response_time:.4f} sec")
    else:
        logging.error(f"Error {response.status_code}: {response.text}")
        print(f"Error {response.status_code}: {response.text}")
 
except Exception as e:
    logging.error(f"Exception: {str(e)}")
    print(f"Exception: {str(e)}")