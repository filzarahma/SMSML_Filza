import pandas as pd
import numpy as np
import joblib
import requests
import json
import os

def get_user_input():
    """
    Function to get user input for heart disease prediction one by one
    """
    print("\n=== Please enter the following information for heart disease prediction ===")
    
    try:
        # Age input
        age = input("Age: ")
        
        # Sex input
        sex = input("Sex (M/F): ")
        
        # Chest Pain Type input
        print("\nChest Pain Type options:")
        print("TA: Typical Angina")
        print("ATA: Atypical Angina")
        print("NAP: Non-anginal Pain")
        print("ASY: Asymptomatic")
        chest_pain_type = input("Chest Pain Type (TA/ATA/NAP/ASY): ")
        
        # Resting Blood Pressure input
        resting_bp = input("Resting Blood Pressure (in mm Hg): ")
        
        # Cholesterol input
        cholesterol = input("Cholesterol (in mg/dl): ")
        
        # Fasting Blood Sugar input
        fasting_bs = input("Fasting Blood Sugar > 120 mg/dl (0/1): ")
        
        # Resting ECG input
        print("\nResting ECG options:")
        print("Normal: Normal")
        print("ST: ST-T wave abnormality")
        print("LVH: Left ventricular hypertrophy")
        resting_ecg = input("Resting ECG (Normal/ST/LVH): ")
        
        # Max Heart Rate input
        max_hr = input("Maximum Heart Rate achieved: ")
        
        # Exercise Angina input
        exercise_angina = input("Exercise Induced Angina (Y/N): ")
        
        # Oldpeak input
        oldpeak = input("Oldpeak (ST depression induced by exercise relative to rest): ")
        
        # ST Slope input
        print("\nST Slope options:")
        print("Up: Upsloping")
        print("Flat: Flat")
        print("Down: Downsloping")
        st_slope = input("ST Slope (Up/Flat/Down): ")
        
        # Create a dictionary with user inputs
        user_data = {
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain_type,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }
        
        # Print summary of inputs for confirmation
        print("\n=== Input Summary ===")
        for key, value in user_data.items():
            print(f"{key}: {value}")
        
        confirm = input("\nIs this information correct? (Y/N): ")
        if confirm.upper() != 'Y':
            print("Let's try again.")
            return None
            
        return user_data
        
    except Exception as e:
        print(f"Error during input: {str(e)}")
        return None

def preprocess_data(user_data):
    """
    Function to preprocess user data using the saved preprocessor pipeline
    """
    # Convert user data to DataFrame
    df = pd.DataFrame([user_data])
    
    try:
        # Load the preprocessor pipeline
        # First try with the full path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        preprocessor_path = os.path.join(script_dir, '8. Lainnya/preprocessor_pipeline.joblib')
        
        if not os.path.exists(preprocessor_path):
            # Try relative path
            preprocessor_path = '8. Lainnya/preprocessor_pipeline.joblib'
        
        print(f"Loading preprocessor from: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        
        print("Input data for preprocessing:")
        print(df)
        
        # Preprocess the data
        processed_data = preprocessor.transform(df)
        
        print("Data successfully preprocessed")
        return processed_data
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def predict_heart_disease(processed_data):
    """
    Function to send processed data to API and get prediction
    """
    try:
        # Convert numpy array to list for JSON serialization
        data = processed_data.tolist()
        
        # Prepare the request payload
        payload = json.dumps({"data": data})
        
        # Set the headers
        headers = {'Content-Type': 'application/json'}
        
        print("Sending request to prediction API...")
        # Send POST request to the API
        response = requests.post('http://127.0.0.1:8000/predict', 
                                data=payload, 
                                headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None

def main():
    """
    Main function to run the heart disease prediction
    """
    print("\n=== Heart Disease Prediction System ===")
    print("This system predicts heart disease based on your health parameters")
    
    while True:
        # Get user input
        user_data = get_user_input()
        
        if user_data is None:
            retry = input("Would you like to try again? (Y/N): ")
            if retry.upper() != 'Y':
                break
            continue
        
        # Preprocess the data
        processed_data = preprocess_data(user_data)
        
        if processed_data is not None:
            # Get prediction from API
            prediction_result = predict_heart_disease(processed_data)
            
            if prediction_result is not None:
                # Display prediction result
                print("\n=== Prediction Result ===")
                
                # Extract the prediction and probability
                prediction = prediction_result.get('prediction', None)
                probability = prediction_result.get('probability', None)
                
                if prediction == 1:
                    print("Heart Disease: POSITIVE")
                    print("The model predicts that you may have heart disease.")
                else:
                    print("Heart Disease: NEGATIVE")
                    print("The model predicts that you do not have heart disease.")
                
                if probability:
                    if isinstance(probability, list):
                        prob = probability[0] if prediction == 1 else 1 - probability[0]
                        print(f"Confidence: {prob:.2%}")
                    else:
                        print(f"Confidence: {probability:.2%}")
            else:
                print("Failed to get prediction from the API.")
        else:
            print("Failed to preprocess the input data.")
        
        another = input("\nWould you like to make another prediction? (Y/N): ")
        if another.upper() != 'Y':
            break

if __name__ == "__main__":
    main()
