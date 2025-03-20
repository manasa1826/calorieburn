import json
import pickle
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ModelInput(BaseModel): 
    Gender: int 
    Age: int  
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

# Determine the absolute path to the saved model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'saved_models', 'calorie_model.pkl')

# Debugging: Print the actual path where it looks for the model
print(f"Looking for model at: {model_path}")

# Load the trained model safely
calorie_model = None
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            calorie_model = pickle.load(f)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found at {model_path}")

@app.post('/calorie_burn')
def predict_calorie(input_parameters: ModelInput):
    if calorie_model is None:
        return {"error": "Model not loaded"}

    input_list = np.array([[  
        input_parameters.Gender, input_parameters.Age, input_parameters.Height,
        input_parameters.Weight, input_parameters.Duration, 
        input_parameters.Heart_Rate, input_parameters.Body_Temp
    ]])

    prediction = calorie_model.predict(input_list)
    return {"calories_burned": float(prediction[0])}  # Ensure JSON compatibility
