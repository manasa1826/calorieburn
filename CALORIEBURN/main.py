import json
import pickle
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define input model
class ModelInput(BaseModel): 
    Gender: int 
    Age: int  
    Height: float
    Weight: float
    Duration: float
    Heart_Rate: float
    Body_Temp: float

# Load the trained model safely
model_path = os.path.join(os.getcwd(), 'CALORIEBURN', 'saved_models', 'calorie_model.pkl')

if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            calorie_model = pickle.load(f)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        calorie_model = None
else:
    print(f"❌ Model file not found at: {model_path}")
    calorie_model = None

# Root route to check if API is running
@app.get("/")
def home():
    return {"message": "Calorie Burn API is running! Use /calorie_burn for predictions."}

# Prediction endpoint
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
