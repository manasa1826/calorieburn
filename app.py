import streamlit as st
import pickle
import os
import numpy as np

# Title of the app
st.title("Calorie Burn Prediction App")

# Load the model
working_dir = os.getcwd()
model_path = os.path.join(working_dir, 'saved_models', 'calorie_model.sav')

try:
    # Load the pre-trained model
    calorie_model = pickle.load(open(model_path, 'rb'))
    st.success("Model loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input fields for user data
st.sidebar.header("Input Features")

# Define input fields based on your model's features
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=36.5)
activity_level = st.sidebar.number_input("Activity Level (1-5)", min_value=1, max_value=5, value=3)

# Create input data array
input_data = (age, gender, height, weight, heart_rate, body_temp, activity_level)

# Convert input data to numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
if st.sidebar.button("Predict Calorie Burn"):
    try:
        prediction = calorie_model.predict(input_data_reshaped)
        st.success(f"Predicted Calorie Burn: {prediction[0]:.2f} calories")
    except Exception as e:
        st.error(f"Error making prediction: {e}")