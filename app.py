import streamlit as st
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Calorie Burnt Prediction",
    layout="wide",
    page_icon="🔥"
)

# Load the saved model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = 'saved_models/calorie_model.sav'
# Debugging: Print the model path
print(f"Loading model from: {model_path}")

try:
    calorie_model = pickle.load(model_path)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'calorie_model.sav' exists in the 'saved_models' directory.")
    st.stop()

# Page title
st.title('Calorie Burnt Prediction')

# Getting the input data from the user
col1, col2 = st.columns(2)

with col1:
    # Gender input
    gender = st.selectbox('Gender', ('Male', 'Female'))
    gender = 1 if gender == 'Male' else 0

    # Age input
    age = st.number_input('Age', min_value=1, max_value=120, value=25)

    # Height input with unit selection
    height_unit = st.selectbox('Height Unit', ('cm', 'feet'))
    height = st.number_input('Height', min_value=0.0, value=170.0)
    if height_unit == 'feet':
        height = height * 30.48  # Convert feet to cm

    # Weight input with unit selection
    weight_unit = st.selectbox('Weight Unit', ('kg', 'lbs'))
    weight = st.number_input('Weight', min_value=0.0, value=70.0)
    if weight_unit == 'lbs':
        weight = weight * 0.453592  # Convert lbs to kg

with col2:
    # Duration input in minutes
    duration = st.number_input('Duration (minutes)', min_value=0, value=30)

    # Heart Rate input with validation
    heart_rate = st.number_input('Heart Rate (bpm)', min_value=67, max_value=128, value=80)

    # Body Temperature input with unit selection
    temp_unit = st.selectbox('Body Temperature Unit', ('Celsius', 'Fahrenheit'))
    body_temp = st.number_input('Body Temperature', min_value=0.0, value=37.0)
    if temp_unit == 'Fahrenheit':
        body_temp = (body_temp - 32) * 5/9  # Convert Fahrenheit to Celsius

# Code for Prediction
calorie_diagnosis = ''

# Creating a button for Prediction
if st.button('Calorie Burnt Test Result'):
    try:
        # Prepare user input for the model
        user_input = [gender, age, height, weight, duration, heart_rate, body_temp]
        user_input = np.asarray(user_input).reshape(1, -1)

        # Make prediction using the model
        calorie_prediction = calorie_model.predict(user_input)

        # Display result based on prediction
        calorie_diagnosis = f'The estimated calories burnt are: {calorie_prediction[0]:.2f} kcal'

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.success(calorie_diagnosis)