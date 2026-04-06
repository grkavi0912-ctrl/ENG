import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# --- This is a dummy comment to force re-upload after Dockerfile fix ---
# --- Adding another line to ensure content change detection ---
# --- Adding yet another line for version update ---
# --- Forcing another update to ensure commit detection ---
# --- And one more for good measure to ensure changes are always picked up ---

# Load model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="grkavi0912/ENG",
        filename="best_eng_model_v1.joblib",
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

# Streamlit UI for Engine Condition Prediction
st.title("Engine Condition Prediction")
st.write("Enter engine sensor values to predict its condition")

# Inputs based on the engine dataset
engine_rpm = st.number_input("Engine RPM", min_value=0, value=700)
lub_oil_pressure = st.number_input("Lub Oil Pressure (bar/kPa)", min_value=0.0, value=3.0)
fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, value=6.0)
coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, value=2.0)
lub_oil_temp = st.number_input("Lub Oil Temperature (°C)", min_value=0.0, value=75.0)
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, value=80.0)

# Create DataFrame (column names must match training data exactly)
input_df = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp
}])

# Prediction
if st.button("Predict Engine Condition"):
    try:
        prediction = model.predict(input_df)[0]

        if prediction == 0:
            result = "Engine is operating NORMALLY ✅"
            st.success(result)
        else:
            result = "Engine requires MAINTENANCE ⚠️"
            st.warning(result)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
