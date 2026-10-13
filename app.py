import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Page configuration
st.set_page_config(
    page_title="Seattle Weather Prediction",
    page_icon="🌦️",
    layout="wide"
)

# Load saved model
@st.cache_resource
def load_model():
    data = joblib.load("weather_prediction_model.pkl")
    return data["model"], data["features"]

try:
    model, feature_names = load_model()
except Exception as e:
    st.error(
        "Could not load the saved model. "
        "Check that weather_prediction_model.pkl "
        "is in the same folder as app.py."
    )
    st.exception(e)
    st.stop()

# Header
st.title("🌦️ Seattle Weather Prediction")
st.write(
    "Predict a weather category using your "
    "trained Machine Learning model."
)

st.divider()

# Input section
st.subheader("🌤️ Enter Weather Details")

col1, col2 = st.columns(2)

with col1:
    precipitation = st.number_input(
        "Precipitation (mm)",
        min_value=0.0,
        value=0.0,
        step=0.1
    )

    temp_max = st.number_input(
        "Maximum Temperature (°C)",
        value=15.0,
        step=0.5
    )

    temp_min = st.number_input(
        "Minimum Temperature (°C)",
        value=8.0,
        step=0.5
    )

with col2:
    wind = st.number_input(
        "Wind Speed",
        min_value=0.0,
        value=3.0,
        step=0.1
    )

    selected_date = st.date_input(
        "Date",
        value=date(2015, 6, 15)
    )

st.caption(
    "Enter weather measurements and a date "
    "to classify the weather."
)

# Predict
if st.button("🔍 Predict Weather", type="primary"):
    if temp_min > temp_max:
        st.error(
            "Minimum temperature cannot exceed "
            "maximum temperature."
        )
    else:
        # Prepare input in the model's required format
        input_data = pd.DataFrame([{
            "precipitation": precipitation,
            "temp_max": temp_max,
            "temp_min": temp_min,
            "wind": wind,
            "year": selected_date.year,
            "month": selected_date.month,
            "day": selected_date.day
        }])

        input_data = input_data[feature_names]

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("🎯 Prediction Result")

        st.success(f"Predicted Weather: {str(prediction).title()}")

        # Display entered measurements
        m1, m2, m3 = st.columns(3)

        m1.metric("Maximum Temp", f"{temp_max:.1f} °C")
        m2.metric("Minimum Temp", f"{temp_min:.1f} °C")
        m3.metric("Wind Speed", f"{wind:.1f}")

        st.info(
            "This is a machine learning classification "
            "based on your trained dataset, not a live "
            "weather forecast."
        )

st.divider()
st.caption("Project: Seattle Weather Prediction | Python + Scikit-learn + Streamlit")
