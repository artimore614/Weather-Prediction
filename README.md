# 🌦️ Seattle Weather Prediction

A machine-learning Streamlit app that predicts a weather category from user-entered Seattle weather measurements.

## Live App
After deployment, add your Streamlit Community Cloud URL here.

## Features
- Enter precipitation, maximum/minimum temperature, and wind speed
- Choose a date for the prediction
- Predict a weather category using the trained model
- Simple, interactive Streamlit interface

## Project Files
- `app.py` — Streamlit application
- `weather_prediction_model.pkl` — trained model
- `requirements.txt` — Python dependencies
- `cleaning.ipynb`, `EDA.ipynb`, `feature_selection.ipynb`, `model_selection.ipynb` — project notebooks
- `seattle-weather.csv` — dataset (if included in this repository)

## Run Locally
Install dependencies:
```bash
pip install -r requirements.txt
```

Start the app:
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose repository `artimore614/Weather-Prediction`, branch `main`, and main file `app.py`.
4. Click **Deploy** and wait for the app to build.

The app uses a trained model and user-provided inputs; it does not fetch live weather forecasts.

## Technologies
Python · Pandas · Scikit-learn · Joblib · Streamlit

## Author
**Arti More** · [GitHub](https://github.com/artimore614) · [LinkedIn](https://www.linkedin.com/in/arti-more-158b1734a)
