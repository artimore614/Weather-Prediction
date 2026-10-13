# 🌦️ Seattle Weather Prediction

A machine-learning web application that predicts a weather category from weather measurements using a trained model and an interactive Streamlit interface.

## 🚀 Live Demo

The app is deployed on **Streamlit Community Cloud**. Add the public app URL here once copied from the deployed app's Share menu.

## ✨ Features

- Predicts a weather category from user-entered measurements
- Takes precipitation, maximum temperature, minimum temperature, wind speed, and date as inputs
- Extracts year, month, and day from the selected date
- Uses a saved, trained machine-learning model for prediction
- Provides a simple web interface built with Streamlit

> **Note:** This project predicts from the inputs provided by the user. It does not fetch live weather data or provide a real-time forecast.

## 🧰 Tech Stack

- **Python** — application logic
- **Pandas** — input data preparation
- **Scikit-learn** — machine-learning model support
- **Joblib** — loading the saved model
- **Streamlit** — interactive web application

## 📁 Project Structure

```text
Weather-Prediction/
├── app.py
├── weather_prediction_model.pkl
├── requirements.txt
├── cleaning.ipynb
├── EDA.ipynb
├── feature_selection.ipynb
├── model_selection.ipynb
└── seattle-weather.csv
```

*Some notebook or dataset files may be present depending on the files committed to the repository.*

## ▶️ Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/artimore614/Weather-Prediction.git
   cd Weather-Prediction
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment

This project is configured for Streamlit Community Cloud.

- **Repository:** [artimore614/Weather-Prediction](https://github.com/artimore614/Weather-Prediction)
- **Branch:** `main`
- **Main file:** `app.py`

## 👩‍💻 Author

**Arti More**

- GitHub: [@artimore614](https://github.com/artimore614)
- LinkedIn: [Arti More](https://www.linkedin.com/in/arti-more-158b1734a)
