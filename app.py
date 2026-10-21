import streamlit as st
import pandas as pd
import joblib
from datetime import date

st.set_page_config(page_title="Seattle Weather | ML Dashboard", page_icon="🌦️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: #f4f8fc; color: #172b4d; }
[data-testid="stHeader"] { background: rgba(244,248,252,.96); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#102a43 0%,#174a6e 100%); }
[data-testid="stSidebar"] * { color: #ffffff !important; }
.hero { padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(115deg,#123b5d,#176b87,#55a6b8); color: #fff; margin-bottom: 1.1rem; box-shadow: 0 8px 24px rgba(18,59,93,.15); }
.hero h1 { color: #fff !important; margin: 0; font-size: 2.15rem; }
.hero p { color: #eaf8ff !important; margin: .45rem 0 0; font-size: 1rem; }
h1,h2,h3,h4,p,li,label,.stMarkdown { color: #172b4d; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #d5e4f2; padding:15px 16px; border-radius:14px; box-shadow:0 3px 12px rgba(15,57,87,.06); }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color:#49627b !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#123b5d !important; font-weight:700; }
.section-card { background:#fff; color:#172b4d; padding:1rem 1.2rem; border:1px solid #d5e4f2; border-radius:15px; box-shadow:0 4px 14px rgba(15,57,87,.05); }
.section-card p,.section-card strong { color:#172b4d !important; }
.result-card { padding:1.25rem; border-radius:18px; background:#e7f8ef; border:1px solid #a7e3c0; color:#14532d; text-align:center; }
.result-card h2,.result-card p { color:#14532d !important; }
.small-note { color:#52677d !important; font-size:.9rem; }
div.stButton > button { width:100%; border-radius:12px; padding:.7rem 1rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# --- Login and signup (passwords are stored as salted hashes) ---
import sqlite3
import hashlib
import hmac
import secrets

DB_PATH = "weather_users.db"

def init_user_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)

def hash_password(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 200_000
    ).hex()

def create_user(username, password):
    username = username.strip()
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    salt = secrets.token_bytes(16).hex()
    password_hash = hash_password(password, salt)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                (username, salt, password_hash),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username already exists. Please choose another."

def verify_user(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return False
    salt, saved_hash = row
    return hmac.compare_digest(hash_password(password, salt), saved_hash)

init_user_db()

if "weather_user" not in st.session_state:
    st.session_state.weather_user = None

if not st.session_state.weather_user:
    st.markdown("""
    <div class="hero">
      <h1>🌦️ Seattle Weather Prediction</h1>
      <p>Sign in or create an account to access the weather prediction dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["🔐 Login", "✨ Sign Up"])

    with login_tab:
        with st.form("login_form"):
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Login", type="primary", use_container_width=True)
        if login_submit:
            if verify_user(login_username, login_password):
                st.session_state.weather_user = login_username.strip()
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with signup_tab:
        with st.form("signup_form"):
            signup_username = st.text_input("Choose a username", key="signup_username")
            signup_password = st.text_input("Create password (minimum 8 characters)", type="password", key="signup_password")
            signup_confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)
        if signup_submit:
            if signup_password != signup_confirm:
                st.error("Passwords do not match.")
            else:
                created, message = create_user(signup_username, signup_password)
                if created:
                    st.success(message)
                else:
                    st.error(message)
    st.info("Your password is stored as a salted hash, not as plain text.")
    st.stop()

@st.cache_resource
def load_model():
    saved = joblib.load("weather_prediction_model.pkl")
    return saved["model"], saved["features"]

try:
    model, feature_names = load_model()
except Exception as exc:
    st.error("Could not load the trained model. Make sure weather_prediction_model.pkl is in the same repository folder as app.py.")
    st.exception(exc)
    st.stop()

st.markdown("""
<div class="hero">
  <h1>🌦️ Seattle Weather Prediction</h1>
  <p>Interactive Machine Learning dashboard · Explore weather conditions and predict a category</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌤️ Weather Inputs")
    st.caption("Enter the conditions you want the model to classify.")
    selected_date = st.date_input("📅 Date", value=date(2015, 6, 15))
    precipitation = st.number_input("🌧️ Precipitation (mm)", min_value=0.0, value=0.0, step=0.1)
    temp_max = st.number_input("🌡️ Maximum temperature (°C)", value=15.0, step=0.5)
    temp_min = st.number_input("❄️ Minimum temperature (°C)", value=8.0, step=0.5)
    wind = st.number_input("💨 Wind speed", min_value=0.0, value=3.0, step=0.1)
    predict_clicked = st.button("🔍 Predict Weather", type="primary")
    st.markdown("---")
    st.caption("Trained ML model · Seattle weather dataset")

st.subheader("📊 Input Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Maximum Temperature", f"{temp_max:.1f} °C")
k2.metric("Minimum Temperature", f"{temp_min:.1f} °C")
k3.metric("Precipitation", f"{precipitation:.1f} mm")
k4.metric("Wind Speed", f"{wind:.1f}")

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown("### 🌡️ Temperature Range")
    chart_data = pd.DataFrame({"Temperature (°C)": [temp_min, temp_max]}, index=["Minimum", "Maximum"])
    st.bar_chart(chart_data, height=250)
    st.caption(f"Selected date: {selected_date.strftime('%d %B %Y')}")

with right:
    st.markdown("### 🧭 Conditions Summary")
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"**Date:** {selected_date.strftime('%d %b %Y')}")
    st.markdown(f"**Temperature difference:** {temp_max - temp_min:.1f} °C")
    st.markdown(f"**Precipitation:** {'Rainfall recorded' if precipitation > 0 else 'No precipitation entered'}")
    st.markdown(f"**Wind level:** {'High' if wind >= 6 else 'Moderate' if wind >= 3 else 'Low'}")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    if temp_min > temp_max:
        st.error("Minimum temperature cannot be greater than maximum temperature. Please correct the inputs in the sidebar.")
    else:
        input_data = pd.DataFrame([{
            "precipitation": precipitation, "temp_max": temp_max, "temp_min": temp_min,
            "wind": wind, "year": selected_date.year, "month": selected_date.month, "day": selected_date.day
        }])
        input_data = input_data[feature_names]
        prediction = str(model.predict(input_data)[0]).title()
        emoji = {"Rain":"🌧️","Sun":"☀️","Drizzle":"🌦️","Snow":"❄️","Fog":"🌫️"}.get(prediction,"🌤️")
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        st.markdown(f'<div class="result-card"><div style="font-size:2.5rem">{emoji}</div><h2>{prediction}</h2><p>Predicted weather category</p></div>', unsafe_allow_html=True)
        st.markdown("#### Prediction details")
        r1, r2, r3 = st.columns(3)
        r1.metric("Temperature Range", f"{temp_min:.1f}–{temp_max:.1f} °C")
        r2.metric("Precipitation", f"{precipitation:.1f} mm")
        r3.metric("Wind Speed", f"{wind:.1f}")
        with st.expander("View model input data"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)
        st.info("This is a prediction from your trained ML model, not a live weather forecast.")

st.markdown("---")
with st.expander("ℹ️ About this project"):
    st.write("A machine-learning classification dashboard using weather measurements and date features from the Seattle weather dataset.")
    st.write("**Tech stack:** Python · Pandas · Scikit-learn · Joblib · Streamlit")
st.markdown('<p class="small-note">Seattle Weather Prediction · Machine Learning Dashboard · Built with Streamlit</p>', unsafe_allow_html=True)
