#-----------------------------------------
# IMPORTS
#-----------------------------------------
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Predicción Demanda Eléctrica",
    layout="centered"
)

#-----------------------------------------
# Ruta del modelo
#-----------------------------------------
BASE_DIR = Path().resolve()  # raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Modelo cargado correctamente")
except FileNotFoundError:
    st.error(f"❌ No se encontró el modelo en: {MODEL_PATH}")

# --------------------------------
# SIDEBAR: Selección de sección
# --------------------------------
st.sidebar.title("Menú")
seccion = st.sidebar.radio("Selecciona sección", ["Predicción", "EDA"])

# -----------------------------
# SECCIÓN PREDICCIÓN
# -----------------------------
if seccion == "Predicción" and 'model' in locals():
    st.title("⚡ Predicción de Demanda Eléctrica")
    st.subheader("Introduce los valores")

    # Demanda hace 1 hora
    col1, col2 = st.columns([1, 0.3])
    with col1:
        demanda_lag_1_input = st.text_input("Demanda hace 1 hora (MW)", value="", max_chars=6)
    with col2:
        st.markdown("Ej: 27000")

    # Demanda hace 24 horas
    col1, col2 = st.columns([1, 0.3])
    with col1:
        demanda_lag_24_input = st.text_input("Demanda hace 24 horas (MW)", value="", max_chars=6)
    with col2:
        st.markdown("Ej: 27000")

    # Demanda hace 168 horas
    col1, col2 = st.columns([1, 0.3])
    with col1:
        demanda_lag_168_input = st.text_input("Demanda hace 168 horas (MW)", value="", max_chars=6)
    with col2:
        st.markdown("Ej: 27000")

    # Media móvil 24h
    col1, col2 = st.columns([1, 0.3])
    with col1:
        media_movil_24h_input = st.text_input("Media móvil 24h (MW)", value="", max_chars=6)
    with col2:
        st.markdown("Ej: 27000")

    # -----------------------------
    # Conversión segura a float
    # -----------------------------
    def to_float(value, default=27000.0):
        try:
            return float(value)
        except:
            return default

    demanda_lag_1 = to_float(demanda_lag_1_input)
    demanda_lag_24 = to_float(demanda_lag_24_input)
    demanda_lag_168 = to_float(demanda_lag_168_input)
    media_movil_24h = to_float(media_movil_24h_input)

    hora = input_con_ejemplo("Hora del día (0-23)", 18, suffix="h")
    mes = input_con_ejemplo("Mes", 1)
    
    # Inputs tipo select
    es_finde = st.selectbox("¿Es fin de semana?", ["Sí", "No"])
    es_finde_num = 1 if es_finde == "Sí" else 0
    dia_semana = input_con_ejemplo("Día de la semana (0=Lunes)", 2)

    st.markdown("### 🌡️ Temperaturas por región")
    temp_mad = input_con_ejemplo("Región Central", 30)
    temp_val = input_con_ejemplo("Región Sureste", 29)
    temp_pv = input_con_ejemplo("Región Norte", 22)
    temp_cat = input_con_ejemplo("Región Noreste", 28)
    temp_and = input_con_ejemplo("Región Sur", 33)

    # -----------------------------
    # DataFrame para el modelo
    # -----------------------------
    X_input = pd.DataFrame([{
        "demanda_lag_1": demanda_lag_1,
        "demanda_lag_24": demanda_lag_24,
        "demanda_lag_168": demanda_lag_168,
        "media_movil_24h": media_movil_24h,
        "hora": hora,
        "mes": mes,
        "es_finde": es_finde_num,
        "dia_semana": dia_semana,
        "Madrid_temperature_2m": temp_mad,
        "Valencia_temperature_2m": temp_val,
        "Pais_Vasco_temperature_2m": temp_pv,
        "Cataluna_temperature_2m": temp_cat,
        "Andalucia_temperature_2m": temp_and
    }])

    # -----------------------------
    # Alineación con columnas del modelo
    # -----------------------------
    for col in model.feature_names_in_:
        if col not in X_input.columns:
            X_input[col] = 0.0
    X_input = X_input[model.feature_names_in_]

    # -----------------------------
    # Predicción
    # -----------------------------
    if st.button("Calcular"):
        pred = model.predict(X_input)[0]
        st.success(f"📈 La predicción de demanda real es de **{pred:,.0f} MW**")

# -----------------------------
# SECCIÓN EDA
# -----------------------------
if seccion == "EDA":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.info("Aquí podrás cargar y visualizar datos del proyecto, agregar gráficas y resúmenes estadísticos.")
