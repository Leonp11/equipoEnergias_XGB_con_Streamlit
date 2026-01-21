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
# PARTE 1: Inputs de demanda
# -----------------------------
st.title("⚡ Predicción de Demanda Eléctrica")
st.subheader("Introduce los valores")

def float_input_safe(label, ejemplo=27000):
    # Contenedor horizontal
    col_input, col_ej = st.columns([1, 0.4])

    with col_input:
        val_str = st.text_input(
            f"{label} (MW)", 
            value="", 
            max_chars=10,
            key=label,
            help=None  # Quita signo de ?
        )

        # Conversión segura a float
        try:
            val_clean = float(val_str.replace(".", "").replace(",", "")) if val_str else ejemplo
        except:
            val_clean = ejemplo

    with col_ej:
        # Leyenda Ej. al lado derecho
        st.markdown(
            f"<div style='color:gray; text-align:center; margin-top:10px;'>Ej.: {ejemplo:,}</div>",
            unsafe_allow_html=True
        )

    return val_clean

demanda_lag_1 = float_input_safe("Demanda hace 1 hora")
demanda_lag_24 = float_input_safe("Demanda hace 24 horas")
demanda_lag_168 = float_input_safe("Demanda hace 168 horas")
media_movil_24h = float_input_safe("Media móvil 24h")




# Inputs tipo slider / select
hora = st.number_input("Hora del día (0-23)", min_value=0, max_value=23, value=18, step=1)
mes = st.number_input("Mes", min_value=1, max_value=12, value=1, step=1)

es_finde = st.selectbox("¿Es fin de semana?", ["Sí", "No"])
es_finde_num = 1 if es_finde == "Sí" else 0

dia_semana = st.number_input("Día de la semana (0=Lunes)", min_value=0, max_value=6, value=2, step=1)

st.markdown("### 🌡️ Temperaturas por región")
temp_mad = st.number_input("Región Central (ºC)", value=30.0)
temp_val = st.number_input("Región Sureste (ºC)", value=29.0)
temp_pv = st.number_input("Región Norte (ºC)", value=22.0)
temp_cat = st.number_input("Región Noreste (ºC)", value=28.0)
temp_and = st.number_input("Región Sur (ºC)", value=33.0)

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
