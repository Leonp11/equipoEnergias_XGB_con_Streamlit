#-----------------------------------------
# Los IMPORTS
#-----------------------------------------
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Predicción Demanda Eléctrica",
    layout="centered"
)

st.title("⚡ Predicción de Demanda Eléctrica")

#-----------------------------------------
# Ruta del modelo
#-----------------------------------------
# Usamos la carpeta raíz desde donde se ejecuta Streamlit
BASE_DIR = Path().resolve()  # raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"

# Cargamos el modelo
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Modelo cargado correctamente")
except FileNotFoundError:
    st.error(f"❌ No se encontró el modelo en: {MODEL_PATH}")

# -----------------------------
# INPUTS USUARIO
# -----------------------------
st.subheader("Introduce los valores")

demanda_lag_1 = st.number_input("Demanda hace 1 hora (MW)", value=28000.0)
demanda_lag_24 = st.number_input("Demanda hace 24 horas (MW)", value=27500.0)
demanda_lag_168 = st.number_input("Demanda hace 168 horas (MW)", value=26000.0)
media_movil_24h = st.number_input("Media móvil 24h (MW)", value=27000.0)

hora = st.slider("Hora del día", 0, 23, 18)
mes = st.slider("Mes", 1, 12, 1)

es_finde = st.selectbox("¿Es fin de semana?", [0, 1])
dia_semana = st.slider("Día de la semana (0=Lunes)", 0, 6, 2)

temp = st.number_input("Temperatura Madrid (ºC)", value=8.5)
temp_aparente = st.number_input("Temperatura aparente (ºC)", value=7.0)
temp_suelo = st.number_input("Temperatura suelo (ºC)", value=6.2)

# -----------------------------
# DATAFRAME PARA EL MODELO
# -----------------------------
X_input = pd.DataFrame([{
    "demanda_lag_1": demanda_lag_1,
    "demanda_lag_24": demanda_lag_24,
    "demanda_lag_168": demanda_lag_168,
    "media_movil_24h": media_movil_24h,
    "hora": hora,
    "mes": mes,
    "es_finde": es_finde,
    "dia_semana": dia_semana,
    "Madrid_temperature_2m": temp,
    "Madrid_apparent_temperature": temp_aparente,
    "Madrid_soil_temperature_0_to_7cm": temp_suelo
}])

# Orden seguro
if 'model' in locals():
    X_input = X_input[model.feature_names_in_]

# -----------------------------
# PREDICCIÓN
# -----------------------------
if st.button("🔮 Predecir demanda") and 'model' in locals():
    pred = model.predict(X_input)[0]
    st.success(f"📈 Demanda estimada: **{pred:,.0f} MW**")

