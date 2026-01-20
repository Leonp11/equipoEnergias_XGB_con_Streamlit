#-----------------------------------------
# Los IMPORTS
#-----------------------------------------
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(
    page_title="Predicción Demanda Eléctrica",
    layout="centered"
)

st.title("⚡ Predicción de Demanda Eléctrica")

#-----------------------------------------
# Ruta del modelo
#-----------------------------------------
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
st.subheader("Introduce los valores históricos y climáticos")

demanda_lag_1 = st.number_input("Demanda hace 1 hora (MW)", value=28000.0)
demanda_lag_24 = st.number_input("Demanda hace 24 horas (MW)", value=27500.0)
demanda_lag_168 = st.number_input("Demanda hace 168 horas (MW)", value=26000.0)
media_movil_24h = st.number_input("Media móvil 24h (MW)", value=27000.0)

hora = st.slider("Hora del día", 0, 23, 18)
mes = st.slider("Mes", 1, 12, 1)

es_finde = st.selectbox("¿Es fin de semana?", [0, 1])
dia_semana = st.slider("Día de la semana (0=Lunes)", 0, 6, 2)

st.markdown("### 🌡️ Temperaturas por región")
temp_mad = st.number_input("Madrid (ºC)", value=30.0)
temp_val = st.number_input("Valencia (ºC)", value=29.0)
temp_pv = st.number_input("País Vasco (ºC)", value=22.0)
temp_cat = st.number_input("Cataluña (ºC)", value=28.0)
temp_and = st.number_input("Andalucía (ºC)", value=33.0)

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
    "Madrid_temperature_2m": temp_mad,
    "Valencia_temperature_2m": temp_val,
    "Pais_Vasco_temperature_2m": temp_pv,
    "Cataluna_temperature_2m": temp_cat,
    "Andalucia_temperature_2m": temp_and
}])

# -----------------------------
# ALINEACIÓN ROBUSTA CON EL MODELO
# -----------------------------
if 'model' in locals():
    # Reindexa según las columnas exactas del modelo y rellena con 0.0 si falta alguna
    X_input = X_input.reindex(columns=model.feature_names_in_, fill_value=0.0)

# -----------------------------
# PREDICCIÓN
# -----------------------------
if st.button("🔮 Predecir demanda") and 'model' in locals():
    pred = model.predict(X_input)[0]
    st.success(f"📈 Demanda estimada: **{pred:,.0f} MW**")
