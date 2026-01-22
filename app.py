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
    col_input, col_ej = st.columns([0.2, 0.4])

    with col_input:
        val_str = st.text_input(
            f"{label} (MW)", 
            value="", 
            max_chars=10,
            key=label,
            help=f"Ej. {ejemplo}"  # ahora la guía está en el tooltip de ayuda
        )

        # Conversión segura a float
        try:
            val_clean = float(val_str.replace(".", "").replace(",", "")) if val_str else ejemplo
        except:
            val_clean = ejemplo

    with col_ej:
        # Columna vacía, ya no necesitamos la leyenda fuera de la caja
        st.write("")

    return val_clean

demanda_lag_1 = float_input_safe("Demanda hace 1 hora")
demanda_lag_24 = float_input_safe("Demanda hace 24 horas")
demanda_lag_168 = float_input_safe("Demanda hace 168 horas")
media_movil_24h = float_input_safe("Media móvil 24h")


import streamlit as st

# -----------------------------
# Slider interactivo de hora con color fijo azul cobalto y emoji
# -----------------------------

# Columnas para unificar ancho de sliders (2/1)
col1, col2 = st.columns([2,1])

with col1:
    # Slider de hora
    hora_real = st.slider(
        "Hora del día",
        min_value=0,
        max_value=23,
        value=18,  # valor por defecto = 6 PM
        step=1
    )

# Emoji dinámico según día/noche
icono = "☀️" if 6 <= hora_real <= 18 else "🌙"

# Color fijo del slider: Azul cobalto (#0047AB)
st.markdown(f"""
<style>
div[data-baseweb="slider"] input[type="range"] {{
    accent-color: #0047AB;
}}
</style>
""", unsafe_allow_html=True)

# Mostrar la hora seleccionada con emoji
st.markdown(f"<div style='margin-top:5px; margin-bottom:10px; color:#0047AB; font-weight:bold;'>Hora seleccionada: {hora_real}h {icono}</div>", unsafe_allow_html=True)


# -----------------------------
# Slider para el día de la semana (mismo tamaño)
# -----------------------------

dias_semana_nombres = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

with col1:
    dia_semana = st.slider(
        "Día de la semana",
        min_value=1,
        max_value=7,
        value=3,  # Por defecto Miércoles
        step=1
    )

# Nombre del día seleccionado
dia_nombre = dias_semana_nombres[dia_semana - 1]

# Calcular si es fin de semana
es_finde_num = 1 if dia_semana in [6, 7] else 0
es_finde_texto = "Sí" if es_finde_num == 1 else "No"

# Mostrar información del día
st.markdown(f"<div style='margin-top:5px; margin-bottom:5px; font-weight:bold;'>Día seleccionado: {dia_nombre}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='margin-bottom:10px;'>Es fin de semana: {es_finde_texto}</div>", unsafe_allow_html=True)


# -----------------------------
# MES + ESTACIÓN DEL AÑO
# -----------------------------

meses = {
    "Enero": 1,
    "Febrero": 2,
    "Marzo": 3,
    "Abril": 4,
    "Mayo": 5,
    "Junio": 6,
    "Julio": 7,
    "Agosto": 8,
    "Septiembre": 9,
    "Octubre": 10,
    "Noviembre": 11,
    "Diciembre": 12
}

mes_nombre = st.selectbox("Mes", list(meses.keys()))
mes = meses[mes_nombre]

# Cálculo de estación
if mes in [12, 0, 2]:
    estacion = "❄️ Invierno"
elif mes in [3, 4, 5]:
    estacion = "🌱 Primavera"
elif mes in [6, 7, 8]:
    estacion = "☀️ Verano"
else:
    estacion = "🍂 Otoño"

# Mostrar estación
st.markdown(f"<div style='margin-top:5px; margin-bottom:15px; font-weight:bold;'>Estación del año: {estacion}</div>", unsafe_allow_html=True)


st.markdown("### 🌡️ Temperaturas por región")

# Rango de temperaturas
temp_valores = list(range(-15, 49))  # -15 a 48ºC

# Columnas para hacer el layout compacto
col1, col2, col3 = st.columns(3)

with col1:
    temp_mad = st.selectbox("Región Central (ºC)", temp_valores, index=temp_valores.index(30))
    temp_val = st.selectbox("Región Sureste (ºC)", temp_valores, index=temp_valores.index(29))

with col2:
    temp_pv = st.selectbox("Región Norte (ºC)", temp_valores, index=temp_valores.index(22))
    temp_cat = st.selectbox("Región Noreste (ºC)", temp_valores, index=temp_valores.index(28))

with col3:
    temp_and = st.selectbox("Región Sur (ºC)", temp_valores, index=temp_valores.index(33))



# -----------------------------
# DataFrame para el modelo
# -----------------------------
X_input = pd.DataFrame([{
    "demanda_lag_1": demanda_lag_1,
    "demanda_lag_24": demanda_lag_24,
    "demanda_lag_168": demanda_lag_168,
    "media_movil_24h": media_movil_24h,
    "hora": hora_real,
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
