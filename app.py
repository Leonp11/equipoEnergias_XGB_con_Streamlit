# -----------------------------------------
# Los IMPORTS
# -----------------------------------------

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import random

BASE_DIR = Path().resolve()  # raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Predicción Demanda Eléctrica",
    layout="centered"
)

# --------------------------------
# SIDEBAR: Selección de sección
# --------------------------------
st.sidebar.title("Menú")
seccion = st.sidebar.radio("Selecciona sección", ["Predicción", "EDA"])

# -----------------------------
# PARTE 1: Inputs de demanda
# -----------------------------
st.markdown("<h1>Predicción de Demanda Eléctrica ⚡</h1>", unsafe_allow_html=True)
st.markdown("<h3>Introduce los valores</h3>", unsafe_allow_html=True)

def float_input_safe(label, ejemplo=27000):
    col_input, col_ej = st.columns([0.2, 0.4])

    with col_input:
        val_str = st.text_input(
            f"{label} (MW)", 
            value="", 
            max_chars=10,
            key=label,
            help=f"Ej. {ejemplo}"
        )

        try:
            val_clean = float(val_str.replace(".", "").replace(",", "")) if val_str else ejemplo
        except:
            val_clean = ejemplo

    with col_ej:
        st.write("")

    return val_clean

demanda_lag_1 = float_input_safe("Demanda hace 1 hora")
demanda_lag_24 = float_input_safe("Demanda hace 24 horas")
demanda_lag_168 = float_input_safe("Demanda hace 168 horas")
media_movil_24h = float_input_safe("Media móvil 24h")

# -----------------------------
# BLOQUE: Hora del día
# -----------------------------
col1, col2 = st.columns([2,1])

with col1:
    hora_real = st.slider(
        "Hora del día",
        min_value=0,
        max_value=23,
        value=18,
        step=1
    )

    icono = "☀️" if 6 <= hora_real <= 18 else "🌙"

    # Mensaje de hora dentro del mismo bloque
    st.markdown(
        f"""
        <div style="
            margin-top:5px;
            margin-bottom:20px;
            font-weight:bold;
            font-size:18px;
            color:#f39f18;
        ">
            Hora seleccionada: {hora_real}h {icono}
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# BLOQUE: Día de la semana
# -----------------------------
dias_semana_nombres = {
    "Lunes": 1,
    "Martes": 2,
    "Miércoles": 3,
    "Jueves": 4,
    "Viernes": 5,
    "Sábado": 6,
    "Domingo": 7
}

with col1:
    dia_nombre = st.selectbox(
        "Día de la semana",
        list(dias_semana_nombres.keys()),
        index=2
    )

dia_semana = dias_semana_nombres[dia_nombre]
es_finde_num = 1 if dia_semana in [6, 7] else 0
es_finde_texto = "Sí" if es_finde_num == 1 else "No"

st.markdown(
    f"<div style='margin-top:5px; font-weight:bold; font-size:16px;'>"
    f"Día seleccionado: {dia_nombre}</div>",
    unsafe_allow_html=True
)
st.markdown(
    f"<div style='margin-bottom:10px; font-weight:bold; font-size:16px;'>"
    f"Es fin de semana: {es_finde_texto}</div>",
    unsafe_allow_html=True
)

# -----------------------------
# MES + ESTACIÓN DEL AÑO
# -----------------------------
meses = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
    "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}
col1, col2 = st.columns([0.2,0.4])
with col1:
    mes_nombre = st.selectbox("Mes", list(meses.keys()))
    mes = meses[mes_nombre]

if mes in [12, 1, 2]:
    estacion = "❄️ Invierno"
elif mes in [3, 4, 5]:
    estacion = "🌱 Primavera"
elif mes in [6, 7, 8]:
    estacion = "☀️ Verano"
else:
    estacion = "🍂 Otoño"
st.markdown(f"<div style='margin-top:5px; margin-bottom:15px; font-weight:bold; font-size:16px;'>{estacion}</div>", unsafe_allow_html=True)

# -----------------------------
# TEMPERATURA SEGÚN REGIÓN
# -----------------------------
st.markdown("<h3>Temperaturas por región 🌡️ </h3>", unsafe_allow_html=True)
temp_valores = list(range(-15, 49))
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
for col in model.feature_names_in_:
    if col not in X_input.columns:
        X_input[col] = 0.0
X_input = X_input[model.feature_names_in_]

# -----------------------------
# Cargar dataset histórico para comparación
# -----------------------------
HIST_PATH = BASE_DIR / "data" / "processed" / "dataset_consulta.csv"
try:
    df_hist = pd.read_csv(HIST_PATH)
    df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
    df_hist["dia_semana"] = df_hist["fecha"].dt.weekday + 1
except FileNotFoundError:
    st.error(f"No se encontró el dataset histórico en: {HIST_PATH}")
    df_hist = pd.DataFrame()

anos_disponibles = df_hist["year"].unique() if not df_hist.empty else []

# -----------------------------
# Predicción + comparación fija
# -----------------------------
if st.button("Calcular"):
    # --- Bloque verde: predicción ---
    pred = model.predict(X_input)[0]
    st.markdown(
        f"""
        <div style="
            background-color:#d4edda;
            color:#155724;
            padding:10px 20px;
            border-radius:5px;
            text-align:center;
        ">
            <div style="font-size:18px; font-weight:normal;">La predicción de demanda real es de:</div>
            <div style="font-size:28px; font-weight:bold;">{pred:,.0f} MW</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Bloque amarillo: comparación con años fijos ---
    if not df_hist.empty:
        for año in [2022, 2024]:
            comparacion = df_hist[
                (df_hist["year"] == año) &
                (df_hist["mes"] == mes) &
                (df_hist["dia_semana"] == dia_semana) &
                (df_hist["hora"] == hora_real)
            ]
            if not comparacion.empty:
                valor_real = comparacion["demanda_real"].values[0]
                st.markdown(
                    f"""
                    <div style="
                        background-color:#fff3cd;
                        color:#856404;
                        padding:8px 15px;
                        border-radius:5px;
                        margin-bottom:5px;
                    ">
                        En esta fecha y hora del año {año} la demanda real fue de {valor_real:,.0f} MW
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#fff3cd;
                        color:#856404;
                        padding:8px 15px;
                        border-radius:5px;
                        margin-bottom:5px;
                    ">
                        En esta fecha y hora del año {año} no hay datos disponibles.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# -----------------------------
# SECCIÓN EDA
# -----------------------------
if seccion == "EDA":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.info("inserte aquí verborrea y grafiquitos.")
