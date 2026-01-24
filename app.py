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
# PARTE 1: Inputs de demanda con sliders destacados
# -----------------------------
st.markdown("<h1>Predicción de Demanda Eléctrica ⚡</h1>", unsafe_allow_html=True)

# Función para determinar color según valor MW (rangos exactos)
def color_por_demanda(val):
    if 24000 <= val <= 31000:
        return "#2ecc71"  # verde
    elif 31001 <= val <= 36000:
        return "#f1c40f"  # amarillo
    elif 36001 <= val <= 41000:
        return "#e67e22"  # naranja
    else:  # 40001-50000
        return "#e74c3c"  # rojo

# Función para mostrar slider con bloque coloreado según valor
def demanda_slider_coloreada(label, valor_inicial=27000, min_val=24000, max_val=47000):
    col_slider, col_val = st.columns([3,1])
    with col_slider:
        val = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=valor_inicial,
            step=100
        )
    color_actual = color_por_demanda(val)
    with col_val:
        # Mostrar valor con bloque de color más compacto
        st.markdown(
            f"""
            <div style="
                background-color:{color_actual};
                color:black;
                padding:3px 10px;
                border-radius:5px;
                font-weight:bold;
                font-size:14px;
                text-align:center;
                width:90px;
            ">
                {val:,} MW
            </div>
            """,
            unsafe_allow_html=True
        )
    return val

# Crear bloque principal con ancho 3/4 de la página
st.markdown(
    """
    <div style="
        background-color:#f39f18;
        padding:15px;
        border-radius:10px;
        width:75%;
        margin-bottom:20px;
    ">
    """,
    unsafe_allow_html=True
)

# Reducir tamaño de todos los sliders
st.markdown("""
<style>
div[data-baseweb="slider"] {
    width: 70% !important;
}
</style>
""", unsafe_allow_html=True)

# Sliders
demanda_lag_1 = demanda_slider_coloreada("Demanda hace 1 hora", 27000)
demanda_lag_24 = demanda_slider_coloreada("Demanda hace 24 horas", 27000)
demanda_lag_168 = demanda_slider_coloreada("Demanda hace 168 horas", 27000)
media_movil_24h = demanda_slider_coloreada("Media móvil 24h", 27000)

# Cerrar bloque visual
st.markdown("</div>", unsafe_allow_html=True)


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
