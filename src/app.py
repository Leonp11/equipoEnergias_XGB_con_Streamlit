# -----------------------------------------
# IMPORTS
# -----------------------------------------
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# -----------------------------------------
# PATHS (FIX PARA RENDER)
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
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
# SIDEBAR
# --------------------------------
st.sidebar.title("Menú")
seccion = st.sidebar.radio("Selecciona sección", ["EDA", "Predicción"], index=0)

# ===========================
# SECCIÓN: EDA
# ===========================
if seccion == "EDA":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("<div style='margin-bottom:30px;'></div>", unsafe_allow_html=True)

    st.subheader("1. El Problema de Negocio y el Contexto")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)
    st.write("""
Decidimos abordar uno de los problemas más costosos y críticos del sector industrial: la predicción de la demanda eléctrica.
El sistema eléctrico no puede almacenar energía a gran escala; lo que se genera debe consumirse al instante.
El reto principal que enfrentamos no fue solo técnico, sino de comportamiento: la demanda eléctrica es el resultado de millones de decisiones humanas.
    """)
    st.image(BASE_DIR / "data" / "Images" / "01.png", use_column_width=True)

    st.subheader("2. La Estrategia de Datos")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)
    st.write("""
Al principio, planteamos la hipótesis de que la demanda dependía casi exclusivamente de la temperatura.
Sin embargo, al analizar los datos, vimos que la correlación era moderada (~0.4).
    """)
    st.image(BASE_DIR / "data" / "Images" / "02.png", use_column_width=True)

    st.subheader("3. La verdadera clave fue entender la inercia temporal")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)
    st.write("""
El mejor predictor del consumo actual es el pasado inmediato:
- Hace 1 hora
- Hace 24 horas
- Hace 7 días
    """)
    st.image(BASE_DIR / "data" / "Images" / "03.png", use_column_width=True)

    st.subheader("4. La Batalla de Modelos")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)
    st.write("""
XGBoost vs N-BEATS.
N-BEATS obtuvo un R² negativo, mientras XGBoost alcanzó 0.99.
    """)
    st.image(BASE_DIR / "data" / "Images" / "04.png", use_column_width=True)

    st.subheader("5. Validación y Resultados")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)
    st.write("""
Validación temporal estricta.
El modelo replica correctamente patrones diarios y fines de semana.
    """)
    st.image(BASE_DIR / "data" / "Images" / "05.png", use_column_width=True)

    st.subheader("6. Limitaciones y Observaciones")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)
    st.write("""
- Dependencia fuerte del dato reciente  
- Eventos imprevisibles  
- Falta de variables económicas  
- Filomena fue un outlier  
    """)

# ===========================
# SECCIÓN: PREDICCIÓN
# ===========================
if seccion == "Predicción":

    st.markdown(
        "<h1 style='text-align:center; font-size:32px; font-weight:bold; margin-bottom:30px;'>⚡ Predicción de Demanda Eléctrica ⚡</h1>",
        unsafe_allow_html=True
    )

    # --------------------------
    # LAGS
    # --------------------------
    st.subheader("📊 Demanda real anterior")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)

    def demanda_slider(label, value):
        return st.slider(label, 24000, 47000, value, step=100)

    demanda_lag_1 = demanda_slider("Demanda hace 1 hora", 27000)
    demanda_lag_24 = demanda_slider("Demanda hace 24 horas", 27000)
    demanda_lag_168 = demanda_slider("Demanda hace 7 días", 27000)
    media_movil_24h = demanda_slider("Media últimas 24 horas", 27000)

    # --------------------------
    # CONTEXTO TEMPORAL
    # --------------------------
    st.subheader("📅 Fecha")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)

    hora_real = st.slider("Hora del día", 0, 23, 18)

    dias = {"Lunes":1,"Martes":2,"Miércoles":3,"Jueves":4,"Viernes":5,"Sábado":6,"Domingo":7}
    dia_nombre = st.selectbox("Día de la semana", list(dias.keys()), index=2)
    dia_semana = dias[dia_nombre]
    es_finde = 1 if dia_semana in [6,7] else 0

    meses = {"Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,"Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12}
    mes_nombre = st.selectbox("Mes", list(meses.keys()))
    mes = meses[mes_nombre]

    # --------------------------
    # TEMPERATURAS
    # --------------------------
    st.subheader("🌡️ Temperaturas por Región")
    st.markdown("<div style='height:5px; background-color:#f39f18; width:50px; margin-bottom:15px;'></div>", unsafe_allow_html=True)

    temp = lambda l, v: st.selectbox(l, range(-15,49), index=range(-15,49).index(v))

    temp_mad = temp("Madrid", 30)
    temp_val = temp("Valencia", 29)
    temp_pv  = temp("País Vasco", 22)
    temp_cat = temp("Cataluña", 28)
    temp_and = temp("Andalucía", 33)

    # --------------------------
    # INPUT
    # --------------------------
    X_input = pd.DataFrame([{
        "demanda_lag_1": demanda_lag_1,
        "demanda_lag_24": demanda_lag_24,
        "demanda_lag_168": demanda_lag_168,
        "media_movil_24h": media_movil_24h,
        "hora": hora_real,
        "mes": mes,
        "es_finde": es_finde,
        "dia_semana": dia_semana,
        "Madrid_temperature_2m": temp_mad,
        "Valencia_temperature_2m": temp_val,
        "Pais_Vasco_temperature_2m": temp_pv,
        "Cataluna_temperature_2m": temp_cat,
        "Andalucia_temperature_2m": temp_and
    }])

    X_input = X_input[model.feature_names_in_]

    # --------------------------
    # HISTÓRICO
    # --------------------------
    HIST_PATH = BASE_DIR / "data" / "processed" / "dataset_consulta.csv"
    df_hist = pd.read_csv(HIST_PATH)
    df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
    df_hist["dia_semana"] = df_hist["fecha"].dt.weekday + 1

    # --------------------------
    # PREDICCIÓN
    # --------------------------
    if st.button("Calcular"):
        pred = model.predict(X_input)[0]
        st.success(f"Demanda estimada: {pred:,.0f} MW")
