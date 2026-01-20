#-----------------------------------------
# Imports
#-----------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

#-----------------------------------------
# Configuración de la página
#-----------------------------------------
st.set_page_config(
    page_title="Predicción Demanda Eléctrica",
    layout="wide"
)

#-----------------------------------------
# Sidebar: selector de página
#-----------------------------------------
page = st.sidebar.selectbox("Selecciona una sección:", ["Predicción", "Exploración de Datos"])

#-----------------------------------------
# Ruta base y carga de datos/modelo
#-----------------------------------------
BASE_DIR = Path().resolve()
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset_completo.csv"

# Carga modelo
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"No se encontró el modelo en: {MODEL_PATH}")
    model = None

# Carga datos
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"No se encontró el dataset en: {DATA_PATH}")
    df = None

#-----------------------------------------
# Página: Predicción
#-----------------------------------------
if page == "Predicción" and model is not None:
    st.title("⚡ Predicción de Demanda Eléctrica")

    # Inputs del usuario
    st.subheader("Introduce los valores")

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

    # Crear DataFrame de entrada
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

    # Alineación robusta con el modelo
    for col in model.feature_names_in_:
        if col not in X_input.columns:
            X_input[col] = 0.0
    X_input = X_input[model.feature_names_in_]

    # Botón de predicción
    if st.button("🔮 Predecir demanda"):
        pred = model.predict(X_input)[0]
        st.success(f"📈 Demanda estimada: **{pred:,.0f} MW**")

#-----------------------------------------
# Página: Exploración de Datos
#-----------------------------------------
elif page == "Exploración de Datos" and df is not None:
    st.title("📊 Exploración de Datos")

    st.write("Dimensiones del dataset:", df.shape)
    st.dataframe(df.head())

    st.subheader("Visualización de la Demanda Real")
    if 'demanda_real' in df.columns and 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
        plt.figure(figsize=(12, 5))
        plt.plot(df['fecha'], df['demanda_real'], color='royalblue', linewidth=0.7)
        plt.xlabel("Fecha")
        plt.ylabel("Demanda (MW)")
        plt.title("Serie Temporal: Demanda Eléctrica")
        st.pyplot(plt.gcf())
        plt.clf()

    st.subheader("Mapa de calor horario vs día de la semana")
    if 'demanda_real' in df.columns:
        df['hora'] = df['fecha'].dt.hour
        df['dia_semana'] = df['fecha'].dt.dayofweek
        pivot_table = df.pivot_table(values='demanda_real', index='dia_semana', columns='hora', aggfunc='mean')
        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot_table, cmap='viridis', cbar_kws={'label': 'MW'})
        plt.xlabel("Hora")
        plt.ylabel("Día de la semana")
        st.pyplot(plt.gcf())
        plt.clf()
