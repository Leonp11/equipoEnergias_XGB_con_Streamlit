import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


st.markdown(
    """
    <style>
    /* Centrar verticalmente el contenido del sidebar */
    .css-1d391kg {  /* contenedor principal del sidebar */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* Aumentar tamaño de los botones/flechitas y cambiar color */
    .css-1v3fvcr button {
        transform: scale(4);  /* 4 veces más grande */
        color: yellow;         /* color de texto (flechas) */
        margin: 10px 0;
    }

    /* Ajustar padding para que el centro se vea bien */
    .css-1d391kg > div {
        padding-top: 150px;
    }

    /* Opcional: que el hover también mantenga el color amarillo */
    .css-1v3fvcr button:hover {
        color: yellow;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# -----------------------------
# Configuración página
# -----------------------------
st.set_page_config(page_title="Predicción Demanda Eléctrica", layout="wide")

# -----------------------------
# Sidebar con menú
# -----------------------------
opcion = st.sidebar.radio("Navegación", ["Predicción", "Datos / EDA"])

# -----------------------------
# Ruta del modelo y datos
# -----------------------------
BASE_DIR = Path().resolve()
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "dataset_completo.csv"

# Carga del modelo
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"No se encontró el modelo en: {MODEL_PATH}")

# -----------------------------
# Pantalla según opción seleccionada
# -----------------------------
if opcion == "Predicción":
    st.title("⚡ Predicción de Demanda Eléctrica")

    # --- Inputs usuario ---
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

    # --- Crear DataFrame ---
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

    # --- Alinear columnas con el modelo ---
    if 'model' in locals():
        for col in model.feature_names_in_:
            if col not in X_input.columns:
                X_input[col] = 0.0
        X_input = X_input[model.feature_names_in_]

    # --- Predicción ---
    if st.button("🔮 Predecir demanda") and 'model' in locals():
        pred = model.predict(X_input)[0]
        st.success(f"📈 Demanda estimada: **{pred:,.0f} MW**")


elif opcion == "Datos / EDA":
    st.title("📊 Datos y Análisis Exploratorio (EDA)")

    # --- Cargar dataset ---
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['fecha'])
        st.success("✅ Dataset cargado correctamente")
        st.write(df.head())
    except FileNotFoundError:
        st.error(f"No se encontró el dataset en: {DATA_PATH}")

    # --- Ejemplo gráfico simple ---
    if 'df' in locals():
        st.subheader("Serie temporal de la demanda")
        plt.figure(figsize=(12, 4))
        plt.plot(df['fecha'], df['demanda_real'], color='royalblue')
        plt.xlabel("Fecha")
        plt.ylabel("Demanda (MW)")
        st.pyplot(plt)
