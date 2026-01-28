# -----------------------------------------
# Los IMPORTS
# -----------------------------------------
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# -----------------------------------------
# Rutas base
# -----------------------------------------
BASE_DIR = Path().resolve()  # raíz del proyecto
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
HIST_PATH = BASE_DIR / "data" / "processed" / "dataset_consulta.csv"

# -----------------------------------------
# Cargar modelo
# -----------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"No se encontró el modelo en: {MODEL_PATH}")
    model = None

# -----------------------------------------
# Configuración de la página
# -----------------------------------------
st.set_page_config(
    page_title="Predicción Demanda Eléctrica",
    layout="centered"
)


# -----------------------------------------
# Sidebar: Selección de sección
# -----------------------------------------
st.sidebar.title("Menú")
seccion = st.sidebar.radio("Selecciona sección", ["EDA", "Predicción"])

# -----------------------------------------
# Cargar dataset histórico
# -----------------------------------------
try:
    df_hist = pd.read_csv(HIST_PATH)
    df_hist["fecha"] = pd.to_datetime(df_hist["fecha"])
    df_hist["dia_semana"] = df_hist["fecha"].dt.weekday + 1
except FileNotFoundError:
    st.error(f"No se encontró el dataset histórico en: {HIST_PATH}")
    df_hist = pd.DataFrame()

# -----------------------------------------
# SECCIÓN: EDA
# -----------------------------------------
if seccion == "EDA":
    st.title("")
    st.markdown(
        "<h2 style='text-align:center;'>📊 Análisis Exploratorio de Datos (EDA) 📊</h2>",
        unsafe_allow_html=True
    )




    # ---------- 1. Problema de Negocio ----------
    st.subheader("1. El Problema de Negocio y el Contexto")
    st.markdown(
        """
        Decidimos abordar uno de los problemas más costosos y críticos del sector industrial: la predicción de la demanda eléctrica.  
        El sistema eléctrico no puede almacenar energía a gran escala; lo que se genera debe consumirse al instante.  
        El reto principal que enfrentamos no fue solo técnico, sino de comportamiento: la demanda eléctrica es el resultado de millones de decisiones humanas.
        """
    )
    img_path = BASE_DIR / "data" / "Images" / "01.png"
    st.image(img_path, use_column_width=True, caption="Contexto del problema")

    # ---------- 2. Estrategia de Datos ----------
    st.subheader("2. La Estrategia de Datos")
    st.markdown(
        """
        Al principio, planteamos la hipótesis de que la demanda dependía casi exclusivamente de la temperatura.  
        Sin embargo, al analizar los datos en profundidad, nos dimos cuenta de que las variables climáticas clásicas solo tenían una correlación moderada con el consumo real (alrededor de un 0.4).
        """
    )
    img_path = BASE_DIR / "data" / "Images" / "02.png"
    st.image(img_path, use_column_width=True, caption="Análisis inicial de correlación")

    # ---------- 3. Inercia temporal ----------
    st.subheader("3. La verdadera clave: entender la inercia temporal")
    st.markdown(
        """
        Llegamos a la conclusión de que el mejor predictor del consumo actual no es el clima, sino el pasado inmediato:  

        1. Cuánto consumimos hace una hora.  
        2. Cuánto consumimos ayer a esta misma hora.  
        3. Cuánto consumimos la semana pasada.  

        Por ello, construimos variables de 'Lags' o retardos temporales.
        """
    )
    img_path = BASE_DIR / "data" / "Images" / "03.png"
    st.image(img_path, use_column_width=True, caption="Variables de Lags construidas")

    # ---------- 4. Batalla de Modelos ----------
    st.subheader("4. La Batalla de Modelos: XGBoost vs N-BEATS")
    st.markdown(
        """
        Con los datos listos, llegamos a la fase de modelado. Decidimos poner a competir a nuestro modelo basado en árboles (XGBoost) contra una arquitectura de Deep Learning moderna (N-BEATS).  

        En las métricas, N-BEATS nos dio un R² negativo (-34). Fue incapaz de encontrar patrones estables con el volumen de datos disponible.  
        En contraste, nuestro modelo XGBoost alcanzó 0.99.
        """
    )
    img_path = BASE_DIR / "data" / "Images" / "04.png"
    st.image(img_path, use_column_width=True, caption="Comparativa de modelos")

    # ---------- 5. Validación y Resultados ----------
    st.subheader("5. Validación y Resultados")
    st.markdown(
        """
        Para evitar engañarnos con métricas de entrenamiento, diseñamos una validación temporal estricta.  
        Mantuvimos el R² superior a 0.99 y, visualmente, el modelo replicó perfectamente la dinámica diaria y las caídas de los fines de semana.
        """
    )
    img_path = BASE_DIR / "data" / "Images" / "05.png"
    st.image(img_path, use_column_width=True, caption="Validación temporal del modelo")

    # ---------- 6. Limitaciones y Observaciones ----------
    st.subheader("6. Limitaciones y Observaciones")
    st.markdown(
        """
        - **Dependencia del Dato Reciente:** Nuestro modelo depende mucho del dato de 'hace una hora'.  
        - **Eventos Imprevisibles:** Si ocurre una nueva pandemia o una crisis energética anómala, el modelo tardará en reaccionar.  
        - **Falta de Variables Económicas:** Actualmente miramos calendario y clima. Pero sabemos que el precio de la luz afecta al consumo industrial.  
        - **Filomena:** fue un outlier importante en los datos.
        """
    )

# -----------------------------------------
# SECCIÓN: Predicción
# -----------------------------------------
if seccion == "Predicción" and model is not None:
    st.markdown(
        "<h2 style='text-align:center;'>⚡ Predicción de Demanda Eléctrica ⚡</h2>",
        unsafe_allow_html=True
    )

    # Función para color de demanda
    def color_por_demanda(val):
        if 24000 <= val <= 31000:
            return "#2ecc71"
        elif 31001 <= val <= 36000:
            return "#f1c40f"
        elif 36001 <= val <= 41000:
            return "#e67e22"
        else:
            return "#e74c3c"

    # Función para slider coloreado
    def demanda_slider_coloreada(label, valor_inicial=27000, min_val=24000, max_val=47000):
        col_slider, col_val = st.columns([3,1])
        with col_slider:
            val = st.slider(label, min_value=min_val, max_value=max_val, value=valor_inicial, step=100)
        color_actual = color_por_demanda(val)
        with col_val:
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

    # Sliders principales
    demanda_lag_1 = demanda_slider_coloreada("Demanda hace 1 hora")
    demanda_lag_24 = demanda_slider_coloreada("Demanda hace 24 horas")
    demanda_lag_168 = demanda_slider_coloreada("Demanda hace 168 horas")
    media_movil_24h = demanda_slider_coloreada("Media móvil 24h")

    # Hora del día
    col1, col2 = st.columns([2,1])
    with col1:
        hora_real = st.slider("Hora del día", 0, 23, 18)
        icono = "☀️" if 6 <= hora_real <= 18 else "🌙"
        st.markdown(f"<div style='font-weight:bold; font-size:18px; color:#f39f18;'>Hora seleccionada: {hora_real}h {icono}</div>", unsafe_allow_html=True)

    # Día de la semana
    dias_semana_nombres = {"Lunes":1,"Martes":2,"Miércoles":3,"Jueves":4,"Viernes":5,"Sábado":6,"Domingo":7}
    col1, col2 = st.columns([0.2,0.4])
    with col1:
        dia_nombre = st.selectbox("Día de la semana", list(dias_semana_nombres.keys()), index=2)
    dia_semana = dias_semana_nombres[dia_nombre]
    es_finde_num = 1 if dia_semana in [6,7] else 0

    # Mes y estación
    meses = {"Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,"Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12}
    col1, col2 = st.columns([0.2,0.4])
    with col1:
        mes_nombre = st.selectbox("Mes", list(meses.keys()))
        mes = meses[mes_nombre]

    if mes in [12,1,2]:
        estacion = "❄️ Invierno"
    elif mes in [3,4,5]:
        estacion = "🌱 Primavera"
    elif mes in [6,7,8]:
        estacion = "☀️ Verano"
    else:
        estacion = "🍂 Otoño"
    st.markdown(f"<div style='font-weight:bold; font-size:16px;'>{estacion}</div>", unsafe_allow_html=True)

    # Temperaturas por región
    st.markdown("<h3>Temperaturas por región 🌡️ </h3>", unsafe_allow_html=True)
    temp_valores = list(range(-15,49))
    col1, col2, col3 = st.columns(3)
    with col1:
        temp_mad = st.selectbox("Región Central (ºC)", temp_valores, index=temp_valores.index(30))
        temp_val = st.selectbox("Región Sureste (ºC)", temp_valores, index=temp_valores.index(29))
    with col2:
        temp_pv = st.selectbox("Región Norte (ºC)", temp_valores, index=temp_valores.index(22))
        temp_cat = st.selectbox("Región Noreste (ºC)", temp_valores, index=temp_valores.index(28))
    with col3:
        temp_and = st.selectbox("Región Sur (ºC)", temp_valores, index=temp_valores.index(33))

    # DataFrame para predicción
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

    # Botón de cálculo
    if st.button("Calcular"):
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

        # Comparación con años fijos
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
