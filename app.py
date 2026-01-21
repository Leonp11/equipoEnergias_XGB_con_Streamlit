if seccion == "Predicción" and 'model' in locals():
    st.title("⚡ Predicción de Demanda Eléctrica")
    st.subheader("Introduce los valores")

    # -----------------------------
    # INPUTS DE DEMANDA
    # -----------------------------
    col1, col2 = st.columns([3, 1])
    with col1:
        demanda_lag_1 = st.text_input("Demanda hace 1 hora", placeholder="ej. 28000")
    with col2:
        st.markdown("MW")

    col1, col2 = st.columns([3, 1])
    with col1:
        demanda_lag_24 = st.text_input("Demanda hace 24 horas", placeholder="ej. 27500")
    with col2:
        st.markdown("MW")

    col1, col2 = st.columns([3, 1])
    with col1:
        demanda_lag_168 = st.text_input("Demanda hace 168 horas", placeholder="ej. 26000")
    with col2:
        st.markdown("MW")

    col1, col2 = st.columns([3, 1])
    with col1:
        media_movil_24h = st.text_input("Media móvil 24h", placeholder="ej. 27000")
    with col2:
        st.markdown("MW")

    # -----------------------------
    # Validación de inputs
    # -----------------------------
    try:
        demanda_lag_1 = float(demanda_lag_1)
        demanda_lag_24 = float(demanda_lag_24)
        demanda_lag_168 = float(demanda_lag_168)
        media_movil_24h = float(media_movil_24h)
    except ValueError:
        st.warning("⚠️ Introduce valores numéricos válidos")
        st.stop()

    # -----------------------------
    # HORAS, MES, FIN DE SEMANA
    # -----------------------------
    hora = st.slider("Hora del día", 0, 23, 18)
    mes = st.slider("Mes", 1, 12, 1)
    es_finde = st.selectbox("¿Es fin de semana?", ["Sí", "No"])
    dia_semana = st.slider("Día de la semana (0=Lunes)", 0, 6, 2)

    # -----------------------------
    # TEMPERATURAS POR REGIÓN
    # -----------------------------
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
        "es_finde": es_finde,
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
