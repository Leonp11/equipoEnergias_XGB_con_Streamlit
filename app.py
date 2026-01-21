# -----------------------------
# PARTE 1: Inputs de demanda y variables
# -----------------------------
st.title("⚡ Predicción de Demanda Eléctrica")
st.subheader("Introduce los valores")

# Función para crear input seguro con ejemplo al lado
def float_input_safe(label, ejemplo=27000.0, suffix="MW"):
    val_str = st.text_input(
        f"{label} ({suffix})", 
        value="", 
        max_chars=10, 
        key=label
    )
    try:
        val = float(val_str)
    except:
        val = ejemplo
    # Mostramos ejemplo al lado, centrado y en color tenue
    st.markdown(
        f"<div style='text-align:center; color:gray; font-size:14px;'>Ej.: {ejemplo}</div>", 
        unsafe_allow_html=True
    )
    return val

# Inputs de demanda
demanda_lag_1 = float_input_safe("Demanda hace 1 hora", 27000)
demanda_lag_24 = float_input_safe("Demanda hace 24 horas", 27000)
demanda_lag_168 = float_input_safe("Demanda hace 168 horas", 27000)
media_movil_24h = float_input_safe("Media móvil 24h", 27000)

# Hora y mes
hora = float_input_safe("Hora del día (0-23)", 18, suffix="h")
mes = float_input_safe("Mes", 1)

# Inputs tipo select
es_finde = st.selectbox("¿Es fin de semana?", ["Sí", "No"])
es_finde_num = 1 if es_finde == "Sí" else 0
dia_semana = float_input_safe("Día de la semana (0=Lunes)", 2)

# Temperaturas por región
st.markdown("### 🌡️ Temperaturas por región")
temp_mad = float_input_safe("Región Central", 30)
temp_val = float_input_safe("Región Sureste", 29)
temp_pv = float_input_safe("Región Norte", 22)
temp_cat = float_input_safe("Región Noreste", 28)
temp_and = float_input_safe("Región Sur", 33)


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
