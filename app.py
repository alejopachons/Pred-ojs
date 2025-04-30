import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Importancia de Variables", layout="centered")

st.title("🔍 Análisis de Importancia de Variables usando RandomForestRegressor")
st.write("Sube un archivo Excel para analizar qué variables explican mejor tu variable objetivo.")

# 1. Cargar archivo
uploaded_file = st.file_uploader("📂 Sube tu archivo Excel (.xlsx o .xls)", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Vista previa de tus datos")
    st.dataframe(df)

    with st.form("form_variable_selection"):
        st.subheader("Selecciona las variables para el modelo")
        features = st.multiselect("Variables independientes (X)", df.columns.tolist())
        target = st.selectbox("Variable objetivo (Y)", df.columns.tolist())
        submitted = st.form_submit_button("Analizar")

    if submitted and features and target:
        try:
            X = df[features]
            y = df[target]

            # Entrenar modelo
            model = RandomForestRegressor(random_state=0)
            model.fit(X, y)

            # Importancia de variables
            importances = model.feature_importances_
            results = pd.DataFrame({
                "Variable": features,
                "Importancia": importances
            }).sort_values(by="Importancia", ascending=False)

            st.subheader("📊 Importancia de cada variable")
            st.dataframe(results)

            # Gráfico
            fig, ax = plt.subplots()
            sns.barplot(x="Importancia", y="Variable", data=results, ax=ax, palette="viridis")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Hubo un error al procesar los datos: {e}")
