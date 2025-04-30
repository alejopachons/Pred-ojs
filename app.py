import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Comparación de Modelos", layout="centered")

st.title("🔍 Comparación de Importancia de Variables entre Modelos")
st.write("Sube un archivo Excel para comparar modelos de regresión y su análisis de importancia de variables.")

# Cargar archivo
uploaded_file = st.file_uploader("📂 Sube tu archivo Excel (.xlsx o .xls)", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Vista previa de tus datos")
    st.dataframe(df)

    with st.form("form_variable_selection"):
        st.subheader("Selecciona las variables para el modelo")
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        features = st.multiselect("Variables independientes (X)", numeric_columns)
        target = st.selectbox("Variable objetivo (Y)", df.columns.tolist())
        submitted = st.form_submit_button("Comparar modelos")

    if submitted and features and target:
        try:
            X = df[features]
            y = df[target]

            # -------------------------
            # Entrenar RandomForest
            # -------------------------
            rf_model = RandomForestRegressor(random_state=0)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X)
            rf_importances = rf_model.feature_importances_

            # -------------------------
            # Entrenar LinearRegression
            # -------------------------
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            lr_pred = lr_model.predict(X)
            lr_coefficients = lr_model.coef_

            # -------------------------
            # Métricas
            # -------------------------
            rf_r2 = r2_score(y, rf_pred)
            lr_r2 = r2_score(y, lr_pred)

            rf_mae = mean_absolute_error(y, rf_pred)
            lr_mae = mean_absolute_error(y, lr_pred)

            st.subheader("📈 Comparación de desempeño (sobre datos de entrenamiento)")
            st.write(pd.DataFrame({
                "Modelo": ["Random Forest", "Linear Regression"],
                "R²": [rf_r2, lr_r2],
                "MAE": [rf_mae, lr_mae]
            }))

            # -------------------------
            # Comparar importancia
            # -------------------------
            st.subheader("📊 Importancia / Coeficientes de Variables")
            importance_df = pd.DataFrame({
                "Variable": features,
                "RandomForest": rf_importances * 100,
                "LinearRegression": lr_coefficients
            })

            st.dataframe(importance_df)

            # Gráfico comparativo
            fig, ax = plt.subplots(figsize=(8, 6))
            importance_df.set_index("Variable")[["RandomForest", "LinearRegression"]].plot.barh(ax=ax)
            plt.title("Comparación de Importancia / Coeficientes por Variable")
            plt.xlabel("Valor")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al procesar el modelo: {e}")
