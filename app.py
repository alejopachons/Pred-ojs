import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Comparación de Modelos", layout="centered")

st.title("🔍 Comparación de Importancia de Variables entre Modelos")
st.write("Sube un archivo Excel para comparar RandomForest y GradientBoosting en regresión.")

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
        target = st.selectbox("Variable objetivo (Y)", numeric_columns)
        submitted = st.form_submit_button("Comparar modelos")

    if submitted and features and target:
        try:
            X = df[features]
            y = df[target]

            # -------------------------
            # Random Forest
            # -------------------------
            rf_model = RandomForestRegressor(random_state=0)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X)
            rf_importances = rf_model.feature_importances_
            rf_r2 = r2_score(y, rf_pred)
            rf_mae = mean_absolute_error(y, rf_pred)

            # -------------------------
            # Gradient Boosting
            # -------------------------
            gb_model = GradientBoostingRegressor(random_state=0)
            gb_model.fit(X, y)
            gb_pred = gb_model.predict(X)
            gb_importances = gb_model.feature_importances_
            gb_r2 = r2_score(y, gb_pred)
            gb_mae = mean_absolute_error(y, gb_pred)

            # -------------------------
            # Métricas comparativas
            # -------------------------
            st.subheader("📈 Comparación de desempeño (sobre datos de entrenamiento)")
            st.write(pd.DataFrame({
                "Modelo": ["Random Forest", "Gradient Boosting"],
                "R²": [rf_r2, gb_r2],
                "MAE": [rf_mae, gb_mae]
            }))

            # -------------------------
            # Importancia de variables
            # -------------------------
            st.subheader("📊 Importancia de Variables")
            importance_df = pd.DataFrame({
                "Variable": features,
                "Random Forest (%)": rf_importances * 100,
                "Gradient Boosting (%)": gb_importances * 100
            })

            st.dataframe(importance_df)

            # Gráfico
            fig, ax = plt.subplots(figsize=(8, 6))
            importance_df.set_index("Variable")[["Random Forest (%)", "Gradient Boosting (%)"]].plot.barh(ax=ax)
            plt.title("Importancia de Variables por Modelo")
            plt.xlabel("Importancia (%)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al procesar el modelo: {e}")
