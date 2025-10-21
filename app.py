import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="AI Sales Predictor", layout="wide")
st.title("📊 AI-Powered Sales Prediction & Business Insights Dashboard")
st.markdown("Upload your retail sales dataset (CSV) to analyze and predict performance.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Data Cleaning
    # -------------------------------
    st.subheader("🧹 Data Cleaning & Preparation")
    df = df.dropna(subset=['Sales_Amount', 'Profit'], how='any')
    df = df.fillna(df.mean(numeric_only=True))

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write(f"Numeric Columns used for regression: {numeric_cols}")

    # -------------------------------
    # Feature Selection
    # -------------------------------
    target = st.selectbox("Select Target Column (Y):", numeric_cols, index=numeric_cols.index('Sales_Amount') if 'Sales_Amount' in numeric_cols else 0)
    features = st.multiselect("Select Feature Columns (X):", [col for col in numeric_cols if col != target], default=[col for col in numeric_cols if col != target])

    if len(features) > 0:
        X = df[features]
        y = df[target]

        # -------------------------------
        # Split & Train Model
        # -------------------------------
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # -------------------------------
        # Dashboard Results
        # -------------------------------
        st.subheader("📈 Model Performance")
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("Mean Absolute Error", f"{mae:.2f}")

        # Plot actual vs predicted
        st.subheader("📊 Actual vs Predicted Sales")
        fig, ax = plt.subplots()
        sns.lineplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title("Actual vs Predicted Sales")
        st.pyplot(fig)

        # -------------------------------
        # Feature Importance
        # -------------------------------
        st.subheader("🔍 Feature Importance")
        coeff = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", ascending=False)

        st.bar_chart(coeff.set_index("Feature"))

        # -------------------------------
        # Business Insights
        # -------------------------------
        st.subheader("💡 Business Insights & Recommendations")
        top_positive = coeff.iloc[0]
        top_negative = coeff.iloc[-1]

        st.markdown(f"""
        - 🟢 **Positive Influence:** `{top_positive['Feature']}` has the strongest *positive* impact on sales.
        - 🔴 **Negative Influence:** `{top_negative['Feature']}` has the strongest *negative* impact on sales.
        """)

        # Generate business suggestions dynamically
        st.markdown("### 📢 Recommendations")
        if top_negative["Coefficient"] < 0:
            st.info(f"Reduce dependency on **{top_negative['Feature']}** or optimize its value to improve profitability.")
        if top_positive["Coefficient"] > 0:
            st.success(f"Focus more on **{top_positive['Feature']}** — increasing it could boost sales growth.")
        
        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit + AI Regression | By Dhanush & Team")
