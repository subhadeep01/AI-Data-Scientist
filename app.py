import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import json
import plotly.graph_objects as go

from agents.preprocessing_agent import preprocessing_agent
from agents.modeling_agent import modeling_agent
from agents.evaluation_agent import evaluation_agent
from agents.explainability_agent import explainability_agent

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="AI Data Scientist Agent",
    layout="wide"
)

st.title("🤖 AI Data Scientist Agent")

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # -------------------------
    # CONFIGURATION
    # -------------------------
    col1, col2 = st.columns(2)
    with col1:
        target = st.text_input("🎯 Enter target column")
    with col2:
        problem_type = st.selectbox(
            "📋 Problem Type",
            ["Auto-detect", "Classification", "Regression"],
            help="Select the type of machine learning problem. Auto-detect will infer from your target column."
        )

    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col3, col4, col5 = st.columns(3)
        with col3:
            scaler_type = st.selectbox(
                "Feature Scaling",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
                help="StandardScaler: zero mean, unit variance | MinMaxScaler: scale to [0,1] | RobustScaler: robust to outliers"
            )
        with col4:
            transform_target = st.checkbox(
                "Transform Target",
                value=False,
                help="Apply log transformation to target if highly skewed (regression only)"
            )
        with col5:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for k-fold cross-validation"
            )

    if st.button("🚀 Run Pipeline"):

        # -------------------------
        # VALIDATION
        # -------------------------
        if target.strip() == "":
            st.error("❌ Please enter a target column")
            st.stop()

        if target not in df.columns:
            st.error("❌ Target column not found in dataset")
            st.stop()

        # -------------------------
        # RUN PIPELINE
        # -------------------------
        with st.spinner("Running AI Pipeline... Please wait ⏳"):

            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0
                )

                state = {
                    "data": df,
                    "target_column": target,
                    "problem_type_user": problem_type.lower().replace("-", "_"),
                    "scaler_type": scaler_type.lower().replace("scaler", "").strip(),
                    "transform_target": transform_target,
                    "cv_folds": cv_folds
                }

                # Run agents
                state = preprocessing_agent(state)
                state = modeling_agent(state)
                state = evaluation_agent(state, llm)
                state = explainability_agent(state)

                st.success("✅ Pipeline Completed Successfully!")

            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                st.stop()

        # -------------------------
        # METRICS
        # -------------------------
        st.subheader("📈 Model Metrics")

        problem_type = state.get("problem_type", "classification")
        metrics = state.get("metrics", {})

        if problem_type == "regression":
            # Display regression metrics in a nice format
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", metrics.get("r2", "N/A"))
            with col2:
                st.metric("Adjusted R²", metrics.get("adjusted_r2", "N/A"))
            with col3:
                st.metric("RMSE", metrics.get("rmse", "N/A"))

            col4, col5 = st.columns(2)
            with col4:
                st.metric("MSE", metrics.get("mse", "N/A"))
            with col5:
                st.metric("MAE", metrics.get("mae", "N/A"))

            with st.expander("ℹ️ Metric Explanations"):
                st.markdown("""
                - **R² Score**: Proportion of variance explained by the model (1.0 = perfect)
                - **Adjusted R²**: R² penalized for number of features (prevents overfitting)
                - **RMSE**: Root Mean Squared Error - average prediction error magnitude
                - **MSE**: Mean Squared Error - penalizes larger errors more heavily
                - **MAE**: Mean Absolute Error - average absolute difference from actual values
                """)
        else:
            # Classification metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")

            col4, col5 = st.columns(2)
            with col4:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
            with col5:
                st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")

            with st.expander("ℹ️ Metric Explanations"):
                st.markdown("""
                - **Accuracy**: Proportion of correct predictions out of total predictions
                - **Precision**: Of predicted positives, how many were actually correct (minimize false positives)
                - **Recall**: Of actual positives, how many were correctly identified (minimize false negatives)
                - **F1 Score**: Harmonic mean of Precision and Recall (balanced measure)
                - **ROC AUC**: Area Under ROC Curve (1.0 = perfect, 0.5 = random)
                """)

        st.json(metrics)

        # -------------------------
        # EDA OUTPUTS - Interactive
        # -------------------------
        st.subheader("📊 EDA Visualizations")

        eda_path = "outputs/eda"

        # Load interactive plots metadata
        plots_metadata = state.get("eda_plots", {})

        # STEP 1: Histograms
        st.markdown("### 1️⃣ Numerical Distributions")
        st.markdown("*Shows the range and frequency of numerical values in your data.*")
        if 'histograms' in plots_metadata:
            fig = go.Figure(json.loads(plots_metadata['histograms']))
            st.plotly_chart(fig, use_container_width=True)
        elif os.path.exists(f"{eda_path}/histograms.html"):
            with open(f"{eda_path}/histograms.html", 'r') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
        elif os.path.exists(f"{eda_path}/histograms.png"):
            st.image(f"{eda_path}/histograms.png", use_column_width=True)

        # STEP 2: Countplots (Categorical)
        st.markdown("### 2️⃣ Categorical Distributions")
        st.markdown("*Shows how many items fall into each category. Helps identify popular/uncommon categories.*")
        if 'countplots' in plots_metadata:
            fig = go.Figure(json.loads(plots_metadata['countplots']))
            st.plotly_chart(fig, use_container_width=True)
        elif os.path.exists(f"{eda_path}/countplots.html"):
            with open(f"{eda_path}/countplots.html", 'r') as f:
                components.html(f.read(), height=600, scrolling=True)
        elif os.path.exists(f"{eda_path}/countplots.png"):
            st.image(f"{eda_path}/countplots.png", use_column_width=True)

        # STEP 3: Outlier Analysis
        st.markdown("### 3️⃣ Outlier Analysis")
        st.markdown("*Shows data points that are unusually high or low. These may need special attention.*")
        if 'outliers' in plots_metadata:
            fig = go.Figure(json.loads(plots_metadata['outliers']))
            st.plotly_chart(fig, use_container_width=True)
        elif os.path.exists(f"{eda_path}/outliers.html"):
            with open(f"{eda_path}/outliers.html", 'r') as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
        elif os.path.exists(f"{eda_path}/outliers.png"):
            st.image(f"{eda_path}/outliers.png", use_column_width=True)

        # STEP 4: Numerical Correlation
        st.markdown("### 4️⃣ Numerical Relationships")
        st.markdown("*Shows how numerical variables relate to each other. Dark red = strongly increase together. Dark blue = one increases while other decreases.*")
        if 'corr_numerical' in plots_metadata:
            fig = go.Figure(json.loads(plots_metadata['corr_numerical']))
            st.plotly_chart(fig, use_container_width=True)
        elif os.path.exists(f"{eda_path}/corr_num.html"):
            with open(f"{eda_path}/corr_num.html", 'r') as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        elif os.path.exists(f"{eda_path}/corr_num.png"):
            st.image(f"{eda_path}/corr_num.png", use_column_width=True)

        # STEP 5: Cramer's V Matrix
        st.markdown("### 5️⃣ Categorical Relationships")
        st.markdown("*Shows how categorical variables relate to each other. Brighter colors = stronger connection between categories.*")
        if 'corr_categorical' in plots_metadata:
            fig = go.Figure(json.loads(plots_metadata['corr_categorical']))
            st.plotly_chart(fig, use_container_width=True)
        elif os.path.exists(f"{eda_path}/corr_cat.html"):
            with open(f"{eda_path}/corr_cat.html", 'r') as f:
                components.html(f.read(), height=700, scrolling=True)
        elif os.path.exists(f"{eda_path}/corr_cat.png"):
            st.image(f"{eda_path}/corr_cat.png", use_column_width=True)

        # -------------------------
        # EVALUATION OUTPUTS (Interactive)
        # -------------------------
        st.subheader("📉 How Well the Model Performed")

        eval_plots = state.get("evaluation_plots", {})

        if eval_plots:
            # CLASSIFICATION-SPECIFIC PLOTS
            if problem_type == "classification":
                # Confusion Matrix (Classification)
                if 'confusion_matrix' in eval_plots:
                    st.markdown("**✅ Prediction Accuracy Matrix**")
                    st.markdown("*Shows where the model was correct (diagonal) vs incorrect (off-diagonal).*")
                    fig = go.Figure(json.loads(eval_plots['confusion_matrix']))
                    st.plotly_chart(fig, use_container_width=True)

                # ROC Curve (Classification)
                if 'roc_curve' in eval_plots:
                    st.markdown("**📈 Model Quality (ROC Curve)**")
                    st.markdown("*Higher curve = Better model. AUC closer to 1.0 means excellent predictions.*")
                    fig = go.Figure(json.loads(eval_plots['roc_curve']))
                    st.plotly_chart(fig, use_container_width=True)

            # REGRESSION-SPECIFIC PLOTS
            else:
                # Residuals (Regression)
                if 'residuals' in eval_plots:
                    st.markdown("**📊 Prediction Errors (Residuals)**")
                    st.markdown("*Shows how far off the predictions were. Points closer to zero line = better predictions.*")
                    fig = go.Figure(json.loads(eval_plots['residuals']))
                    st.plotly_chart(fig, use_container_width=True)

            # COMMON PLOTS (both classification and regression)
            # Feature Importance
            if 'feature_importance' in eval_plots:
                st.markdown("**🔑 Most Important Features**")
                st.markdown("*These features had the biggest influence on the model's predictions.*")
                fig = go.Figure(json.loads(eval_plots['feature_importance']))
                st.plotly_chart(fig, use_container_width=True)

            # Predictions Distribution
            if 'predictions' in eval_plots:
                st.markdown("**🎯 Prediction Results**")
                st.markdown("*Compares what the model predicted vs the actual values.*")
                fig = go.Figure(json.loads(eval_plots['predictions']))
                st.plotly_chart(fig, use_container_width=True)

        # Fallback to static images (filter by problem type)
        eval_path = "outputs/evaluation"
        if os.path.exists(eval_path):
            static_files = [f for f in os.listdir(eval_path) if f.endswith('.png')]
            # Filter static files based on problem type
            if problem_type == "regression":
                static_files = [f for f in static_files if 'confusion_matrix' not in f and 'roc_curve' not in f]
            else:
                static_files = [f for f in static_files if 'residuals' not in f]

            if static_files and not eval_plots:
                cols = st.columns(2)
                for i, file in enumerate(sorted(static_files)):
                    with cols[i % 2]:
                        st.image(os.path.join(eval_path, file), caption=file.replace('.png', ''))

        # -------------------------
        # SHAP EXPLANATIONS (Interactive)
        # -------------------------
        st.subheader("🔍 Why the Model Makes These Predictions (SHAP)")

        shap_plots = state.get("explainability_plots", {})

        if shap_plots:
            # SHAP Summary (Beeswarm)
            if 'shap_summary' in shap_plots:
                st.markdown("**📊 Feature Impact Summary**")
                st.markdown("*This shows how each feature value affects the prediction. Red = Higher prediction, Blue = Lower prediction.*")
                fig = go.Figure(json.loads(shap_plots['shap_summary']))
                st.plotly_chart(fig, use_container_width=True)

            # SHAP Bar Plot
            if 'shap_bar' in shap_plots:
                st.markdown("**📈 Most Important Features**")
                st.markdown("*Features ranked by their average impact on predictions. Longer bars = More important.*")
                fig = go.Figure(json.loads(shap_plots['shap_bar']))
                st.plotly_chart(fig, use_container_width=True)

            # SHAP Waterfall
            if 'shap_waterfall' in shap_plots:
                st.markdown("**💧 Single Prediction Breakdown**")
                st.markdown("*Shows how each feature contributed to one specific prediction. Red = Pushed prediction up, Green = Pushed prediction down.*")
                fig = go.Figure(json.loads(shap_plots['shap_waterfall']))
                st.plotly_chart(fig, use_container_width=True)

        # Fallback to static SHAP image
        shap_path = "outputs/explainability/shap_summary.png"
        if os.path.exists(shap_path) and not shap_plots:
            st.image(shap_path, caption="SHAP Summary Plot")

        # -------------------------
        # LLM REPORT
        # -------------------------
        st.subheader("🧠 AI-Generated Insights")

        report = state.get("evaluation_report", "")
        if report:
            st.write(report)
        else:
            st.warning("No report generated.")