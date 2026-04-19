import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score
)

from langchain_core.messages import HumanMessage


def create_confusion_matrix_plotly(cm, class_names=None):
    """Create interactive confusion matrix"""
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
        width=500,
        title_x=0.5
    )

    return fig


def create_roc_curve_plotly(fpr, tpr, roc_auc):
    """Create interactive ROC curve"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='darkorange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.2)'
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Baseline',
        line=dict(color='navy', width=2, dash='dash')
    ))

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        width=600,
        title_x=0.5,
        showlegend=True,
        legend=dict(yanchor="bottom", xanchor="right", x=0.99, y=0.01)
    )

    return fig


def create_feature_importance_plotly(importances, feature_names=None, title="Feature Importance"):
    """Create interactive feature importance plot"""
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    # Sort by importance
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    sorted_imp = importances[indices]
    sorted_names = [str(feature_names[i]) for i in indices]

    # Clean up feature names for display
    clean_names = [name.replace('_', ' ').title() for name in sorted_names]

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_imp,
            y=clean_names,
            orientation='h',
            marker_color='steelblue',
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title="Importance Score (Higher = More Important for Prediction)",
        yaxis_title="Feature",
        height=500,
        width=800,
        title_x=0.5,
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_prediction_distribution_plotly(y_test, preds, problem_type, target_name="Target"):
    """Create interactive prediction distribution with user-friendly labels"""

    target_label = target_name.replace('_', ' ').title()

    if problem_type == "classification":
        # For classification: Compare actual vs predicted counts
        actual_counts = pd.Series(y_test).value_counts().sort_index()
        pred_counts = pd.Series(preds).value_counts().sort_index()

        # Combine categories
        all_categories = sorted(set(actual_counts.index) | set(pred_counts.index))

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[str(c) for c in all_categories],
            y=[actual_counts.get(c, 0) for c in all_categories],
            name='Actual Values',
            marker_color='steelblue',
            opacity=0.8,
            hovertemplate=f'{target_label}: %{{x}}<br>Actual Count: %{{y}}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            x=[str(c) for c in all_categories],
            y=[pred_counts.get(c, 0) for c in all_categories],
            name='Predicted by Model',
            marker_color='coral',
            opacity=0.8,
            hovertemplate=f'{target_label}: %{{x}}<br>Predicted Count: %{{y}}<extra></extra>'
        ))

        fig.update_layout(
            title=f"📊 Model Predictions vs Actual {target_label} Distribution",
            xaxis_title=f"{target_label} Category",
            yaxis_title="Number of Cases",
            barmode='group',
            height=500,
            width=800,
            title_x=0.5,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )

    else:
        # For regression: Use subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Distribution of {target_label} Values",
                f"How Well Model Predicts {target_label}"
            ],
            horizontal_spacing=0.12
        )

        # Left: Histogram comparing actual vs predicted
        fig.add_trace(
            go.Histogram(
                x=y_test,
                name=f"Actual {target_label}",
                nbinsx=30,
                marker_color='steelblue',
                opacity=0.7,
                hovertemplate=f'Actual {target_label}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Histogram(
                x=preds,
                name=f"Predicted {target_label}",
                nbinsx=30,
                marker_color='coral',
                opacity=0.7,
                hovertemplate=f'Predicted {target_label}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=1, col=1
        )

        # Right: Actual vs Predicted scatter
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=preds,
                mode='markers',
                marker=dict(size=10, opacity=0.5, color='green'),
                name='Predictions',
                hovertemplate=f'Actual {target_label}: %{{x}}<br>Predicted {target_label}: %{{y}}<extra></extra>'
            ),
            row=1, col=2
        )

        # Perfect prediction line
        min_val = min(min(y_test), min(preds))
        max_val = max(max(y_test), max(preds))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction Line',
                hovertemplate='Perfect Prediction (Actual = Predicted)<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=500,
            width=1100,
            showlegend=True,
            title_text=f"📈 {target_label} Prediction Results",
            title_x=0.5,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="center", x=0.5)
        )

        # Update axes labels
        fig.update_xaxes(title_text=f"Actual {target_label}", row=1, col=2)
        fig.update_yaxes(title_text=f"Predicted {target_label}", row=1, col=2)

    return fig


def create_residuals_plotly(y_test, preds):
    """Create interactive residuals plot for regression"""
    residuals = y_test - preds

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residuals vs Predicted", "Residuals Distribution"]
    )

    # Residuals scatter
    fig.add_trace(
        go.Scatter(
            x=preds,
            y=residuals,
            mode='markers',
            marker=dict(size=8, opacity=0.6, color='blue'),
            hovertemplate='Predicted: %{x}<br>Residual: %{y}<extra></extra>'
        ),
        row=1, col=1
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Residuals histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='green',
            opacity=0.7,
            hovertemplate='Residual: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=450,
        width=900,
        title_text="Residual Analysis",
        title_x=0.5,
        xaxis_title="Predicted",
        yaxis_title="Residuals"
    )

    return fig


def evaluation_agent(state, llm):
    os.makedirs("outputs/evaluation", exist_ok=True)

    model = state["best_model"]
    X_test = state["X_test"]
    y_test = state["y_test"]
    model_name = state["model_name"]
    problem_type = state.get("problem_type", "classification")

    preds = model.predict(X_test)

    metrics = {}
    interactive_plots = {}

    # Get cross-validation results from modeling
    cv_results = state.get("cv_results", {})
    cv_folds_used = state.get("cv_folds_used", 5)
    model_results = state.get("model_results", {})

    # Get best model's CV metrics
    best_model_cv = model_results.get(model_name, {})

    # -------------------------
    # CLASSIFICATION
    # -------------------------
    if problem_type == "classification":

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        # Get precision, recall, F1 (weighted average for multi-class)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        metrics["accuracy"] = round(acc, 4)
        metrics["precision"] = round(precision, 4)
        metrics["recall"] = round(recall, 4)
        metrics["f1_score"] = round(f1, 4)

        # Add CV metrics if available
        if best_model_cv:
            metrics["cv_accuracy_mean"] = round(best_model_cv.get("cv_accuracy_mean", acc), 4)
            metrics["cv_accuracy_std"] = round(best_model_cv.get("cv_accuracy_std", 0), 4)

        print(f"\nClassification Metrics (using {cv_folds_used}-fold CV):")
        if best_model_cv:
            print(f"  CV Accuracy: {best_model_cv.get('cv_accuracy_mean', acc):.4f} (+/- {best_model_cv.get('cv_accuracy_std', 0)*2:.4f})")
        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        # -------------------------
        # Confusion Matrix (Interactive)
        # -------------------------
        cm = confusion_matrix(y_test, preds)

        # Static version
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("outputs/evaluation/confusion_matrix.png")
        plt.close()

        # Interactive version
        cm_fig = create_confusion_matrix_plotly(cm)
        cm_fig.write_html("outputs/evaluation/confusion_matrix.html")
        interactive_plots['confusion_matrix'] = cm_fig.to_json()

        # -------------------------
        # ROC Curve (Interactive)
        # -------------------------
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]

                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)

                # Static version
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.savefig("outputs/evaluation/roc_curve.png")
                plt.close()

                # Interactive version
                roc_fig = create_roc_curve_plotly(fpr, tpr, roc_auc)
                roc_fig.write_html("outputs/evaluation/roc_curve.html")
                interactive_plots['roc_curve'] = roc_fig.to_json()

                metrics["roc_auc"] = roc_auc

        except Exception as e:
            print("ROC skipped:", e)

    # -------------------------
    # REGRESSION
    # -------------------------
    else:
        n = len(y_test)
        p = X_test.shape[1] if hasattr(X_test, 'shape') else len(state.get("feature_names", []))

        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)

        # Adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
        if p > 0 and n > p + 1:
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adjusted_r2 = r2

        metrics["r2"] = round(r2, 4)
        metrics["adjusted_r2"] = round(adjusted_r2, 4)
        metrics["mse"] = round(mse, 4)
        metrics["rmse"] = round(rmse, 4)
        metrics["mae"] = round(mae, 4)

        # Add CV metrics if available
        if best_model_cv:
            metrics["cv_r2_mean"] = round(best_model_cv.get("cv_r2_mean", r2), 4)
            metrics["cv_r2_std"] = round(best_model_cv.get("cv_r2_std", 0), 4)
            metrics["cv_rmse"] = round(best_model_cv.get("cv_rmse", rmse), 4)

        print(f"\nRegression Metrics (using {cv_folds_used}-fold CV):")
        if best_model_cv:
            print(f"  CV R²: {best_model_cv.get('cv_r2_mean', r2):.4f} (+/- {best_model_cv.get('cv_r2_std', 0)*2:.4f})")
            print(f"  CV RMSE: {best_model_cv.get('cv_rmse', rmse):.4f}")
        print(f"  Test R²: {r2:.4f}")
        print(f"  Adjusted R²: {adjusted_r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

        # Residuals plot (regression only)
        try:
            residuals_fig = create_residuals_plotly(y_test, preds)
            residuals_fig.write_html("outputs/evaluation/residuals.html")
            interactive_plots['residuals'] = residuals_fig.to_json()
        except Exception as e:
            print("Residuals plot skipped:", e)

    # -------------------------
    # FEATURE IMPORTANCE (Interactive)
    # -------------------------
    try:
        feature_names = state.get("feature_names", None)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

            # Static version
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances, y=np.arange(len(importances)))
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature Index")
            plt.savefig("outputs/evaluation/feature_importance.png")
            plt.close()

            # Interactive version
            imp_fig = create_feature_importance_plotly(
                importances, feature_names, "Feature Importance"
            )
            imp_fig.write_html("outputs/evaluation/feature_importance.html")
            interactive_plots['feature_importance'] = imp_fig.to_json()

        elif hasattr(model, "coef_"):
            coefs = model.coef_.flatten()

            # Static version
            plt.figure(figsize=(10, 6))
            sns.barplot(x=coefs, y=np.arange(len(coefs)))
            plt.title("Feature Coefficients")
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature Index")
            plt.savefig("outputs/evaluation/feature_importance.png")
            plt.close()

            # Interactive version
            coef_fig = create_feature_importance_plotly(
                np.abs(coefs), feature_names, "Feature Coefficients (Absolute)"
            )
            coef_fig.write_html("outputs/evaluation/feature_importance.html")
            interactive_plots['feature_importance'] = coef_fig.to_json()

    except Exception as e:
        print("Feature importance skipped:", e)

    # -------------------------
    # Prediction Distribution (Interactive)
    # -------------------------
    try:
        target_name = state.get("target_column", "Target")
        pred_fig = create_prediction_distribution_plotly(y_test, preds, problem_type, target_name)
        pred_fig.write_html("outputs/evaluation/predictions.html")
        interactive_plots['predictions'] = pred_fig.to_json()

        # Static fallback
        plt.figure()
        sns.histplot(preds, kde=True)
        plt.title("Prediction Distribution")
        plt.savefig("outputs/evaluation/predictions.png")
        plt.close()
    except Exception as e:
        print("Prediction distribution skipped:", e)

    # Save interactive plots metadata
    with open("outputs/evaluation/interactive_plots.json", "w") as f:
        json.dump(interactive_plots, f)

    # -------------------------
    # LLM SUMMARY
    # -------------------------
    summary_prompt = f"""
    You are a senior data scientist.

    Model Used: {model_name}
    Problem Type: {problem_type}
    Metrics: {metrics}

    Provide a professional summary including:
    - Model performance interpretation
    - Whether model is good or not
    - Any risks (overfitting, imbalance)
    - Suggestions for improvement
    """

    summary = llm.invoke([HumanMessage(content=summary_prompt)]).content

    return {
        **state,
        "metrics": metrics,
        "evaluation_report": summary,
        "evaluation_plots": interactive_plots
    }