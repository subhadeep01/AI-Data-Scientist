import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import plotly.graph_objects as go


def create_shap_summary_plotly(shap_values, X_test, feature_names=None, max_display=20, problem_type="classification"):
    """Create interactive SHAP summary plot (beeswarm style)

    Args:
        shap_values: SHAP values array (samples x features)
        X_test: Test data (DataFrame or array)
        feature_names: List of feature names
        max_display: Maximum number of features to display
        problem_type: 'classification' or 'regression' for appropriate labels
    """
    import pandas as pd

    # Convert X_test to DataFrame with proper feature names
    if isinstance(X_test, np.ndarray):
        if feature_names is None or len(feature_names) != X_test.shape[1]:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test_df = X_test.copy()
        if feature_names is not None and len(feature_names) == X_test_df.shape[1]:
            X_test_df.columns = feature_names
        else:
            feature_names = list(X_test_df.columns)

    # Ensure shap_values is 2D array
    shap_values = np.asarray(shap_values)
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)

    print(f"SHAP values shape: {shap_values.shape}, X_test shape: {X_test_df.shape}")
    print(f"Feature names: {len(feature_names)}")

    # Calculate mean absolute SHAP values for ordering
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Get top features by importance
    if len(mean_abs_shap) > max_display:
        top_indices = np.argsort(mean_abs_shap)[-max_display:]
    else:
        top_indices = np.argsort(mean_abs_shap)

    # Build data for beeswarm plot
    plot_data = []
    for i, idx in enumerate(top_indices):
        feature_name = feature_names[idx].replace('_', ' ').title()
        original_name = feature_names[idx]

        # Get SHAP values and feature values for this feature
        shap_vals = shap_values[:, idx]
        feat_vals = X_test_df.iloc[:, idx].values

        for j, (sv, fv) in enumerate(zip(shap_vals, feat_vals)):
            plot_data.append({
                'shap_value': sv,
                'feature_value': fv,
                'feature_name': feature_name,
                'original_name': original_name
            })

    # Create DataFrame for easier plotting
    plot_df = pd.DataFrame(plot_data)

    # Create the beeswarm plot
    fig = go.Figure()

    # Get unique features in order (highest importance at top)
    unique_features = [feature_names[i].replace('_', ' ').title() for i in top_indices]

    # Calculate global min/max for consistent color scale
    all_feature_values = plot_df['feature_value'].values
    vmin, vmax = np.percentile(all_feature_values, [1, 99])  # Use percentiles to handle outliers

    # Add scatter plot for each feature
    for i, feature_name in enumerate(unique_features):
        feature_data = plot_df[plot_df['feature_name'] == feature_name]

        # Add jitter to y-position for beeswarm effect
        y_positions = [i] * len(feature_data)

        fig.add_trace(go.Scatter(
            x=feature_data['shap_value'],
            y=y_positions,
            mode='markers',
            marker=dict(
                size=7,
                color=feature_data['feature_value'],
                colorscale='RdBu_r',  # Red for high values, Blue for low
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(
                    title="Feature Value",
                    titleside="right",
                    thickness=15,
                    len=0.6
                ) if i == 0 else None,
                showscale=(i == 0),
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            name=feature_name,
            hovertemplate=(
                f"<b>{feature_data['original_name'].iloc[0]}</b><br>" +
                "SHAP Value: %{x:.4f}<br>" +
                "Feature Value: %{marker.color:.4f}<extra></extra>"
            ),
            showlegend=False
        ))

    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1.5, opacity=0.6)

    # Determine labels based on problem type
    if problem_type == "classification":
        impact_label = "SHAP Value (Impact on Prediction Probability)"
        subtitle = "Red = High Feature Value pushes toward class 1 | Blue = Low Feature Value pushes toward class 0"
    else:
        impact_label = "SHAP Value (Impact on Prediction)"
        subtitle = "Red = High Feature Value increases prediction | Blue = Low Feature Value decreases prediction"

    fig.update_layout(
        title=dict(
            text=f"🔍 SHAP Feature Impact Summary<br><sub>{subtitle}</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=impact_label,
        yaxis_title="Feature",
        height=max(450, 35 * len(unique_features) + 150),
        width=950,
        template="plotly_white",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(unique_features))),
            ticktext=unique_features,
            autorange="reversed",
            tickfont=dict(size=11)
        ),
        xaxis=dict(zeroline=False, showgrid=True, gridwidth=1, gridcolor='lightgray'),
        margin=dict(l=150, r=100, t=100, b=80)
    )

    return fig


def create_shap_bar_plotly(shap_values, feature_names=None, max_display=20):
    """Create interactive SHAP feature importance bar chart"""

    # Handle multi-class SHAP values (list of arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Ensure shap_values is 2D array
    shap_values = np.asarray(shap_values)
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)

    if feature_names is None or len(feature_names) != shap_values.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(shap_values.shape[1])]

    # Clean feature names for display
    clean_feature_names = [name.replace('_', ' ').title() for name in feature_names]

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort and get top features
    indices = np.argsort(mean_abs_shap)[-max_display:]
    sorted_features = [clean_feature_names[i] for i in indices]
    sorted_values = mean_abs_shap[indices]
    original_names = [feature_names[i] for i in indices]

    # Create custom hover text
    hover_text = [f"Feature: {orig}<br>Average |SHAP|: {val:.4f}"
                  for orig, val in zip(original_names, sorted_values)]

    fig = go.Figure(data=[
        go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation='h',
            marker_color='steelblue',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        )
    ])

    fig.update_layout(
        title="🔍 Feature Importance (Mean |SHAP Value|)",
        xaxis_title="Average Absolute Impact",
        yaxis_title="Feature",
        height=max(400, 25 * len(sorted_features) + 100),
        width=800,
        title_x=0.5,
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )

    return fig


def create_waterfall_plotly(shap_values, feature_names, feature_values, base_value=0):
    """Create interactive SHAP waterfall plot for a single prediction"""

    # Handle multi-class SHAP values (list of arrays)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Ensure arrays
    shap_values = np.asarray(shap_values).flatten()
    feature_values = np.asarray(feature_values).flatten()

    # Validate lengths match
    if len(shap_values) != len(feature_names):
        print(f"WARNING: SHAP values length ({len(shap_values)}) != feature names ({len(feature_names)})")
        # Truncate to minimum
        min_len = min(len(shap_values), len(feature_names))
        shap_values = shap_values[:min_len]
        feature_values = feature_values[:min_len]
        feature_names = feature_names[:min_len]

    n_features = min(10, len(feature_names))

    # Get top features by absolute SHAP value
    indices = np.argsort(np.abs(shap_values))[-n_features:]

    # Clean feature names for display
    clean_feature_names = [feature_names[i].replace('_', ' ').title() for i in indices]
    selected_shap = shap_values[indices]
    selected_values = feature_values[indices]

    # Build waterfall
    measure = ["relative"] * n_features + ["total"]
    x = list(selected_shap) + [base_value + sum(selected_shap)]
    text = [f"{f}<br>Value: {v:.3f}" for f, v in zip(clean_feature_names, selected_values)] + ["Final Prediction"]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=measure,
        x=list(range(n_features + 1)),
        text=text,
        y=x,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#FF6B6B"}},
        increasing={"marker": {"color": "#4ECDC4"}},
        totals={"marker": {"color": "#45B7D1"}}
    ))

    fig.update_layout(
        title="🔍 How One Prediction Was Made (Feature Contributions)",
        xaxis_title="",
        yaxis_title="Contribution to Prediction",
        height=500,
        width=900,
        title_x=0.5,
        template="plotly_white",
        xaxis=dict(showticklabels=False)
    )

    return fig


def explainability_agent(state):
    os.makedirs("outputs/explainability", exist_ok=True)

    model = state["best_model"]
    X_train = state["X_train"]
    X_test = state["X_test"]

    interactive_plots = {}

    try:
        # Get feature names - ensure we have proper feature names
        feature_names = state.get("feature_names", None)

        # Debug logging
        print(f"Feature names from state: {feature_names is not None}")

        if feature_names is None or len(feature_names) == 0:
            if hasattr(X_test, 'columns'):
                feature_names = list(X_test.columns)
                print(f"Using X_test.columns: {len(feature_names)} features")
            else:
                feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
                print(f"WARNING: Using generic feature names")
        else:
            print(f"Using state feature_names: {len(feature_names)} features")
            # Ensure feature_names matches X_test shape
            if hasattr(X_test, 'shape') and X_test.shape[1] != len(feature_names):
                print(f"WARNING: Feature names count ({len(feature_names)}) doesn't match X_test columns ({X_test.shape[1]})")
                feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

        # Tree models
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            print(f"TreeExplainer shap_values type: {type(shap_values)}")

        else:
            # General fallback
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            print(f"General Explainer shap_values type: {type(shap_values)}")

        # Handle SHAP values format
        shap_values_array = shap_values
        problem_type = state.get("problem_type", "classification")

        if isinstance(shap_values, list):
            # Multi-class classification: list of arrays, one per class
            if problem_type == "classification":
                # For binary classification, use the positive class (index 1)
                # For multi-class, we could average or use a specific class
                if len(shap_values) > 1:
                    shap_values_array = shap_values[1]  # Use positive class for binary
                    print(f"Using SHAP values for positive class (class 1), shape: {shap_values_array.shape}")
                else:
                    shap_values_array = shap_values[0]
                    print(f"Using SHAP values[0], shape: {shap_values_array.shape}")
            else:  # regression - shouldn't be a list, but handle gracefully
                shap_values_array = shap_values[0] if len(shap_values) > 0 else shap_values
                print(f"Using SHAP values[0] for regression, shape: {shap_values_array.shape}")
        elif hasattr(shap_values, 'values'):
            # SHAP Explanation object (newer API)
            shap_values_array = shap_values.values
            print(f"Extracted values from Explanation object, shape: {shap_values_array.shape}")

        # Ensure 2D array for both classification and regression
        if len(shap_values_array.shape) == 3:
            # Multi-class with shape (samples, features, classes) - take positive class
            shap_values_array = shap_values_array[:, :, 1] if shap_values_array.shape[2] > 1 else shap_values_array[:, :, 0]
            print(f"Reshaped from 3D to 2D: {shap_values_array.shape}")

        # Static summary plot - use feature_names
        plt.figure()
        shap.summary_plot(shap_values_array, X_test, feature_names=feature_names, show=False)
        plt.savefig("outputs/explainability/shap_summary.png", bbox_inches='tight')
        plt.close()

        # Interactive summary plot (beeswarm style)
        try:
            summary_fig = create_shap_summary_plotly(
                shap_values_array, X_test, feature_names,
                problem_type=problem_type
            )
            summary_fig.write_html("outputs/explainability/shap_summary.html")
            interactive_plots['shap_summary'] = summary_fig.to_json()
            print(f"SHAP beeswarm plot generated successfully for {problem_type}")
        except Exception as e:
            print(f"Interactive SHAP summary failed: {e}")
            import traceback
            traceback.print_exc()

        # Interactive bar plot (feature importance)
        try:
            bar_fig = create_shap_bar_plotly(shap_values_array, feature_names)
            bar_fig.write_html("outputs/explainability/shap_bar.html")
            interactive_plots['shap_bar'] = bar_fig.to_json()
        except Exception as e:
            print(f"Interactive SHAP bar plot failed: {e}")
            import traceback
            traceback.print_exc()

        # Waterfall plot for first sample
        try:
            sample_shap = shap_values_array[0] if len(shap_values_array.shape) > 1 else shap_values_array
            sample_features = X_test[0] if isinstance(X_test, np.ndarray) else X_test.iloc[0].values

            waterfall_fig = create_waterfall_plotly(sample_shap, feature_names, sample_features)
            waterfall_fig.write_html("outputs/explainability/shap_waterfall.html")
            interactive_plots['shap_waterfall'] = waterfall_fig.to_json()
        except Exception as e:
            print(f"SHAP waterfall plot failed: {e}")
            import traceback
            traceback.print_exc()

        # Save interactive plots metadata
        with open("outputs/explainability/interactive_plots.json", "w") as f:
            json.dump(interactive_plots, f)

        print("SHAP explanation generated successfully!")

    except Exception as e:
        print("SHAP failed:", e)
        import traceback
        traceback.print_exc()

    return {
        **state,
        "explainability_plots": interactive_plots
    }