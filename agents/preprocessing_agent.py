import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import chi2_contingency
from scipy.stats import skew
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------
# Helper: Cramér’s V
# -------------------------
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


# -------------------------
# Remove high correlation (numerical)
# -------------------------
def drop_high_corr_num(df, threshold, target):
    num_df = df.select_dtypes(include=np.number)

    corr_matrix = num_df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper.columns
        if any(upper[col] > threshold) and col != target
    ]

    print("Dropped numerical columns:", to_drop)
    return df.drop(columns=to_drop)


# -------------------------
# Remove high correlation (categorical)
# -------------------------
def drop_high_corr_cat(df, cat_cols, threshold, target):
    to_drop = set()

    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            col1, col2 = cat_cols[i], cat_cols[j]

            if col1 == target or col2 == target:
                continue

            score = cramers_v(df[col1], df[col2])

            if score > threshold:
                to_drop.add(col2)

    print("Dropped categorical columns:", list(to_drop))
    return df.drop(columns=list(to_drop))


# -------------------------
# Remove low correlation with target (numerical)
# -------------------------
def drop_low_corr_with_target(df, num_cols, target, min_threshold=0.1):
    """Drop numerical features with correlation less than threshold to target"""
    if target not in df.columns or not num_cols:
        return df

    to_drop = []

    # Only consider numerical target
    if df[target].dtype in ["int64", "float64"]:
        for col in num_cols:
            if col == target:
                continue
            corr = df[col].corr(df[target])
            abs_corr = abs(corr)

            if abs_corr < min_threshold:
                to_drop.append({
                    'column': col,
                    'correlation': corr,
                    'abs_correlation': abs_corr
                })

    if to_drop:
        print(f"\n[*] Dropping numerical features with |correlation| < {min_threshold} to target:")
        for info in to_drop:
            print(f"  - {info['column']}: r = {info['correlation']:.4f}")
        df = df.drop(columns=[d['column'] for d in to_drop])
    else:
        print(f"\n[*] All numerical features have |correlation| >= {min_threshold} with target")

    return df


# -------------------------
# Remove low correlation with target (categorical)
# -------------------------
def drop_low_cramers_with_target(df, cat_cols, target, min_threshold=0.1):
    """Drop categorical features with Cramer's V less than threshold to target"""
    if target not in df.columns or not cat_cols:
        return df

    to_drop = []

    for col in cat_cols:
        if col == target:
            continue

        score = cramers_v(df[col], df[target])

        if score < min_threshold:
            to_drop.append({
                'column': col,
                'cramers_v': score
            })

    if to_drop:
        print(f"\n[*] Dropping categorical features with Cramer's V < {min_threshold} to target:")
        for info in to_drop:
            print(f"  - {info['column']}: V = {info['cramers_v']:.4f}")
        df = df.drop(columns=[d['column'] for d in to_drop])
    else:
        print(f"\n[*] All categorical features have Cramer's V >= {min_threshold} with target")

    return df


# -------------------------
# Helper: Cap outliers at 99th percentile
# -------------------------
def cap_outliers(df, num_cols, target, lower_percentile=0.01, upper_percentile=0.99):
    """Cap outliers at specified percentiles for numerical columns"""
    capped_cols = []

    for col in num_cols:
        if col == target:
            continue

        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)

        # Count outliers before capping
        lower_outliers = (df[col] < lower_bound).sum()
        upper_outliers = (df[col] > upper_bound).sum()

        if lower_outliers > 0 or upper_outliers > 0:
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            capped_cols.append({
                'column': col,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'lower_outliers': lower_outliers,
                'upper_outliers': upper_outliers
            })

    if capped_cols:
        print(f"\n[*] Capped outliers at {lower_percentile*100:.0f}th / {upper_percentile*100:.0f}th percentile:")
        for info in capped_cols:
            print(f"  - {info['column']}: {info['lower_outliers']} lower, {info['upper_outliers']} upper outliers capped")
    else:
        print("\n[*] No significant outliers to cap")

    return df


# -------------------------
# Helper: Create subplot matrix for histograms
# -------------------------
def create_histograms_plotly(df, num_cols, target):
    """Create interactive histograms for numerical columns"""
    if len(num_cols) == 0:
        return None

    # Filter out target from histograms for better view
    plot_cols = [c for c in num_cols if c != target]

    if len(plot_cols) == 0:
        return None

    n_cols = min(3, len(plot_cols))
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols

    # Calculate safe vertical spacing
    v_spacing = min(0.15, 1.0 / (n_rows + 1))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=plot_cols,
        horizontal_spacing=0.1,
        vertical_spacing=v_spacing
    )

    for i, col in enumerate(plot_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                showlegend=False,
                hovertemplate=f'{col}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=row,
            col=col_idx
        )

    fig.update_layout(
        title_text="📊 Numerical Distributions (Histograms)",
        height=300 * n_rows,
        title_x=0.5
    )

    return fig


# -------------------------
# Helper: Create countplots for categorical (grid layout like histograms)
# -------------------------
def create_countplots_plotly(df, cat_cols, target, max_cols=12):
    """Create interactive countplots for categorical columns in a grid"""
    plot_cols = [c for c in cat_cols if c != target]

    # Limit to max_cols most interesting categorical columns
    if len(plot_cols) > max_cols:
        # Prioritize columns with fewer unique values (more informative)
        plot_cols = sorted(plot_cols, key=lambda c: df[c].nunique())[:max_cols]

    if len(plot_cols) == 0:
        return None

    n_cols = min(3, len(plot_cols))
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols

    # Calculate safe vertical spacing
    v_spacing = min(0.15, 1.0 / (n_rows + 1))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[c.replace('_', ' ').title() for c in plot_cols],
        horizontal_spacing=0.12,
        vertical_spacing=v_spacing
    )

    for i, col in enumerate(plot_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        value_counts = df[col].value_counts().nlargest(10)

        fig.add_trace(
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                marker_color='steelblue',
                showlegend=False,
                hovertemplate=f'{col}: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=row,
            col=col_idx
        )

    fig.update_layout(
        title_text="📊 Categorical Distributions (Count Plots)",
        height=max(300, 250 * n_rows),
        title_x=0.5,
        template="plotly_white"
    )

    return fig


# -------------------------
# Helper: Create outlier analysis with boxplots
# -------------------------
def create_outlier_plots_plotly(df, num_cols, target):
    """Create interactive outlier boxplots"""
    plot_cols = [c for c in num_cols if c != target]

    if len(plot_cols) == 0:
        return None

    n_cols = min(3, len(plot_cols))
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols

    # Calculate safe vertical spacing
    v_spacing = min(0.15, 1.0 / (n_rows + 1))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=plot_cols,
        horizontal_spacing=0.1,
        vertical_spacing=v_spacing
    )

    for i, col in enumerate(plot_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        fig.add_trace(
            go.Box(
                y=df[col],
                name=col,
                showlegend=False,
                boxpoints='outliers',
                hovertemplate=f'{col}: %{{y}}<extra></extra>'
            ),
            row=row,
            col=col_idx
        )

    fig.update_layout(
        title_text="📦 Outlier Analysis (Boxplots)",
        height=350 * n_rows,
        title_x=0.5
    )

    return fig


# -------------------------
# Helper: Numerical correlation heatmap
# -------------------------
def create_num_correlation_plotly(df, num_cols):
    """Create interactive numerical correlation heatmap"""
    if len(num_cols) < 2:
        return None

    corr_matrix = df[num_cols].corr()

    fig = go.Figure(data=[
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title="🔗 Numerical Correlation Matrix",
        height=600,
        width=700,
        title_x=0.5,
        template="plotly_white"
    )

    return fig


# -------------------------
# Helper: Cramer's V heatmap for categorical
# -------------------------
def create_cramers_v_plotly(df, cat_cols):
    """Create interactive Cramer's V heatmap"""
    if len(cat_cols) < 2:
        return None

    cramers_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)

    for col1 in cat_cols:
        for col2 in cat_cols:
            cramers_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

    cramers_matrix = cramers_matrix.astype(float)

    fig = go.Figure(data=[
        go.Heatmap(
            z=cramers_matrix.values,
            x=cramers_matrix.columns,
            y=cramers_matrix.index,
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            text=np.round(cramers_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x} vs %{y}<br>Cramers V: %{z:.3f}<extra></extra>'
        )
    ])

    fig.update_layout(
        title="🔗 Cramer's V Matrix (Categorical Associations)",
        height=600,
        width=700,
        title_x=0.5,
        template="plotly_white"
    )

    return fig


# -------------------------
# MAIN AGENT
# -------------------------
def preprocessing_agent(state):
    os.makedirs("outputs/eda", exist_ok=True)

    df = state["data"].copy()
    target = state["target_column"]

    print("\n===== BASIC EDA =====")
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())

    # Identify columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Store interactive figures to return
    interactive_plots = {}

    # -------------------------
    # STEP 1: HISTOGRAMS (Numerical)
    # -------------------------
    print("\n[1/6] Generating histograms for numerical variables...")
    hist_fig = create_histograms_plotly(df, num_cols, target)
    if hist_fig:
        hist_fig.write_html("outputs/eda/histograms.html")
        hist_fig.write_image("outputs/eda/histograms.png", scale=2)
        interactive_plots['histograms'] = hist_fig.to_json()

    # -------------------------
    # STEP 2: COUNTPLOTS (Categorical)
    # -------------------------
    print("\n[2/6] Generating countplots for categorical variables...")
    count_fig = create_countplots_plotly(df, cat_cols, target)
    if count_fig:
        count_fig.write_html("outputs/eda/countplots.html")
        count_fig.write_image("outputs/eda/countplots.png", scale=2)
        interactive_plots['countplots'] = count_fig.to_json()

    # -------------------------
    # STEP 3: OUTLIER ANALYSIS
    # -------------------------
    print("\n[3/6] Generating outlier analysis...")
    outlier_fig = create_outlier_plots_plotly(df, num_cols, target)
    if outlier_fig:
        outlier_fig.write_html("outputs/eda/outliers.html")
        outlier_fig.write_image("outputs/eda/outliers.png", scale=2)
        interactive_plots['outliers'] = outlier_fig.to_json()

    # -------------------------
    # HANDLE MISSING VALUES
    # -------------------------
    print("\n[*] Handling missing values...")
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown")

    # Update columns after filling
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # -------------------------
    # CAP OUTLIERS (1st / 99th percentile)
    # -------------------------
    print("\n[*] Capping outliers at 1st and 99th percentile...")
    df = cap_outliers(df, num_cols, target, lower_percentile=0.01, upper_percentile=0.99)

    # -------------------------
    # STEP 4: CORRELATION MATRIX (Numerical)
    # -------------------------
    print("\n[4/6] Generating numerical correlation matrix...")
    num_corr_fig = create_num_correlation_plotly(df, num_cols)
    if num_corr_fig:
        num_corr_fig.write_html("outputs/eda/corr_num.html")
        num_corr_fig.write_image("outputs/eda/corr_num.png", scale=2)
        interactive_plots['corr_numerical'] = num_corr_fig.to_json()

    # -------------------------
    # STEP 5: CRAMER'S V MATRIX (Categorical)
    # -------------------------
    print("\n[5/6] Generating Cramer's V matrix...")
    cramers_fig = create_cramers_v_plotly(df, cat_cols)
    if cramers_fig:
        cramers_fig.write_html("outputs/eda/corr_cat.html")
        cramers_fig.write_image("outputs/eda/corr_cat.png", scale=2)
        interactive_plots['corr_categorical'] = cramers_fig.to_json()

    # -------------------------
    # REMOVE LOW CORRELATION WITH TARGET
    # -------------------------
    print("\n[*] Removing features with low correlation to target...")
    df = drop_low_corr_with_target(df, num_cols, target, min_threshold=0.1)
    df = drop_low_cramers_with_target(df, cat_cols, target, min_threshold=0.1)

    # Update columns after low correlation removal
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # -------------------------
    # REMOVE MULTICOLLINEARITY
    # -------------------------
    print("\n[*] Removing multicollinearity...")
    df = drop_high_corr_num(df, threshold=0.8, target=target)

    # Update columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(cat_cols) > 1:
        df = drop_high_corr_cat(df, cat_cols, threshold=0.8, target=target)

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # -------------------------
    # ONE HOT ENCODING
    # -------------------------
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Save interactive plots metadata
    with open("outputs/eda/interactive_plots.json", "w") as f:
        json.dump(interactive_plots, f)

    # -------------------------
    # SCALING
    # -------------------------
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]

    # Save feature names before scaling
    feature_names = list(X.columns)

    # Get scaler type from state (default to standard)
    scaler_type = state.get("scaler_type", "standard").lower().strip()

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
        print(f"[*] Using MinMaxScaler (scales features to [0, 1])")
    elif scaler_type == "robust":
        scaler = RobustScaler()
        print(f"[*] Using RobustScaler (robust to outliers)")
    elif scaler_type == "none":
        scaler = None
        print(f"[*] Skipping feature scaling")
    else:
        scaler = StandardScaler()
        print(f"[*] Using StandardScaler (zero mean, unit variance)")

    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # -------------------------
    # TARGET TRANSFORMATION (for regression)
    # -------------------------
    transform_target = state.get("transform_target", False)
    target_transformer = None

    if transform_target and y.dtype in ["int64", "float64"]:
        # Check if target is skewed
        y_skew = skew(y.dropna())
        print(f"\n[*] Target skewness: {y_skew:.4f}")

        if abs(y_skew) > 0.5:  # Threshold for considering transformation
            print(f"[*] Applying log1p transformation to target (skewness > 0.5)")
            y = np.log1p(y)
            target_transformer = "log1p"
        else:
            print(f"[*] Target is not highly skewed, no transformation needed")

    # -------------------------
    # TRAIN TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return {
        **state,
        "processed_data": df_encoded,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "eda_plots": interactive_plots,
        "target_transformer": target_transformer
    }