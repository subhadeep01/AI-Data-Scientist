# AI Data Scientist Agent

An automated machine learning pipeline built with Streamlit and LangChain that performs end-to-end data science tasks including EDA, preprocessing, model training with hyperparameter tuning, evaluation, and explainability.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)

## Features

- **Automated EDA**: Interactive visualizations for numerical and categorical data
- **Smart Preprocessing**: Missing value handling, outlier capping, correlation analysis, feature selection
- **Multiple ML Algorithms**: Logistic Regression, Random Forest, SVM, KNN, XGBoost, CatBoost
- **Hyperparameter Tuning**: GridSearchCV with k-fold cross-validation
- **Feature Scaling**: StandardScaler, MinMaxScaler, RobustScaler, or no scaling
- **Target Transformation**: Automatic log transformation for skewed regression targets
- **Cross-Validation**: Configurable k-fold CV for reliable model evaluation
- **SHAP Explainability**: Interactive beeswarm plots, bar charts, and waterfall plots
- **LLM-Generated Insights**: AI-powered analysis and recommendations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-DataScientist-Agent
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google API key:
```bash
# Create a .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Optional Dependencies

For additional algorithms:
```bash
pip install xgboost catboost
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then:
1. Upload a CSV file
2. Enter the target column name
3. Select problem type (Classification/Regression/Auto-detect)
4. Configure advanced options (optional):
   - **Feature Scaling**: Choose between StandardScaler, MinMaxScaler, RobustScaler, or None
   - **Transform Target**: Enable log transformation for skewed regression targets
   - **Cross-Validation Folds**: Set k-fold CV folds (3-10)
5. Click "Run Pipeline"

## Project Structure

```
AI-DataScientist-Agent/
├── app.py                     # Main Streamlit application
├── agents/
│   ├── preprocessing_agent.py # EDA and data preprocessing
│   ├── modeling_agent.py      # Model training and hyperparameter tuning
│   ├── evaluation_agent.py    # Model evaluation and metrics
│   └── explainability_agent.py # SHAP explanations
├── outputs/
│   ├── eda/                   # EDA visualizations
│   ├── evaluation/            # Evaluation plots
│   └── explainability/        # SHAP plots
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Agents Overview

### 1. Preprocessing Agent (`agents/preprocessing_agent.py`)
- Generates histograms for numerical features
- Creates count plots for categorical features
- Performs outlier analysis with boxplots
- Calculates correlation matrices (numerical) and Cramer's V (categorical)
- Handles missing values (median for numerical, mode for categorical)
- Caps outliers at 1st/99th percentile
- Removes low-correlation features
- Performs one-hot encoding
- Applies feature scaling based on user selection
- Applies target transformation for skewed regression data

### 2. Modeling Agent (`agents/modeling_agent.py`)
**Supported Algorithms:**

**Classification:**
- Logistic Regression (C: 0.1, 1, 10)
- Random Forest (n_estimators: 50, 100; max_depth: None, 10)
- SVM (C: 0.1, 1; kernel: linear, rbf)
- KNN (n_neighbors: 3, 5, 7; weights: uniform, distance)
- XGBoost (optional)
- CatBoost (optional)

**Regression:**
- Linear Regression
- Random Forest (n_estimators: 50, 100; max_depth: None, 10)
- SVR (C: 0.1, 1; kernel: linear, rbf)
- KNN (n_neighbors: 3, 5, 7; weights: uniform, distance)
- XGBoost (optional)
- CatBoost (optional)

Features:
- Auto-detection of problem type
- GridSearchCV for hyperparameter tuning
- K-fold cross-validation for model selection
- StratifiedKFold for classification, KFold for regression

### 3. Evaluation Agent (`agents/evaluation_agent.py`)
**Classification Metrics:**
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC (when predict_proba available)
- Confusion Matrix

**Regression Metrics:**
- R² Score, Adjusted R²
- MSE, RMSE, MAE

**Visualizations:**
- Interactive confusion matrix (Plotly)
- ROC curve with AUC
- Residual plots for regression
- Feature importance/coefficient plots
- Prediction distribution comparisons

### 4. Explainability Agent (`agents/explainability_agent.py`)
- SHAP summary plots (beeswarm style)
- Feature importance bar charts
- Waterfall plots for individual predictions
- Interactive Plotly visualizations
- Support for both TreeExplainer and general Explainer

## Configuration Options

### Feature Scaling
- **StandardScaler**: Zero mean, unit variance (default)
- **MinMaxScaler**: Scale to [0, 1] range
- **RobustScaler**: Robust to outliers using percentiles
- **None**: Skip scaling

### Cross-Validation
- Configurable folds: 3-10
- Uses CV scores for model selection (more reliable than single split)
- Reports mean ± standard deviation

### Target Transformation
- Automatically applies log1p transformation when skewness > 0.5
- Only applied to regression problems
- Helps with highly skewed target distributions

## Requirements

Core dependencies:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- shap
- langchain-google-genai
- python-dotenv

Optional:
- xgboost
- catboost

## Example Output

The pipeline generates:
1. **EDA Visualizations**: Histograms, count plots, box plots, correlation matrices
2. **Model Comparison**: CV scores for all algorithms
3. **Best Model**: Selected based on cross-validation performance
4. **Evaluation Plots**: Confusion matrix, ROC curve, residuals, feature importance
5. **SHAP Explanations**: Beeswarm plot showing feature impacts
6. **AI Report**: LLM-generated insights and recommendations

## Notes

- Ensure your Google API key has access to Gemini models
- Large datasets may take longer to process
- CatBoost and XGBoost are optional but recommended for better performance
- All plots are saved as both static PNG and interactive HTML files in `outputs/`
