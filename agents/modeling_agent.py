import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.utils.multiclass import type_of_target
import warnings
warnings.filterwarnings('ignore')

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Optional (if installed)
try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_available = True
except:
    xgb_available = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    catboost_available = True
except:
    catboost_available = False


def modeling_agent(state):
    X_train = state["X_train"]
    y_train = state["y_train"]
    X_test = state["X_test"]
    y_test = state["y_test"]

    # -------------------------
    # 1. Determine problem type
    # -------------------------
    user_problem_type = state.get("problem_type_user", "auto_detect")

    if user_problem_type in ["classification", "regression"]:
        problem_type = user_problem_type
        print(f"\nUser-specified Problem Type: {problem_type}")
    else:
        # Auto-detect
        target_type = type_of_target(y_train)
        if target_type in ["binary", "multiclass"]:
            problem_type = "classification"
        else:
            problem_type = "regression"
        print(f"\nAuto-detected Problem Type: {problem_type} (target_type={target_type})")

    # -------------------------
    # 2. Define models
    # -------------------------
    if problem_type == "classification":

        models = {
            "logistic_regression": {
                "model": LogisticRegression(max_iter=3000),
                "params": {"C": [0.1, 1, 10]}
            },
            "random_forest": {
                "model": RandomForestClassifier(),
                "params": {"n_estimators": [50, 100], "max_depth": [None, 10]}
            },
            "svm": {
                "model": SVC(),
                "params": {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
            },
            "knn": {
                "model": KNeighborsClassifier(),
                "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
            }
        }

        if xgb_available:
            models["xgboost"] = {
                "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "params": {"n_estimators": [50, 100], "max_depth": [3, 5]}
            }

        if catboost_available:
            models["catboost"] = {
                "model": CatBoostClassifier(verbose=0, random_state=42),
                "params": {"iterations": [100, 200], "depth": [4, 6], "learning_rate": [0.1]}
            }

    else:  # regression

        models = {
            "linear_regression": {
                "model": LinearRegression(),
                "params": {}
            },
            "random_forest": {
                "model": RandomForestRegressor(),
                "params": {"n_estimators": [50, 100], "max_depth": [None, 10]}
            },
            "svr": {
                "model": SVR(),
                "params": {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
            },
            "knn": {
                "model": KNeighborsRegressor(),
                "params": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}
            }
        }

        if xgb_available:
            models["xgboost"] = {
                "model": XGBRegressor(),
                "params": {"n_estimators": [50, 100], "max_depth": [3, 5]}
            }

        if catboost_available:
            models["catboost"] = {
                "model": CatBoostRegressor(verbose=0, random_state=42),
                "params": {"iterations": [100, 200], "depth": [4, 6], "learning_rate": [0.1]}
            }

    # -------------------------
    # 3. Get CV Folds from state
    # -------------------------
    cv_folds = state.get("cv_folds", 5)
    print(f"\n[*] Using {cv_folds}-fold cross-validation for model evaluation")

    # -------------------------
    # 4. Train & Select Best Model with CV
    # -------------------------
    best_score = -float("inf")
    best_model = None
    best_name = None
    results = {}
    cv_results = {}

    for name, config in models.items():
        print(f"\nTraining {name}...")

        # Use GridSearchCV for hyperparameter tuning with k-fold CV
        if config["params"]:
            if problem_type == "classification":
                cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = 'accuracy'
            else:
                cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = 'r2'

            grid = GridSearchCV(
                config["model"],
                config["params"],
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=True
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            best_params = grid.best_params_
            print(f"  Best params: {best_params}")
        else:
            model = config["model"]
            model.fit(X_train, y_train)
            best_params = {}

        # Perform k-fold cross-validation for unbiased performance estimate
        if problem_type == "classification":
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Also get test predictions
            preds = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, preds)
            test_f1 = f1_score(y_test, preds, average="weighted")

            print(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
            print(f"  Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

            results[name] = {
                "cv_accuracy_mean": cv_mean,
                "cv_accuracy_std": cv_std,
                "test_accuracy": test_accuracy,
                "test_f1": test_f1,
                "best_params": best_params
            }
            score = cv_mean  # Use CV score for model selection

        else:  # regression
            cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Also calculate RMSE via CV
            neg_mse_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy,
                                           scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-neg_mse_scores.mean())

            # Test predictions
            preds = model.predict(X_test)
            test_r2 = r2_score(y_test, preds)
            test_mse = mean_squared_error(y_test, preds)
            test_rmse = np.sqrt(test_mse)

            print(f"  CV R2: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
            print(f"  CV RMSE: {cv_rmse:.4f}")
            print(f"  Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")

            results[name] = {
                "cv_r2_mean": cv_mean,
                "cv_r2_std": cv_std,
                "cv_rmse": cv_rmse,
                "test_r2": test_r2,
                "test_rmse": test_rmse,
                "test_mse": test_mse,
                "best_params": best_params
            }
            score = cv_mean  # Use CV R2 for model selection

        # Store CV scores for detailed reporting
        cv_results[name] = {
            "cv_scores": cv_scores.tolist(),
            "cv_mean": cv_mean,
            "cv_std": cv_std
        }

        # -------------------------
        # 5. Select Best Model
        # -------------------------
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\n{'='*50}")
    print(f"Best Model: {best_name}")
    print(f"CV Score: {best_score:.4f}")
    print(f"{'='*50}")

    return {
        **state,
        "best_model": best_model,
        "model_name": best_name,
        "model_results": results,
        "cv_results": cv_results,
        "problem_type": problem_type,
        "cv_folds_used": cv_folds
    }