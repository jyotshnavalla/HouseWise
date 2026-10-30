"""Train and evaluate the HouseWise six-feature Ames housing model."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "housing_data.csv"
ARTIFACT_PATH = ROOT / "housing_price_model.joblib"

FEATURES = [
    "Overall Qual",
    "Gr Liv Area",
    "Bedroom AbvGr",
    "Full Bath",
    "Year Built",
    "Garage Cars",
    "Neighborhood",
]
NUMERIC_FEATURES = FEATURES[:-1]
CATEGORICAL_FEATURES = ["Neighborhood"]
TARGET = "SalePrice"


def main():
    df = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=[TARGET]).copy()
    X = df[FEATURES].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    X["Neighborhood"] = X["Neighborhood"].fillna("Unknown").astype(str)
    y = pd.to_numeric(df[TARGET], errors="raise").astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Extra Trees Regressor": (
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [250],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": [0.8, 1.0],
            },
        ),
        "Random Forest Regressor": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": [250],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": [0.8, 1.0],
            },
        ),
        "Hist Gradient Boosting Regressor": (
            HistGradientBoostingRegressor(random_state=42),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [150],
                "model__max_leaf_nodes": [15, 31],
            },
        ),
    }

    best_search = None
    results = []
    for model_name, (estimator, parameter_grid) in candidates.items():
        preprocess = ColumnTransformer(
            [
                (
                    "numeric",
                    SimpleImputer(strategy="median"),
                    NUMERIC_FEATURES,
                ),
                (
                    "neighborhood",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
        pipeline = Pipeline(
            [
                ("preprocess", preprocess),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        test_predictions = search.predict(X_test)
        result = {
            "model_name": model_name,
            "cv_rmse": float(-search.best_score_),
            "test_mae": float(mean_absolute_error(y_test, test_predictions)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
            "test_r2": float(r2_score(y_test, test_predictions)),
            "best_params": search.best_params_,
        }
        results.append(result)
        print(
            f"{model_name}: CV RMSE ${result['cv_rmse']:,.0f}; "
            f"test MAE ${result['test_mae']:,.0f}; "
            f"test RMSE ${result['test_rmse']:,.0f}; "
            f"test R² {result['test_r2']:.3f}"
        )
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search = search
            best_result = result

    assert best_search is not None
    pipeline = best_search.best_estimator_
    metrics = {
        "mae": best_result["test_mae"],
        "rmse": best_result["test_rmse"],
        "r2": best_result["test_r2"],
        "cv_rmse": best_result["cv_rmse"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows": int(len(df)),
        "feature_count": len(FEATURES),
        "model_name": best_result["model_name"],
        "target": TARGET,
        "features": FEATURES,
        "model_comparison": results,
        "price_min": float(y.min()),
        "price_max": float(y.max()),
        "currency": "USD",
    }
    defaults = {
        feature: int(round(float(X[feature].median())))
        for feature in NUMERIC_FEATURES
    }
    neighborhood_counts = X["Neighborhood"].value_counts()
    defaults["Neighborhood"] = str(neighborhood_counts.index[0])
    artifact = {
        "pipeline": pipeline,
        "metrics": metrics,
        "defaults": defaults,
        "options": {
            "Neighborhood": sorted(X["Neighborhood"].unique().tolist())
        },
    }
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"\nSelected model: {metrics['model_name']}")
    print(f"Saved trained model: {ARTIFACT_PATH}")
    print(f"Selected test RMSE: ${metrics['rmse']:,.0f}; R²: {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
