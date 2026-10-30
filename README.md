# HouseWise — Smart Housing Price Prediction

HouseWise is a single-page Streamlit application that estimates a home's sale price from seven dataset-backed details: overall quality, living area, bedrooms, full bathrooms, year built, garage capacity, and locality/neighborhood. It also collects previous-owner count for information only; that input is excluded from prediction because the supplied dataset does not record ownership history.

## Dataset and scope

The model uses the included 2,930-record Ames Housing dataset. Its target, `SalePrice`, is in **US dollars** and reflects historical Ames, Iowa transactions. All seven prediction inputs are actual dataset features; there are no hidden or fabricated feature values. Missing numeric values are imputed with medians calculated from the training split by the saved pipeline, and neighborhood is one-hot encoded.

The prediction is an educational estimate based on historical sales, not a current-market appraisal or guaranteed selling price.

## Run locally (Windows / VS Code)

From this `HouseWise_AI` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local URL, usually `http://localhost:8501`.

## Model training and evaluation

The saved `housing_price_model.joblib` contains the complete imputation-and-estimator pipeline and is loaded directly by the app; the app does not retrain on launch.

To compare models and rebuild the artifact:

```powershell
python train_model.py
```

Training reserves a fixed 20% holdout test set (`random_state=42`), uses three-fold cross-validation and a small parameter grid to tune Extra Trees, Random Forest, and Hist Gradient Boosting regressors on the remaining training data, then selects the model with the lowest cross-validation RMSE. The script prints cross-validation RMSE and holdout RMSE, MAE, and R² for each model. The selected model, preprocessing, feature order, and measured metrics are saved in the artifact. Metrics are empirical results, not guaranteed performance.

## Project files

- `app.py` — single-page prediction interface
- `train_model.py` — cross-validated model selection and artifact generation
- `housing_data.csv` — Ames Housing data
- `housing_price_model.joblib` — saved trained preprocessing and prediction pipeline
- `requirements.txt` — runtime dependencies
