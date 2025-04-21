# src/utils/metrics.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd # Added import for type hint if needed

# Keep calculate_regression_metrics as is
def calculate_regression_metrics(y_true, y_pred) -> dict:
    """
    Calcule les métriques d'évaluation pour un modèle de régression.

    Args:
        y_true (array-like): Valeurs réelles.
        y_pred (array-like): Valeurs prédites par le modèle.

    Returns:
        dict: dictionnaire contenant les métriques calculées (MSE, RMSE, MAE, R²).
    """
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae, # Corrected: Was calculating MAE twice, removed redundant calculation
        'r2': r2 # Corrected: Was calculating R2 twice, removed redundant calculation
    }


# Correct the function call in evaluate_model
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Évalue un modèle sur des données de test.

    Args:
        model: modèle à évaluer (doit avoir une méthode `predict`)
        X_test (DataFrame): Features de test
        y_test (Series): Valeurs réelles pour comparer les prédictions

    Returns:
        dict: métriques d'évaluation du modèle
    """
    if not hasattr(model, 'predict'):
        raise TypeError("Le modèle fourni doit avoir une méthode 'predict'.")

    y_pred = model.predict(X_test)

    # Ensure y_pred is flattened if model.predict returns multiple outputs or nested arrays
    # This might depend on the specific model's predict method structure
    # If predict returns a tuple (e.g., admissions, occupancy), this function needs adjustment
    # Assuming here model.predict returns a single array corresponding to y_test
    if isinstance(y_pred, tuple) and len(y_pred) > 0:
         # Assuming the first element is the relevant prediction if multiple are returned
         # This assumption might need refinement based on the model structure
        y_pred = y_pred[0]
    elif isinstance(y_pred, list):
         y_pred = np.array(y_pred) # Convert list to numpy array

    # Ensure y_pred is 1D array
    y_pred = np.ravel(y_pred)

    # *** Correction: Call the correct metrics function ***
    return calculate_regression_metrics(y_true=y_test, y_pred=y_pred)