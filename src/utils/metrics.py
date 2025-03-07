# src/utils/metrics.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


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
        'mae': mean_absolute_error(y_true=y_true, y_pred=y_pred),
        'r2': r2_score(y_true=y_true, y_pred=y_pred)
    }


def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle sur des données de test.

    Args:
        model: modèle à évaluer
        X_test (DataFrame): Features de test
        y_test (Series): Valeurs réelles pour comparer les prédictions

    Returns:
        dict: métriques d'évaluation du modèle
    """
    y_pred = model.predict(X_test)
    return calculate_metrics(y_true=y_test, y_pred=y_pred)
