# src/hospital_prediction/train.py

import pandas as pd
import numpy as np
import logging
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, Dict, Any, List, Union, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import des modules locaux
from src.hospital_prediction.model import HospitalPredictionModel
from src.utils.visualization import plot_feature_importance, save_figure

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.train')

def train_admission_model(X_train: pd.DataFrame, y_train: pd.Series, 
                          model_params: Dict[str, Any] = None, cv: int = 5) -> Tuple[Any, Dict[str, float]]:
    """
    Entraîne un modèle pour prédire les admissions hospitalières
    
    Args:
        X_train: Features d'entraînement
        y_train: Variable cible (admissions)
        model_params: Paramètres pour le modèle
        cv: Nombre de plis pour la validation croisée
    
    Returns:
        Tuple (modèle entraîné, métriques d'entraînement)
    """
    logger.info("Entraînement du modèle de prédiction des admissions")
    
    # Créer une instance du modèle
    model = HospitalPredictionModel()
    
    # Configuration par défaut des paramètres
    if model_params is None:
        model_params = {
            'gb_n_estimators': 100,
            'gb_learning_rate': 0.1,
            'gb_max_depth': 4
        }
    
    # Extraction des métriques du modèle
    metrics = {}
    
    # Entraîner uniquement le modèle d'admissions
    # On passe un DataFrame vide pour y_occupancy car on ne l'utilise pas ici
    dummy_occupancy = pd.Series(np.zeros(len(y_train)))
    
    model.train(X_train, y_train, dummy_occupancy)
    
    # Évaluer sur les données d'entraînement
    y_pred = model.admission_model.predict(X_train)
    
    metrics['mse'] = mean_squared_error(y_train, y_pred)
    metrics['mae'] = mean_absolute_error(y_train, y_pred)
    metrics['r2'] = r2_score(y_train, y_pred)
    
    logger.info(f"Modèle d'admissions entraîné - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return model.admission_model, metrics

def train_occupancy_model(X_train: pd.DataFrame, y_train: pd.Series, 
                           model_params: Dict[str, Any] = None, cv: int = 5) -> Tuple[Any, Dict[str, float]]:
    """
    Entraîne un modèle pour prédire le taux d'occupation des lits
    
    Args:
        X_train: Features d'entraînement
        y_train: Variable cible (taux d'occupation)
        model_params: Paramètres pour le modèle
        cv: Nombre de plis pour la validation croisée
    
    Returns:
        Tuple (modèle entraîné, métriques d'entraînement)
    """
    logger.info("Entraînement du modèle de prédiction du taux d'occupation")
    
    # Créer une instance du modèle
    model = HospitalPredictionModel()
    
    # Configuration par défaut des paramètres
    if model_params is None:
        model_params = {
            'rf_n_estimators': 100,
            'rf_max_depth': None,
            'rf_min_samples_split': 2
        }
    
    # Extraction des métriques du modèle
    metrics = {}
    
    # Entraîner uniquement le modèle de taux d'occupation
    # On passe un DataFrame vide pour y_admissions car on ne l'utilise pas ici
    dummy_admissions = pd.Series(np.zeros(len(y_train)))
    
    model.train(X_train, dummy_admissions, y_train)
    
    # Évaluer sur les données d'entraînement
    y_pred = model.occupancy_model.predict(X_train)
    
    metrics['mse'] = mean_squared_error(y_train, y_pred)
    metrics['mae'] = mean_absolute_error(y_train, y_pred)
    metrics['r2'] = r2_score(y_train, y_pred)
    
    logger.info(f"Modèle de taux d'occupation entraîné - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return model.occupancy_model, metrics

def optimize_model_parameters(X: pd.DataFrame, y_admissions: pd.Series, y_occupancy: pd.Series,
                             param_grid: Dict[str, List[Any]] = None, 
                             cv: int = 5, n_jobs: int = -1) -> Dict[str, Any]:
    """
    Optimise les paramètres des modèles en utilisant la validation croisée
    
    Args:
        X: Features
        y_admissions: Variable cible (admissions)
        y_occupancy: Variable cible (taux d'occupation)
        param_grid: Grille de paramètres à tester
        cv: Nombre de plis pour la validation croisée
        n_jobs: Nombre de jobs parallèles
    
    Returns:
        Dictionnaire des meilleurs paramètres
    """
    logger.info("Optimisation des paramètres des modèles")
    
    # Grille de paramètres par défaut
    if param_grid is None:
        param_grid = {
            'gb_params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5]
            },
            'rf_params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        }
    
    # Configuration de la validation croisée adaptée aux séries temporelles
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Créer une instance du modèle
    model = HospitalPredictionModel()
    
    # Optimisation des paramètres pour le modèle d'admissions
    logger.info("Optimisation des paramètres pour le modèle d'admissions")
    best_params = {}
    
    # Préparer le modèle pour GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(random_state=42)
    
    # Configuration de la recherche par grille
    gb_grid = GridSearchCV(
        gb, 
        param_grid=param_grid['gb_params'],
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    gb_grid.fit(X, y_admissions)
    best_params['gb_params'] = gb_grid.best_params_
    
    logger.info(f"Meilleurs paramètres pour le modèle d'admissions: {best_params['gb_params']}")
    
    # Optimisation des paramètres pour le modèle de taux d'occupation
    logger.info("Optimisation des paramètres pour le modèle de taux d'occupation")
    
    # Préparer le modèle pour GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    
    # Configuration de la recherche par grille
    rf_grid = GridSearchCV(
        rf, 
        param_grid=param_grid['rf_params'],
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=n_jobs,
        verbose=1
    )
    
    rf_grid.fit(X, y_occupancy)
    best_params['rf_params'] = rf_grid.best_params_
    
    logger.info(f"Meilleurs paramètres pour le modèle de taux d'occupation: {best_params['rf_params']}")
    
    return best_params

def train_full_hospital_model(df: pd.DataFrame, test_size: float = 0.2, 
                              random_state: int = 42, model_dir: str = 'models',
                              optimize_params: bool = False) -> Tuple[HospitalPredictionModel, Dict[str, Any], Dict[str, str]]:
    """
    Fonction principale pour entraîner le modèle complet de prédiction hospitalière
    
    Args:
        df: DataFrame contenant les données prétraitées
        test_size: Proportion des données à utiliser pour le test
        random_state: Graine aléatoire pour la reproductibilité
        model_dir: Répertoire pour sauvegarder les modèles
        optimize_params: Si True, optimise les paramètres des modèles
    
    Returns:
        Tuple (modèle entraîné, métriques d'évaluation, chemins des fichiers sauvegardés)
    """
    logger.info("Entraînement du modèle complet de prédiction hospitalière")
    
    # Vérifier les colonnes nécessaires
    required_cols = ['totalAdmissions', 'avgOccupancyRate']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Colonne manquante: {col}")
            raise ValueError(f"La colonne {col} est requise pour l'entraînement du modèle")
    
    # Créer une instance du modèle
    model = HospitalPredictionModel(model_dir=model_dir)
    
    # Prétraiter les données
    df_processed = model.preprocess_data(df)
    
    # Préparer les features et les cibles
    X, y_admissions, y_occupancy = model.prepare_features_targets(
        df_processed, 'totalAdmissions', 'avgOccupancyRate'
    )
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train_admissions, y_test_admissions, y_train_occupancy, y_test_occupancy = train_test_split(
        X, y_admissions, y_occupancy, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Division des données: {len(X_train)} exemples d'entraînement, {len(X_test)} exemples de test")
    
    # Optimiser les paramètres si demandé
    if optimize_params:
        logger.info("Optimisation des paramètres des modèles")
        best_params = optimize_model_parameters(X_train, y_train_admissions, y_train_occupancy)
        
        # Configurer les paramètres du modèle
        model_params = {
            'gb_n_estimators': best_params['gb_params']['n_estimators'],
            'gb_learning_rate': best_params['gb_params']['learning_rate'],
            'gb_max_depth': best_params['gb_params']['max_depth'],
            'rf_n_estimators': best_params['rf_params']['n_estimators'],
            'rf_max_depth': best_params['rf_params']['max_depth'],
            'rf_min_samples_split': best_params['rf_params']['min_samples_split']
        }
    else:
        # Paramètres par défaut
        model_params = None
    
    # Entraîner les modèles
    model.train(X_train, y_train_admissions, y_train_occupancy)
    
    # Évaluer les modèles
    test_metrics = model.evaluate(X_test, y_test_admissions, y_test_occupancy)
    
    # Sauvegarder les modèles
    model_paths = model.save()
    
    # Générer et sauvegarder les visualisations d'importance des features
    fig_importance = model.plot_feature_importance()
    
    # Sauvegarder la figure si un répertoire a été spécifié
    if model_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_file = os.path.join(model_dir, f"feature_importance_{timestamp}.png")
        plt.savefig(fig_file)
        logger.info(f"Figure d'importance des features sauvegardée dans {fig_file}")
    
    logger.info("Entraînement du modèle complet terminé")
    
    return model, test_metrics, model_paths

def evaluate_model_performance(model: HospitalPredictionModel, X_test: pd.DataFrame, 
                              y_test_admissions: pd.Series, y_test_occupancy: pd.Series,
                              output_dir: str = None) -> Dict[str, Dict[str, float]]:
    """
    Évalue la performance des modèles sur un ensemble de test
    
    Args:
        model: Modèle hospitalier entraîné
        X_test: Features de test
        y_test_admissions: Cible de test pour les admissions
        y_test_occupancy: Cible de test pour le taux d'occupation
        output_dir: Répertoire pour sauvegarder les résultats d'évaluation
    
    Returns:
        Dictionnaire des métriques d'évaluation
    """
    logger.info("Évaluation de la performance des modèles")
    
    # Prédictions sur l'ensemble de test
    y_pred_admissions = model.admission_model.predict(X_test)
    y_pred_occupancy = model.occupancy_model.predict(X_test)
    
    # Calculer les métriques pour les admissions
    metrics = {
        'admissions': {
            'mse': mean_squared_error(y_test_admissions, y_pred_admissions),
            'mae': mean_absolute_error(y_test_admissions, y_pred_admissions),
            'r2': r2_score(y_test_admissions, y_pred_admissions),
            'rmse': np.sqrt(mean_squared_error(y_test_admissions, y_pred_admissions))
        },
        'occupancy': {
            'mse': mean_squared_error(y_test_occupancy, y_pred_occupancy),
            'mae': mean_absolute_error(y_test_occupancy, y_pred_occupancy),
            'r2': r2_score(y_test_occupancy, y_pred_occupancy),
            'rmse': np.sqrt(mean_squared_error(y_test_occupancy, y_pred_occupancy))
        }
    }
    
    logger.info(f"Modèle d'admissions - MSE: {metrics['admissions']['mse']:.4f}, MAE: {metrics['admissions']['mae']:.4f}, "
          f"R²: {metrics['admissions']['r2']:.4f}, RMSE: {metrics['admissions']['rmse']:.4f}")
    
    logger.info(f"Modèle de taux d'occupation - MSE: {metrics['occupancy']['mse']:.4f}, MAE: {metrics['occupancy']['mae']:.4f}, "
          f"R²: {metrics['occupancy']['r2']:.4f}, RMSE: {metrics['occupancy']['rmse']:.4f}")
    
    # Sauvegarder les métriques dans un fichier JSON si un répertoire a été spécifié
    if output_dir:
        # Créer le répertoire s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")
        
        # Générer le nom du fichier avec un timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(output_dir, f"model_metrics_{timestamp}.json")
        
        # Sauvegarder les métriques
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Métriques d'évaluation sauvegardées dans {metrics_file}")
    
    return metrics

def train_with_cross_validation(df: pd.DataFrame, n_splits: int = 5, target_cols: List[str] = None,
                               random_state: int = 42) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
    """
    Entraîne le modèle en utilisant la validation croisée temporelle
    
    Args:
        df: DataFrame contenant les données prétraitées
        n_splits: Nombre de plis pour la validation croisée
        target_cols: Liste des colonnes cibles
        random_state: Graine aléatoire pour la reproductibilité
    
    Returns:
        Tuple (métriques par pli, métriques moyennes)
    """
    logger.info(f"Entraînement avec validation croisée ({n_splits} plis)")
    
    # Si aucune colonne cible n'est spécifiée, utiliser les colonnes par défaut
    if target_cols is None:
        target_cols = ['totalAdmissions', 'avgOccupancyRate']
    
    # Vérifier les colonnes nécessaires
    for col in target_cols:
        if col not in df.columns:
            logger.error(f"Colonne manquante: {col}")
            raise ValueError(f"La colonne {col} est requise pour l'entraînement du modèle")
    
    # Créer une instance du modèle
    model = HospitalPredictionModel()
    
    # Prétraiter les données
    df_processed = model.preprocess_data(df)
    
    # Préparer les features et les cibles
    X, y_admissions, y_occupancy = model.prepare_features_targets(
        df_processed, target_cols[0], target_cols[1]
    )
    
    # Configuration de la validation croisée temporelle
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialiser les listes pour stocker les métriques
    metrics_by_fold = {
        'admissions': {'mse': [], 'mae': [], 'r2': []},
        'occupancy': {'mse': [], 'mae': [], 'r2': []}
    }
    
    # Effectuer la validation croisée
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        logger.info(f"Pli {i+1}/{n_splits}")
        
        # Diviser les données
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_admissions, y_test_admissions = y_admissions.iloc[train_index], y_admissions.iloc[test_index]
        y_train_occupancy, y_test_occupancy = y_occupancy.iloc[train_index], y_occupancy.iloc[test_index]
        
        # Entraîner les modèles
        model.train(X_train, y_train_admissions, y_train_occupancy)
        
        # Évaluer les modèles
        y_pred_admissions = model.admission_model.predict(X_test)
        y_pred_occupancy = model.occupancy_model.predict(X_test)
        
        # Calculer les métriques pour les admissions
        metrics_by_fold['admissions']['mse'].append(mean_squared_error(y_test_admissions, y_pred_admissions))
        metrics_by_fold['admissions']['mae'].append(mean_absolute_error(y_test_admissions, y_pred_admissions))
        metrics_by_fold['admissions']['r2'].append(r2_score(y_test_admissions, y_pred_admissions))
        
        # Calculer les métriques pour le taux d'occupation
        metrics_by_fold['occupancy']['mse'].append(mean_squared_error(y_test_occupancy, y_pred_occupancy))
        metrics_by_fold['occupancy']['mae'].append(mean_absolute_error(y_test_occupancy, y_pred_occupancy))
        metrics_by_fold['occupancy']['r2'].append(r2_score(y_test_occupancy, y_pred_occupancy))
    
    # Calculer les métriques moyennes
    avg_metrics = {
        'admissions': {
            'mse': np.mean(metrics_by_fold['admissions']['mse']),
            'mae': np.mean(metrics_by_fold['admissions']['mae']),
            'r2': np.mean(metrics_by_fold['admissions']['r2'])
        },
        'occupancy': {
            'mse': np.mean(metrics_by_fold['occupancy']['mse']),
            'mae': np.mean(metrics_by_fold['occupancy']['mae']),
            'r2': np.mean(metrics_by_fold['occupancy']['r2'])
        }
    }
    
    logger.info(f"Validation croisée terminée - "
          f"Modèle d'admissions: MSE={avg_metrics['admissions']['mse']:.4f}, MAE={avg_metrics['admissions']['mae']:.4f}, R²={avg_metrics['admissions']['r2']:.4f} | "
          f"Modèle de taux d'occupation: MSE={avg_metrics['occupancy']['mse']:.4f}, MAE={avg_metrics['occupancy']['mae']:.4f}, R²={avg_metrics['occupancy']['r2']:.4f}")
    
    return metrics_by_fold, avg_metrics

def load_and_train_model(data_path: str, output_dir: str = 'models', test_size: float = 0.2,
                         optimize: bool = False, random_state: int = 42) -> Tuple[HospitalPredictionModel, Dict[str, Any]]:
    """
    Charge les données et entraîne le modèle de prédiction hospitalière
    
    Args:
        data_path: Chemin vers le fichier de données
        output_dir: Répertoire pour sauvegarder les modèles et les résultats
        test_size: Proportion des données à utiliser pour le test
        optimize: Si True, optimise les paramètres des modèles
        random_state: Graine aléatoire pour la reproductibilité
    
    Returns:
        Tuple (modèle entraîné, métriques et informations)
    """
    logger.info(f"Chargement des données depuis {data_path} et entraînement du modèle")
    
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Répertoire créé: {output_dir}")
    
    # Charger les données
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(data_path)
    else:
        logger.error(f"Format de fichier non supporté: {data_path}")
        raise ValueError(f"Format de fichier non supporté: {data_path}")
    
    logger.info(f"Données chargées: {len(df)} enregistrements")
    
    # Entraîner le modèle
    model, metrics, model_paths = train_full_hospital_model(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        model_dir=output_dir,
        optimize_params=optimize
    )
    
    # Créer un dictionnaire pour stocker les résultats
    results = {
        'metrics': metrics,
        'model_paths': model_paths,
        'training_info': {
            'data_path': data_path,
            'output_dir': output_dir,
            'test_size': test_size,
            'optimize': optimize,
            'random_state': random_state,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(df),
            'n_features': len(model.feature_names)
        }
    }
    
    # Sauvegarder les résultats dans un fichier JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f"training_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        # Convertir les chemins en chaînes relatives
        for key, path in results['model_paths'].items():
            results['model_paths'][key] = os.path.relpath(path, output_dir)
        json.dump(results, f, indent=4)
    
    logger.info(f"Résultats d'entraînement sauvegardés dans {results_file}")
    
    return model, results