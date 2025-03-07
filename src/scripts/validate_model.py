# src/scripts/validate_model.py

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error,
    explained_variance_score
)

# Import des modules locaux
from src.hospital_prediction.model import HospitalPredictionModel
from src.hospital_prediction.data_processor import (
    load_data_from_csv, 
    load_data_from_excel,
    split_train_test
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.model_validation')

def load_model_and_data(model_path: str, data_path: str):
    """
    Charger le modèle et les données pour validation
    
    Args:
        model_path: Chemin vers le modèle entraîné
        data_path: Chemin vers le fichier de données
    
    Returns:
        Tuple (modèle, données)
    """
    logger.info(f"Chargement du modèle depuis {model_path}")
    logger.info(f"Chargement des données depuis {data_path}")
    
    # Charger les données
    if data_path.endswith('.csv'):
        data = load_data_from_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        data = load_data_from_excel(data_path)
    else:
        raise ValueError(f"Format de fichier non supporté: {data_path}")
    
    # Charger le modèle
    model = HospitalPredictionModel()
    
    # Trouver les fichiers de modèle et de métadonnées
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('_admission_model.joblib')]
        if not model_files:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {model_path}")
        
        # Sélectionner le modèle le plus récent
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)
        admission_model_path = os.path.join(model_path, model_files[0])
        occupancy_model_path = admission_model_path.replace('_admission_model.joblib', '_occupancy_model.joblib')
        metadata_path = admission_model_path.replace('_admission_model.joblib', '_metadata.json')
    else:
        admission_model_path = model_path
        occupancy_model_path = model_path.replace('_admission_model.joblib', '_occupancy_model.joblib')
        metadata_path = model_path.replace('_admission_model.joblib', '_metadata.json')
    
    # Charger le modèle
    model.load(admission_model_path, occupancy_model_path, metadata_path)
    
    return model, data

def comprehensive_model_validation(
    model: HospitalPredictionModel, 
    data: pd.DataFrame, 
    test_size: float = 0.2, 
    output_dir: str = 'validation_results'
):
    """
    Effectue une validation complète du modèle
    
    Args:
        model: Modèle de prédiction hospitalière
        data: DataFrame contenant les données
        test_size: Proportion des données pour le test
        output_dir: Répertoire de sortie pour les résultats
    
    Returns:
        Dictionnaire des résultats de validation
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Préparer les données
    processed_data = model.preprocess_data(data)
    
    # Diviser les données
    X, y_admissions, y_occupancy = model.prepare_features_targets(
        processed_data, 'totalAdmissions', 'avgOccupancyRate'
    )
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train_admissions, y_test_admissions, y_train_occupancy, y_test_occupancy = model.train_test_split(
        X, y_admissions, y_occupancy, test_size=test_size
    )
    
    # Faire des prédictions
    y_pred_admissions, y_pred_occupancy, _ = model.predict(X_test)
    
    # Calculer les métriques détaillées
    validation_metrics = {
        'admissions': {
            'mae': mean_absolute_error(y_test_admissions, y_pred_admissions),
            'mse': mean_squared_error(y_test_admissions, y_pred_admissions),
            'rmse': np.sqrt(mean_squared_error(y_test_admissions, y_pred_admissions)),
            'r2': r2_score(y_test_admissions, y_pred_admissions),
            'mape': mean_absolute_percentage_error(y_test_admissions, y_pred_admissions),
            'explained_variance': explained_variance_score(y_test_admissions, y_pred_admissions)
        },
        'occupancy': {
            'mae': mean_absolute_error(y_test_occupancy, y_pred_occupancy),
            'mse': mean_squared_error(y_test_occupancy, y_pred_occupancy),
            'rmse': np.sqrt(mean_squared_error(y_test_occupancy, y_pred_occupancy)),
            'r2': r2_score(y_test_occupancy, y_pred_occupancy),
            'mape': mean_absolute_percentage_error(y_test_occupancy, y_pred_occupancy),
            'explained_variance': explained_variance_score(y_test_occupancy, y_pred_occupancy)
        }
    }
    
    # Visualisation des prédictions vs réalité
    plt.figure(figsize=(15, 10))
    
    # Sous-graphique pour les admissions
    plt.subplot(2, 1, 1)
    plt.scatter(y_test_admissions, y_pred_admissions, alpha=0.5)
    plt.plot([y_test_admissions.min(), y_test_admissions.max()], 
             [y_test_admissions.min(), y_test_admissions.max()], 
             'r--', lw=2)
    plt.title('Prédictions vs Réalité - Admissions')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    
    # Sous-graphique pour le taux d'occupation
    plt.subplot(2, 1, 2)
    plt.scatter(y_test_occupancy, y_pred_occupancy, alpha=0.5)
    plt.plot([y_test_occupancy.min(), y_test_occupancy.max()], 
             [y_test_occupancy.min(), y_test_occupancy.max()], 
             'r--', lw=2)
    plt.title('Prédictions vs Réalité - Taux d\'occupation')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_reality.png'))
    plt.close()
    
    # Résidus
    residuals_admissions = y_test_admissions - y_pred_admissions
    residuals_occupancy = y_test_occupancy - y_pred_occupancy
    
    plt.figure(figsize=(15, 10))
    
    # Sous-graphique pour les résidus des admissions
    plt.subplot(2, 1, 1)
    plt.scatter(y_pred_admissions, residuals_admissions, alpha=0.5)
    plt.title('Résidus - Admissions')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Sous-graphique pour les résidus du taux d'occupation
    plt.subplot(2, 1, 2)
    plt.scatter(y_pred_occupancy, residuals_occupancy, alpha=0.5)
    plt.title('Résidus - Taux d\'occupation')
    plt.xlabel('Prédictions')
    plt.ylabel('Résidus')
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals.png'))
    plt.close()
    
    # Distribution des résidus
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    sns.histplot(residuals_admissions, kde=True)
    plt.title('Distribution des Résidus - Admissions')
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    
    plt.subplot(2, 1, 2)
    sns.histplot(residuals_occupancy, kde=True)
    plt.title('Distribution des Résidus - Taux d\'occupation')
    plt.xlabel('Résidus')
    plt.ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_distribution.png'))
    plt.close()
    
    # Sauvegarder les métriques
    with open(os.path.join(output_dir, 'validation_metrics.json'), 'w') as f:
        json.dump(validation_metrics, f, indent=4)
    
    return validation_metrics

def main():
    """
    Point d'entrée principal pour la validation du modèle
    """
    parser = argparse.ArgumentParser(description="Validation du modèle de prédiction hospitalière")
    
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Chemin vers le modèle entraîné"
    )
    parser.add_argument(
        "--data-file", 
        type=str, 
        required=True, 
        help="Chemin vers le fichier de données de test"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="validation_results", 
        help="Répertoire de sortie pour les résultats de validation"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2, 
        help="Proportion des données à utiliser pour le test"
    )
    
    args = parser.parse_args()
    
    try:
        # Charger le modèle et les données
        model, data = load_model_and_data(args.model_path, args.data_file)
        
        # Effectuer la validation
        validation_results = comprehensive_model_validation(
            model, 
            data, 
            test_size=args.test_size, 
            output_dir=args.output_dir
        )
        
        # Afficher un résumé des résultats
        print("\nRésultats de validation du modèle :")
        
        print("\nMétriques pour les admissions :")
        print(f"  - MAE: {validation_results['admissions']['mae']:.4f}")
        print(f"  - MSE: {validation_results['admissions']['mse']:.4f}")
        print(f"  - RMSE: {validation_results['admissions']['rmse']:.4f}")
        print(f"  - R²: {validation_results['admissions']['r2']:.4f}")
        print(f"  - MAPE: {validation_results['admissions']['mape']:.4f}")
        
        print("\nMétriques pour le taux d'occupation :")
        print(f"  - MAE: {validation_results['occupancy']['mae']:.4f}")
        print(f"  - MSE: {validation_results['occupancy']['mse']:.4f}")
        print(f"  - RMSE: {validation_results['occupancy']['rmse']:.4f}")
        print(f"  - R²: {validation_results['occupancy']['r2']:.4f}")
        print(f"  - MAPE: {validation_results['occupancy']['mape']:.4f}")
        
        print("\nValidation terminée. Consultez les résultats détaillés dans le répertoire de sortie.")
    
    except Exception as e:
        logger.error(f"Erreur lors de la validation du modèle : {str(e)}")
        print(f"Erreur : {str(e)}")

# Point d'entrée du script
if __name__ == "__main__":
    main()