# src/scripts/train_model.py

import os
import argparse
import logging
import pandas as pd
import json
from datetime import datetime

# Import des modules locaux
from src.hospital_prediction.data_processor import (
    load_data_from_excel, 
    load_data_from_csv, 
    preprocess_hospital_data, 
    aggregate_daily_data,
    save_processed_data
)
from src.hospital_prediction.train import (
    train_full_hospital_model,
    train_with_cross_validation,
    optimize_model_parameters
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.train_script')

def prepare_training_data(
    input_file: str, 
    output_dir: str = 'data/processed', 
    force_preprocess: bool = False
) -> pd.DataFrame:
    """
    Prépare les données pour l'entraînement
    
    Args:
        input_file: Chemin vers le fichier d'entrée (Excel ou CSV)
        output_dir: Répertoire de sortie pour les données prétraitées
        force_preprocess: Force le retraitement des données
    
    Returns:
        DataFrame préparé pour l'entraînement
    """
    logger.info(f"Préparation des données d'entraînement à partir de {input_file}")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données brutes
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        raw_data = load_data_from_excel(input_file)
    elif input_file.endswith('.csv'):
        raw_data = load_data_from_csv(input_file)
    else:
        raise ValueError(f"Format de fichier non supporté: {input_file}")
    
    # Prétraiter les données
    processed_data = preprocess_hospital_data(raw_data)
    
    # Agréger les données au niveau journalier
    daily_data = aggregate_daily_data(processed_data)
    
    # Sauvegarder les données prétraitées
    processed_file_path = save_processed_data(daily_data, output_dir)
    
    logger.info(f"Données prétraitées sauvegardées dans {processed_file_path}")
    
    return daily_data

def train_model(
    input_file: str, 
    output_dir: str = 'models', 
    test_size: float = 0.2, 
    optimize: bool = False,
    cv_analysis: bool = False
) -> dict:
    """
    Entraîne le modèle de prédiction hospitalière
    
    Args:
        input_file: Chemin vers le fichier de données
        output_dir: Répertoire de sortie pour les modèles
        test_size: Proportion des données à utiliser pour le test
        optimize: Active l'optimisation des hyperparamètres
        cv_analysis: Active l'analyse par validation croisée
    
    Returns:
        Dictionnaire des résultats d'entraînement
    """
    logger.info("Démarrage de l'entraînement du modèle")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Préparer les données
    daily_data = prepare_training_data(input_file)
    
    # Résultats de l'entraînement
    training_results = {
        'input_file': input_file,
        'output_dir': output_dir,
        'timestamp': datetime.now().isoformat(),
        'test_size': test_size,
        'optimize_params': optimize
    }
    
    # Entraînement du modèle complet
    model, metrics, model_paths = train_full_hospital_model(
        daily_data, 
        test_size=test_size, 
        model_dir=output_dir,
        optimize_params=optimize
    )
    
    # Ajouter les métriques aux résultats
    training_results['metrics'] = metrics
    training_results['model_paths'] = model_paths
    
    # Analyse par validation croisée si demandée
    if cv_analysis:
        logger.info("Démarrage de l'analyse par validation croisée")
        metrics_by_fold, avg_metrics = train_with_cross_validation(daily_data)
        training_results['cv_metrics_by_fold'] = metrics_by_fold
        training_results['cv_avg_metrics'] = avg_metrics
    
    # Sauvegarder les résultats dans un fichier JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'training_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=4)
    
    logger.info(f"Résultats d'entraînement sauvegardés dans {results_file}")
    
    return training_results

def main():
    """
    Point d'entrée principal pour l'entraînement du modèle
    """
    # Configuration de l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Entraînement du modèle de prédiction hospitalière")
    
    # Arguments requis
    parser.add_argument(
        "--data-file", 
        type=str, 
        required=True, 
        help="Chemin vers le fichier de données (Excel ou CSV)"
    )
    
    # Arguments optionnels
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models", 
        help="Répertoire de sortie pour les modèles (défaut: models)"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2, 
        help="Proportion des données à utiliser pour le test (défaut: 0.2)"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true", 
        help="Activer l'optimisation des hyperparamètres"
    )
    parser.add_argument(
        "--cross-validation", 
        action="store_true", 
        help="Effectuer une analyse par validation croisée"
    )
    
    # Parsing des arguments
    args = parser.parse_args()
    
    try:
        # Lancer l'entraînement
        results = train_model(
            input_file=args.data_file,
            output_dir=args.output_dir,
            test_size=args.test_size,
            optimize=args.optimize,
            cv_analysis=args.cross_validation
        )
        
        # Afficher un résumé des résultats
        print("\nRésumé de l'entraînement:")
        print(f"  - Fichier de données: {results['input_file']}")
        print(f"  - Répertoire de sortie: {results['output_dir']}")
        print("\nMétriques du modèle d'admissions:")
        admission_metrics = results['metrics']['admissions']
        print(f"  - MSE: {admission_metrics.get('mse', 'N/A'):.4f}")
        print(f"  - MAE: {admission_metrics.get('mae', 'N/A'):.4f}")
        print(f"  - R²: {admission_metrics.get('r2', 'N/A'):.4f}")
        
        print("\nMétriques du modèle de taux d'occupation:")
        occupancy_metrics = results['metrics']['occupancy']
        print(f"  - MSE: {occupancy_metrics.get('mse', 'N/A'):.4f}")
        print(f"  - MAE: {occupancy_metrics.get('mae', 'N/A'):.4f}")
        print(f"  - R²: {occupancy_metrics.get('r2', 'N/A'):.4f}")
        
        print("\nModèle entraîné avec succès!")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle : {str(e)}")
        print(f"Erreur : {str(e)}")

# Point d'entrée du script
if __name__ == "__main__":
    main()