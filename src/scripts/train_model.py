# src/scripts/train_model.py

import sys
import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime

# Correction explicite du chemin relatif pour les imports Python
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.hospital_prediction.data_processor import (
    load_data_from_excel, 
    load_data_from_csv, 
    preprocess_hospital_data, 
    save_processed_data,
    aggregate_daily_data,
    generate_synthetic_data
)
from src.hospital_prediction.train import (
    train_full_hospital_model,
    train_with_cross_validation,
    optimize_model_parameters
)

# Configuration du logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.train_model')


def prepare_data(input_file: str, output_dir: str = 'data/processed') -> pd.DataFrame:
    logger.info(f"Chargement et prétraitement des données depuis : {input_file}")

    # Charger les données
    if input_file.endswith('.xlsx'):
        raw_data = load_data_from_excel(input_file)
    elif input_file.endswith('.csv'):
        raw_data = pd.read_csv(input_file)
    else:
        raise ValueError(f"Format de fichier non supporté : {input_file}")

    # Prétraitement des données au niveau patient
    processed_data = raw_data.pipe(preprocess_hospital_data)
    
    # Agrégation des données au niveau journalier
    daily_data = aggregate_daily_data(processed_data)
    
    # Sauvegarder les données traitées
    processed_data_path = save_processed_data(daily_data, output_dir, "daily_aggregated_data.csv")
    logger.info(f"Données agrégées sauvegardées dans {processed_data_path}")

    return daily_data


def train_model(input_file: str, model_dir: str = 'models', output_dir: str = 'data/processed',
                optimize: bool = False, cross_validation: bool = False):

    logger.info("Démarrage de l'entraînement du modèle...")

    # Vérifier si le fichier est "synthetic" et le générer si nécessaire
    if "synthetic" in input_file and not os.path.exists(input_file):
        logger.info("Génération de données synthétiques...")
        synthetic_data = generate_synthetic_data(365)  # Générer un an de données
        input_file = save_processed_data(synthetic_data, os.path.dirname(input_file), os.path.basename(input_file))
        data = synthetic_data
    else:
        # Préparation des données normales
        data = prepare_data(input_file, output_dir)

    # Entraînement du modèle
    if optimize:
        logger.info("Optimisation des hyperparamètres activée.")
        model, metrics, _ = train_full_hospital_model(data, optimize_params=True)
    elif cross_validation:
        logger.info("Validation croisée activée.")
        metrics_by_fold, metrics = train_with_cross_validation(data)
        model, _, _ = train_full_hospital_model(data)  # Entraîner le modèle final
    else:
        model, metrics, _ = train_full_hospital_model(data)

    # Sauvegarde des résultats
    training_results = {
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    # Assurer que le répertoire existe
    os.makedirs(model_dir, exist_ok=True)
    
    result_path = os.path.join(model_dir, 'training_results.json')
    with open(result_path, 'w') as f:
        json.dump(training_results, f, indent=4)

    logger.info(f"Entraînement terminé avec succès, résultats sauvegardés dans {model_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', required=True, help='Chemin du fichier de données brutes (Excel ou CSV)')
    parser.add_argument('--model-dir', default='models/')
    parser.add_argument('--output-dir', default='data/processed')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--cross-validation', action='store_true')

    args = parser.parse_args()

    train_model(
        input_file=args.data_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        optimize=args.optimize,
        cross_validation=args.cross_validation
    )


if __name__ == '__main__':
    main()