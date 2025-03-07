# src/scripts/generate_predictions.py

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt

# Ajout correct et robuste de la racine du projet au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import des modules locaux
from src.hospital_prediction.predict import (
    load_prediction_model,
    generate_predictions,
    visualize_predictions
)
from src.hospital_prediction.data_processor import (
    load_data_from_csv,
    load_data_from_excel,
    load_latest_processed_data
)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.generate_predictions')


def prepare_input_data(data_path: str) -> pd.DataFrame:
    """
    Prépare les données d'entrée pour la génération de prédictions
    
    Args:
        data_path: Chemin vers le fichier de données
    
    Returns:
        DataFrame préparé pour les prédictions
    """
    logger.info(f"Préparation des données à partir de {data_path}")
    
    # Charger les données en fonction de l'extension
    if data_path.endswith('.csv'):
        historical_data = load_data_from_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        historical_data = load_data_from_excel(data_path)
    else:
        # Essayer de charger les dernières données prétraitées
        try:
            historical_data = load_latest_processed_data(os.path.dirname(data_path))
        except Exception as e:
            logger.error(f"Impossible de charger les données: {str(e)}")
            raise ValueError(f"Format de fichier non supporté ou données introuvables: {data_path}")
    
    # Vérifier si la colonne 'date' est présente
    if 'date' not in historical_data.columns:
        logger.warning("Colonne 'date' absente. Génération d'une colonne 'date' par défaut.")
        historical_data['date'] = pd.date_range(end=datetime.now(), periods=len(historical_data), freq='D')
    
    # Convertir la colonne de date si nécessaire
    if 'date' in historical_data.columns:
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        # Ajouter explicitement les colonnes manquantes dayOfMonth et year
        if 'dayOfMonth' not in historical_data.columns:
            historical_data['dayOfMonth'] = historical_data['date'].dt.day
        if 'year' not in historical_data.columns:
            historical_data['year'] = historical_data['date'].dt.year
    
    return historical_data


def run_predictions(
    data_path: str, 
    model_path: str, 
    output_dir: str = 'predictions', 
    prediction_days: int = 30
) -> dict:
    """
    Génère des prédictions à partir de données historiques et d'un modèle
    
    Args:
        data_path: Chemin vers le fichier de données historiques
        model_path: Chemin vers le modèle entraîné
        output_dir: Répertoire de sortie pour les prédictions
        prediction_days: Nombre de jours à prédire
    
    Returns:
        Dictionnaire contenant les résultats des prédictions
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Préparer les données d'entrée
    historical_data = prepare_input_data(data_path)
    
    # Charger le modèle
    model = load_prediction_model(model_path)
    
    # Générer les prédictions
    future_predictions, alerts = generate_predictions(
        model, 
        historical_data, 
        prediction_days, 
        output_dir
    )
    
    # Visualiser les prédictions
    fig = visualize_predictions(
        model, 
        historical_data, 
        future_predictions, 
        output_dir
    )
    
    # Fermer la figure pour libérer la mémoire
    plt.close(fig)
    
    # Préparer les résultats
    prediction_results = {
        'timestamp': datetime.now().isoformat(),
        'input_data': data_path,
        'model_path': model_path,
        'prediction_days': prediction_days,
        'output_dir': output_dir,
        'alerts': alerts,
        'summary': {
            'mean_admissions': future_predictions['predicted_admissions'].mean(),
            'max_admissions': future_predictions['predicted_admissions'].max(),
            'mean_occupancy': future_predictions['predicted_occupancy'].mean(),
            'max_occupancy': future_predictions['predicted_occupancy'].max(),
            'admission_alerts': sum(future_predictions['admission_alert']),
            'occupancy_alerts': sum(future_predictions['occupancy_alert'])
        }
    }
    
    # Sauvegarder les résultats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(output_dir, f'prediction_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(prediction_results, f, indent=4)
    
    logger.info(f"Résultats des prédictions sauvegardés dans {results_file}")
    
    return prediction_results


def main():
    """
    Point d'entrée principal pour la génération de prédictions
    """
    # Configuration de l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Génération de prédictions pour l'hôpital")
    
    # Arguments requis
    parser.add_argument(
        "--data-file", 
        type=str, 
        required=True, 
        help="Chemin vers le fichier de données historiques (CSV ou Excel)"
    )
    parser.add_argument(
        "--model-dir", 
        type=str, 
        required=True, 
        help="Répertoire contenant les modèles entraînés"
    )
    
    # Arguments optionnels
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="predictions", 
        help="Répertoire de sortie pour les prédictions (défaut: predictions)"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="Nombre de jours à prédire (défaut: 30)"
    )
    
    # Parsing des arguments
    args = parser.parse_args()
    
    try:
        # Lancer la génération de prédictions
        results = run_predictions(
            data_path=args.data_file,
            model_path=args.model_dir,
            output_dir=args.output_dir,
            prediction_days=args.days
        )
        
        # Afficher un résumé des prédictions
        print("\nRésumé des prédictions:")
        print(f"  - Période de prédiction: {args.days} jours")
        print(f"  - Données utilisées: {args.data_file}")
        print(f"  - Répertoire de sortie: {args.output_dir}")
        
        print("\nStatistiques des prédictions:")
        summary = results['summary']
        print(f"  - Admissions moyennes prévues: {summary['mean_admissions']:.2f}")
        print(f"  - Admissions maximales prévues: {summary['max_admissions']:.2f}")
        print(f"  - Taux d'occupation moyen prévu: {summary['mean_occupancy']:.2f}%")
        print(f"  - Taux d'occupation maximum prévu: {summary['max_occupancy']:.2f}%")
        
        print("\nAlertes:")
        print(f"  - Nombre de pics d'admissions prévus: {summary['admission_alerts']}")
        print(f"  - Nombre de jours de saturation prévus: {summary['occupancy_alerts']}")
        
        # Afficher les détails des alertes
        if results['alerts']:
            print("\nDétails des alertes:")
            for alert in results['alerts']:
                print(f"  - {alert['date']}: {alert['message']}")
        
        print("\nPrédictions générées avec succès!")
    
    except Exception as e:
        logger.error(f"Erreur lors de la génération des prédictions : {str(e)}")
        print(f"Erreur : {str(e)}")


# Point d'entrée du script
if __name__ == "__main__":
    main()
