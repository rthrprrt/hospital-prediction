# src/hospital_prediction/predict.py

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import des modules locaux
from src.hospital_prediction.model import HospitalPredictionModel
from src.utils.visualization import plot_predictions, save_figure

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.predict')


def load_prediction_model(model_path: str, metadata_path: Optional[str] = None) -> HospitalPredictionModel:
    """
    Charge un modèle de prédiction entraîné
    
    Args:
        model_path: Chemin vers le répertoire contenant les modèles
        metadata_path: Chemin vers le fichier de métadonnées (optionnel)
    
    Returns:
        Modèle de prédiction hospitalière chargé
    """
    logger.info(f"Chargement du modèle de prédiction depuis {model_path}")
    
    # Si un répertoire est fourni, trouver le modèle le plus récent
    if os.path.isdir(model_path):
        model_files = [f for f in os.listdir(model_path) if f.endswith('_admission_model.joblib')]
        
        if not model_files:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {model_path}")
        
        # Trier par date de modification (le plus récent en premier)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_path, x)), reverse=True)
        
        # Chemins complets des fichiers
        admission_model_path = os.path.join(model_path, model_files[0])
        occupancy_model_path = admission_model_path.replace('_admission_model.joblib', '_occupancy_model.joblib')
        metadata_path = admission_model_path.replace('_admission_model.joblib', '_metadata.json')
    else:
        # Si un chemin de fichier spécifique est fourni
        admission_model_path = model_path
        occupancy_model_path = model_path.replace('_admission_model.joblib', '_occupancy_model.joblib')
    
    # Créer et charger le modèle
    model = HospitalPredictionModel()
    model.load(admission_model_path, occupancy_model_path, metadata_path)
    
    logger.info("Modèle de prédiction chargé avec succès")
    return model


def generate_predictions(
    model: HospitalPredictionModel, 
    historical_data: pd.DataFrame, 
    prediction_days: int = 30, 
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Union[str, float, bool]]]]:
    """
    Génère des prédictions pour les jours futurs
    
    Args:
        model: Modèle de prédiction hospitalière entraîné
        historical_data: DataFrame contenant les données historiques
        prediction_days: Nombre de jours à prédire
        output_dir: Répertoire pour sauvegarder les prédictions
    
    Returns:
        Tuple contenant le DataFrame de prédictions et la liste des alertes
    """
    logger.info(f"Génération de prédictions pour {prediction_days} jours")
    
    # S'assurer que la colonne 'date' est au bon format
    if 'date' in historical_data.columns:
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        if 'dayOfMonth' not in historical_data.columns:
            historical_data['dayOfMonth'] = historical_data['date'].dt.day
        if 'year' not in historical_data.columns:
            historical_data['year'] = historical_data['date'].dt.year
    # La colonne 'date' devrait être présente ; sinon, elle sera ajoutée en amont dans predict_daily_admissions.
    
    # Vérifier si la colonne 'avgOccupancyRate' existe
    if 'avgOccupancyRate' not in historical_data.columns:
        logger.warning("Colonne 'avgOccupancyRate' manquante. Ajout avec une valeur par défaut (75).")
        historical_data['avgOccupancyRate'] = 75
    
    # Vérifier si la colonne 'totalAdmissions' existe
    if 'totalAdmissions' not in historical_data.columns:
        logger.warning("Colonne 'totalAdmissions' manquante. Ajout avec une valeur par défaut (50).")
        historical_data['totalAdmissions'] = 50
    
    # Vérifier que toutes les features requises par le modèle sont présentes
    required_features = model.feature_names if model.feature_names else []
    missing_features = [feat for feat in required_features if feat not in historical_data.columns]
    
    if missing_features:
        logger.warning(f"Colonnes manquantes dans les données: {missing_features}")
        # Ajouter les colonnes manquantes avec des valeurs par défaut
        for feat in missing_features:
            historical_data[feat] = 0
            logger.warning(f"Ajout de la colonne '{feat}' avec des valeurs par défaut (0)")
    
    # Faire les prédictions futures
    future_predictions = model.predict_future(historical_data, days=prediction_days)
    
    # Générer les alertes
    alerts = []
    for idx, row in future_predictions.iterrows():
        # Alertes pour les pics d'admissions
        if row.get('admission_alert', False):
            alerts.append({
                'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
                'type': 'admissions',
                'predicted_value': row['predicted_admissions'],
                'threshold': model.admission_threshold,
                'message': f"⚠️ Pic d'admissions prévu: {row['predicted_admissions']:.1f} admissions"
            })
        
        # Alertes pour le taux d'occupation
        if row.get('occupancy_alert', False):
            alerts.append({
                'date': row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date'],
                'type': 'occupancy',
                'predicted_value': row['predicted_occupancy'],
                'threshold': model.occupancy_threshold,
                'message': f"⚠️ Taux d'occupation critique prévu: {row['predicted_occupancy']:.1f}%"
            })
    
    # Sauvegarder les prédictions si un répertoire est spécifié
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predictions_file = os.path.join(output_dir, f'predictions_{timestamp}.csv')
        future_predictions.to_csv(predictions_file, index=False)
        logger.info(f"Prédictions sauvegardées dans {predictions_file}")
        
        import json
        alerts_file = os.path.join(output_dir, f'alerts_{timestamp}.json')
        with open(alerts_file, 'w') as f:
            json.dump(alerts, f, indent=4)
        logger.info(f"Alertes sauvegardées dans {alerts_file}")
    
    return future_predictions, alerts


def visualize_predictions(
    model: HospitalPredictionModel, 
    historical_data: pd.DataFrame, 
    future_predictions: pd.DataFrame, 
    output_dir: Optional[str] = None
) -> plt.Figure:
    """
    Visualise les prédictions historiques et futures
    
    Args:
        model: Modèle de prédiction hospitalière
        historical_data: DataFrame contenant les données historiques
        future_predictions: DataFrame contenant les prédictions futures
        output_dir: Répertoire pour sauvegarder la visualisation
    
    Returns:
        Figure matplotlib avec les visualisations de prédiction
    """
    logger.info("Préparation de la visualisation des prédictions")
    
    # Extraire les features pour la visualisation
    feature_cols = model.feature_names
    
    # Combiner les données historiques et futures pour la prédiction
    historical_features = historical_data[feature_cols].tail(50)
    future_features = future_predictions[feature_cols]
    
    combined_features = pd.concat([historical_features, future_features]).reset_index(drop=True)
    
    # Générer les dates
    historical_dates = pd.to_datetime(historical_data['date']).tail(50)
    future_dates = pd.to_datetime(future_predictions['date'])
    all_dates = list(historical_dates) + list(future_dates)
    
    # Faire les prédictions sur les données combinées
    y_pred_admissions, y_pred_occupancy, _ = model.predict(combined_features)
    
    # Créer la figure de visualisation
    fig = model.plot_predictions(
        X=combined_features, 
        y_true_admissions=historical_data['totalAdmissions'].tail(50) if 'totalAdmissions' in historical_data.columns else None,
        y_true_occupancy=historical_data['avgOccupancyRate'].tail(50) if 'avgOccupancyRate' in historical_data.columns else None,
        dates=all_dates,
        future_days=len(future_predictions)
    )
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = os.path.join(output_dir, f'predictions_plot_{timestamp}.png')
        fig.savefig(plot_file)
        logger.info(f"Visualisation des prédictions sauvegardée dans {plot_file}")
    
    return fig


def predict_daily_admissions(
    data_path: str, 
    model_path: str, 
    prediction_days: int = 30, 
    output_dir: Optional[str] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Union[str, float, bool]]]]:
    """
    Point d'entrée principal pour la génération de prédictions
    
    Args:
        data_path: Chemin vers le fichier de données historiques
        model_path: Chemin vers le modèle entraîné
        prediction_days: Nombre de jours à prédire
        output_dir: Répertoire pour sauvegarder les résultats
    
    Returns:
        Tuple contenant le DataFrame de prédictions et la liste des alertes
    """
    logger.info(f"Démarrage de la prédiction avec {prediction_days} jours de prévision")
    
    # Charger les données historiques
    if data_path.endswith('.csv'):
        historical_data = pd.read_csv(data_path)
    elif data_path.endswith(('.xls', '.xlsx')):
        historical_data = pd.read_excel(data_path)
    else:
        raise ValueError(f"Format de fichier non supporté: {data_path}")
    
    # Si la colonne 'date' est absente, la générer par défaut
    if 'date' in historical_data.columns:
        historical_data['date'] = pd.to_datetime(historical_data['date'])
    else:
        logger.warning("Colonne 'date' manquante dans les données historiques. Génération d'une colonne 'date' par défaut.")
        historical_data['date'] = pd.date_range(end=datetime.now(), periods=len(historical_data), freq='D')
    
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
    visualize_predictions(
        model, 
        historical_data, 
        future_predictions, 
        output_dir
    )
    
    logger.info("Processus de prédiction terminé avec succès")
    return future_predictions, alerts


# Exemple d'utilisation
if __name__ == "__main__":
    try:
        # Chemins à personnaliser selon votre configuration
        data_path = "data/processed/hospital_data.csv"
        model_path = "models/"
        output_dir = "predictions/"
        
        # Générer des prédictions
        predictions, alerts = predict_daily_admissions(
            data_path=data_path, 
            model_path=model_path, 
            prediction_days=30, 
            output_dir=output_dir
        )
        
        # Afficher un résumé des alertes
        print("\nRésumé des alertes:")
        for alert in alerts:
            print(f"{alert['date']} - {alert['message']}")
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
