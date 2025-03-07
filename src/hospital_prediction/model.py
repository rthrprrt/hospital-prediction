# src/hospital_prediction/model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
import joblib
import os
import json
from datetime import datetime, timedelta
import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.model')


class HospitalPredictionModel:
    """
    Modèle de prédiction pour l'hôpital de la Pitié-Salpêtrière permettant de :
    1. Prédire le nombre d'admissions quotidiennes
    2. Prédire le taux d'occupation des lits
    3. Détecter les pics d'activité potentiels
    """
    
    def __init__(self, model_dir='models'):
        """
        Initialiser le modèle avec le répertoire de stockage des modèles entraînés
        
        Args:
            model_dir: Répertoire pour stocker les modèles entraînés
        """
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Répertoire de modèles créé: {model_dir}")
            
        # Modèles pour les admissions et le taux d'occupation
        self.admission_model = None
        self.occupancy_model = None
        self.feature_names = None
        self.feature_importance_admission = None
        self.feature_importance_occupancy = None
        
        # Paramètres pour la détection des pics d'activité
        self.admission_threshold = None  # Seuil pour considérer un pic d'admissions
        self.occupancy_threshold = None  # Seuil pour considérer un taux d'occupation critique
        
        # Historique des performances
        self.metrics = {
            'admissions': {},
            'occupancy': {}
        }
        
        logger.info("Modèle de prédiction initialisé")
    
    def preprocess_data(self, data_path, date_column='date'):
        """
        Prétraiter les données à partir d'un CSV ou DataFrame
        
        Args:
            data_path: Chemin vers le fichier CSV ou DataFrame contenant les données
            date_column: Nom de la colonne contenant les dates
        
        Returns:
            DataFrame prétraité
        """
        logger.info(f"Prétraitement des données à partir de: {data_path}")
        
        # Charger les données
        if isinstance(data_path, str) and data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            # Convertir la colonne de date en datetime
            df[date_column] = pd.to_datetime(df[date_column])
        else:
            # Si data_path est un DataFrame ou autre structure
            df = pd.DataFrame(data_path)
            if date_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
        
        # Trier par date
        df = df.sort_values(by=date_column)
        
        # S'assurer que les features catégorielles sont bien encodées
        if 'month' in df.columns:
            df['month'] = df['month'].astype(int)
        if 'dayOfWeek' in df.columns:
            df['dayOfWeek'] = df['dayOfWeek'].astype(int)
        
        # Vérifier et traiter les valeurs manquantes
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                logger.info(f"Traitement de {missing_count} valeurs manquantes dans la colonne '{col}'")
                if df[col].dtype in [np.float64, np.int64]:
                    # Imputer les valeurs numériques par la médiane
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Imputer les valeurs catégorielles par le mode
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        logger.info(f"Prétraitement terminé: {len(df)} enregistrements")
        return df
    
    def prepare_features_targets(self, df, target_col_admissions, target_col_occupancy):
        """
        Préparer les features et les cibles pour l'entraînement
        
        Args:
            df: DataFrame prétraité
            target_col_admissions: Nom de la colonne cible pour les admissions
            target_col_occupancy: Nom de la colonne cible pour le taux d'occupation
        
        Returns:
            X: Features, y_admissions: Cible admissions, y_occupancy: Cible taux d'occupation
        """
        logger.info("Préparation des features et des cibles pour l'entraînement")
        
        # Sélectionner les colonnes de features (toutes sauf les cibles et la date)
        feature_cols = [col for col in df.columns if col not in 
                       [target_col_admissions, target_col_occupancy, 'date']]
        
        # Stocker les noms des features
        self.feature_names = feature_cols
        
        # Extraire les features et les cibles
        X = df[feature_cols]
        y_admissions = df[target_col_admissions]
        y_occupancy = df[target_col_occupancy]
        
        logger.info(f"Features sélectionnées: {', '.join(feature_cols)}")
        return X, y_admissions, y_occupancy
    
    def train(self, X, y_admissions, y_occupancy, cv=5, n_jobs=-1):
        """
        Entraîner les modèles de prédiction
        
        Args:
            X: Features d'entraînement
            y_admissions: Cible pour les admissions
            y_occupancy: Cible pour le taux d'occupation
            cv: Nombre de plis pour la validation croisée
            n_jobs: Nombre de jobs parallèles (-1 pour utiliser tous les cœurs)
        """
        logger.info("Entraînement des modèles de prédiction...")
        
        # Configuration de la validation croisée adaptée aux séries temporelles
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Modèle pour les admissions (Gradient Boosting)
        logger.info("Entraînement du modèle d'admissions (Gradient Boosting)")
        gb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ])
        
        gb_pipeline.fit(X, y_admissions)
        self.admission_model = gb_pipeline
        
        # Extraire l'importance des features pour le modèle d'admissions
        gb_importances = self.admission_model.named_steps['gb'].feature_importances_
        self.feature_importance_admission = dict(zip(self.feature_names, gb_importances))
        
        # Calculer les métriques pour le modèle d'admissions
        y_pred_admissions = self.admission_model.predict(X)
        self.metrics['admissions']['mse'] = mean_squared_error(y_admissions, y_pred_admissions)
        self.metrics['admissions']['mae'] = mean_absolute_error(y_admissions, y_pred_admissions)
        self.metrics['admissions']['r2'] = r2_score(y_admissions, y_pred_admissions)
        
        logger.info(f"Modèle d'admissions entraîné. MSE: {self.metrics['admissions']['mse']:.4f}, "
              f"MAE: {self.metrics['admissions']['mae']:.4f}, R²: {self.metrics['admissions']['r2']:.4f}")
        
        # Calculer un seuil pour les pics d'admissions (75e percentile + 1.5 * IQR)
        q75 = np.percentile(y_admissions, 75)
        q25 = np.percentile(y_admissions, 25)
        iqr = q75 - q25
        self.admission_threshold = q75 + 1.5 * iqr
        logger.info(f"Seuil de détection des pics d'admissions fixé à: {self.admission_threshold:.2f}")
        
        # Modèle pour le taux d'occupation (Random Forest)
        logger.info("Entraînement du modèle de taux d'occupation (Random Forest)")
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=n_jobs
            ))
        ])
        
        rf_pipeline.fit(X, y_occupancy)
        self.occupancy_model = rf_pipeline
        
        # Extraire l'importance des features pour le modèle de taux d'occupation
        rf_importances = self.occupancy_model.named_steps['rf'].feature_importances_
        self.feature_importance_occupancy = dict(zip(self.feature_names, rf_importances))
        
        # Calculer les métriques pour le modèle de taux d'occupation
        y_pred_occupancy = self.occupancy_model.predict(X)
        self.metrics['occupancy']['mse'] = mean_squared_error(y_occupancy, y_pred_occupancy)
        self.metrics['occupancy']['mae'] = mean_absolute_error(y_occupancy, y_pred_occupancy)
        self.metrics['occupancy']['r2'] = r2_score(y_occupancy, y_pred_occupancy)
        
        logger.info(f"Modèle de taux d'occupation entraîné. MSE: {self.metrics['occupancy']['mse']:.4f}, "
              f"MAE: {self.metrics['occupancy']['mae']:.4f}, R²: {self.metrics['occupancy']['r2']:.4f}")
        
        # Calculer un seuil pour le taux d'occupation critique (90%)
        self.occupancy_threshold = 90.0
        logger.info(f"Seuil critique pour le taux d'occupation fixé à: {self.occupancy_threshold:.2f}%")
    
    def evaluate(self, X_test, y_test_admissions, y_test_occupancy):
        """
        Évaluer les modèles sur un ensemble de test
        
        Args:
            X_test: Features de test
            y_test_admissions: Cible de test pour les admissions
            y_test_occupancy: Cible de test pour le taux d'occupation
            
        Returns:
            Dict contenant les métriques d'évaluation
        """
        logger.info("Évaluation des modèles sur l'ensemble de test...")
        
        # Prédictions sur l'ensemble de test
        y_pred_admissions = self.admission_model.predict(X_test)
        y_pred_occupancy = self.occupancy_model.predict(X_test)
        
        # Calculer les métriques pour les admissions
        test_metrics = {
            'admissions': {
                'mse': mean_squared_error(y_test_admissions, y_pred_admissions),
                'mae': mean_absolute_error(y_test_admissions, y_pred_admissions),
                'r2': r2_score(y_test_admissions, y_pred_admissions)
            },
            'occupancy': {
                'mse': mean_squared_error(y_test_occupancy, y_pred_occupancy),
                'mae': mean_absolute_error(y_test_occupancy, y_pred_occupancy),
                'r2': r2_score(y_test_occupancy, y_pred_occupancy)
            }
        }
        
        logger.info(f"Évaluation du modèle d'admissions. MSE: {test_metrics['admissions']['mse']:.4f}, "
              f"MAE: {test_metrics['admissions']['mae']:.4f}, R²: {test_metrics['admissions']['r2']:.4f}")
        
        logger.info(f"Évaluation du modèle de taux d'occupation. MSE: {test_metrics['occupancy']['mse']:.4f}, "
              f"MAE: {test_metrics['occupancy']['mae']:.4f}, R²: {test_metrics['occupancy']['r2']:.4f}")
        
        return test_metrics
    
    def predict(self, X):
        """
        Faire des prédictions avec les modèles entraînés
        
        Args:
            X: Features pour la prédiction
            
        Returns:
            Tuple (prédictions des admissions, prédictions du taux d'occupation, alertes)
        """
        if self.admission_model is None or self.occupancy_model is None:
            raise ValueError("Les modèles doivent être entraînés avant de faire des prédictions")
        
        logger.info(f"Prédiction sur {len(X)} échantillons")
        
        # Prédictions
        y_pred_admissions = self.admission_model.predict(X)
        y_pred_occupancy = self.occupancy_model.predict(X)
        
        # Générer des alertes pour les pics potentiels
        alerts = []
        for i in range(len(X)):
            if y_pred_admissions[i] >= self.admission_threshold:
                alerts.append({
                    'index': i,
                    'type': 'admissions',
                    'predicted': y_pred_admissions[i],
                    'threshold': self.admission_threshold,
                    'message': f"⚠️ Pic d'admissions prévu: {y_pred_admissions[i]:.1f} admissions"
                })
            
            if y_pred_occupancy[i] >= self.occupancy_threshold:
                alerts.append({
                    'index': i,
                    'type': 'occupancy',
                    'predicted': y_pred_occupancy[i],
                    'threshold': self.occupancy_threshold,
                    'message': f"⚠️ Taux d'occupation critique prévu: {y_pred_occupancy[i]:.1f}%"
                })
        
        if alerts:
            logger.info(f"{len(alerts)} alertes générées")
        
        return y_pred_admissions, y_pred_occupancy, alerts
    
    def predict_future(self, last_data, days=30):
        """
        Prédire les admissions et le taux d'occupation pour les jours futurs
        
        Args:
            last_data: Dernières données connues (DataFrame)
            days: Nombre de jours à prédire dans le futur
            
        Returns:
            DataFrame contenant les prédictions pour les jours futurs
        """
        if self.admission_model is None or self.occupancy_model is None:
            raise ValueError("Les modèles doivent être entraînés avant de faire des prédictions")
            
        logger.info(f"Prédiction pour les {days} prochains jours")
        
        # Assurer que last_data est un DataFrame
        last_data = pd.DataFrame(last_data) if not isinstance(last_data, pd.DataFrame) else last_data.copy()
        
        # S'assurer que toutes les colonnes nécessaires sont présentes
        if 'date' in last_data.columns:
            if 'dayOfMonth' not in last_data.columns:
                last_data['dayOfMonth'] = last_data['date'].dt.day
            if 'year' not in last_data.columns:
                last_data['year'] = last_data['date'].dt.year
        
        # Vérifier si toutes les colonnes du modèle sont présentes
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in last_data.columns:
                    logger.warning(f"Colonne '{feature}' manquante, ajout d'une valeur par défaut")
                    # Ajouter une valeur par défaut selon le type de feature
                    if feature in ['month', 'dayOfMonth', 'dayOfWeek', 'year']:
                        last_data[feature] = 1
                    elif feature.startswith('is'):
                        last_data[feature] = 0
                    else:
                        last_data[feature] = last_data.get('totalAdmissions', 80) if 'admissions' in feature else last_data.get('avgOccupancyRate', 75)
        
        # Trier par date et prendre les dernières entrées nécessaires
        if 'date' in last_data.columns:
            last_data = last_data.sort_values('date')
        
        # Préparer le DataFrame pour les prédictions futures
        future_data = []
        
        # Dernière date connue
        last_date = pd.to_datetime(last_data['date'].iloc[-1]) if 'date' in last_data.columns else datetime.now()
        
        # Pour chaque jour futur
        for i in range(1, days + 1):
            future_date = last_date + timedelta(days=i)
            
            # Créer une nouvelle entrée avec les caractéristiques de base
            new_entry = {
                'date': future_date,
                'year': future_date.year,
                'month': future_date.month,
                'dayOfMonth': future_date.day,
                'dayOfWeek': future_date.weekday(),
                'isWeekend': 1 if future_date.weekday() >= 5 else 0,
                'isSummer': 1 if future_date.month in [6, 7, 8] else 0,
                'isWinter': 1 if future_date.month in [12, 1, 2] else 0,
                'isSpring': 1 if future_date.month in [3, 4, 5] else 0,
                'isFall': 1 if future_date.month in [9, 10, 11] else 0,
            }
            
            # Si c'est le premier jour de prédiction, utiliser les valeurs des dernières données connues
            if i == 1:
                # Occupancy lags (utiliser les dernières valeurs connues ou prédites)
                new_entry['occupancyLag1'] = float(last_data['avgOccupancyRate'].iloc[-1])
                new_entry['occupancyLag3'] = float(last_data['avgOccupancyRate'].iloc[-3]) if len(last_data) >= 3 else new_entry['occupancyLag1']
                new_entry['occupancyLag7'] = float(last_data['avgOccupancyRate'].iloc[-7]) if len(last_data) >= 7 else new_entry['occupancyLag1']
                
                # Admission lags
                new_entry['admissionsLag1'] = float(last_data['totalAdmissions'].iloc[-1])
                new_entry['admissionsLag3'] = float(last_data['totalAdmissions'].iloc[-3]) if len(last_data) >= 3 else new_entry['admissionsLag1']
                new_entry['admissionsLag7'] = float(last_data['totalAdmissions'].iloc[-7]) if len(last_data) >= 7 else new_entry['admissionsLag1']
                
                # Moyennes mobiles
                window_size = min(7, len(last_data))
                new_entry['admissionsMA7'] = float(last_data['totalAdmissions'].iloc[-window_size:].mean())
                new_entry['occupancyMA7'] = float(last_data['avgOccupancyRate'].iloc[-window_size:].mean())
            else:
                # Utiliser les prédictions précédentes pour les lags
                prev_entries = future_data[-min(i-1, 7):]
                prev_df = pd.DataFrame(prev_entries)
                
                # Occupancy lags
                new_entry['occupancyLag1'] = prev_df['predicted_occupancy'].iloc[-1] if len(prev_df) >= 1 else new_entry.get('occupancyLag1', 75.0)
                new_entry['occupancyLag3'] = prev_df['predicted_occupancy'].iloc[-3] if len(prev_df) >= 3 else new_entry.get('occupancyLag1', 75.0)
                new_entry['occupancyLag7'] = prev_df['predicted_occupancy'].iloc[-7] if len(prev_df) >= 7 else new_entry.get('occupancyLag1', 75.0)
                
                # Admission lags
                new_entry['admissionsLag1'] = prev_df['predicted_admissions'].iloc[-1] if len(prev_df) >= 1 else new_entry.get('admissionsLag1', 1.0)
                new_entry['admissionsLag3'] = prev_df['predicted_admissions'].iloc[-3] if len(prev_df) >= 3 else new_entry.get('admissionsLag1', 1.0)
                new_entry['admissionsLag7'] = prev_df['predicted_admissions'].iloc[-7] if len(prev_df) >= 7 else new_entry.get('admissionsLag1', 1.0)
                
                # Moyennes mobiles (combiner les données réelles et prédites si nécessaire)
                admissions_values = list(prev_df['predicted_admissions'])
                occupancy_values = list(prev_df['predicted_occupancy'])
                
                if len(admissions_values) < 7 and 'totalAdmissions' in last_data.columns:
                    # Compléter avec des données réelles si disponibles
                    additional_values = last_data['totalAdmissions'].iloc[-(7-len(admissions_values)):].tolist()
                    admissions_values = additional_values + admissions_values
                
                if len(occupancy_values) < 7 and 'avgOccupancyRate' in last_data.columns:
                    # Compléter avec des données réelles si disponibles
                    additional_values = last_data['avgOccupancyRate'].iloc[-(7-len(occupancy_values)):].tolist()
                    occupancy_values = additional_values + occupancy_values
                
                # Calculer les moyennes mobiles
                new_entry['admissionsMA7'] = sum(admissions_values[-min(7, len(admissions_values)):]) / min(7, len(admissions_values))
                new_entry['occupancyMA7'] = sum(occupancy_values[-min(7, len(occupancy_values)):]) / min(7, len(occupancy_values))
            
            # Extraire les features pertinentes pour la prédiction
            # Créer un dict avec toutes les features requises
            prediction_features = {}
            
            # Ajouter les features de base avec des valeurs par défaut
            for feature in self.feature_names:
                if feature in new_entry:
                    prediction_features[feature] = new_entry[feature]
                else:
                    # Valeur par défaut selon le type de feature
                    if feature in ['month', 'dayOfMonth', 'dayOfWeek', 'year']:
                        prediction_features[feature] = 1
                    elif feature.startswith('is'):
                        prediction_features[feature] = 0
                    else:
                        prediction_features[feature] = 75 if 'occupancy' in feature else 80
            
            X_future = pd.DataFrame([prediction_features])
            
            # Faire les prédictions
            predicted_admissions = float(self.admission_model.predict(X_future)[0])
            predicted_occupancy = float(self.occupancy_model.predict(X_future)[0])
            
            # Ajouter les prédictions à l'entrée
            new_entry['predicted_admissions'] = predicted_admissions
            new_entry['predicted_occupancy'] = predicted_occupancy
            
            # Vérifier si une alerte doit être générée
            new_entry['admission_alert'] = predicted_admissions >= self.admission_threshold
            new_entry['occupancy_alert'] = predicted_occupancy >= self.occupancy_threshold
            
            # Ajouter l'entrée aux données futures
            future_data.append(new_entry)
        
        # Convertir en DataFrame
        future_df = pd.DataFrame(future_data)
        
        logger.info(f"Prédictions générées pour la période du {future_df['date'].min().strftime('%Y-%m-%d')} au {future_df['date'].max().strftime('%Y-%m-%d')}")
        return future_df
    
    def save(self, filename_prefix=None):
        """
        Sauvegarder les modèles entraînés
        
        Args:
            filename_prefix: Préfixe pour les noms de fichiers (par défaut: timestamp)
        
        Returns:
            Dict avec les chemins des fichiers sauvegardés
        """
        if filename_prefix is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename_prefix = f"hospital_model_{timestamp}"
        
        # Créer le répertoire si nécessaire
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Chemins des fichiers
        admission_model_path = os.path.join(self.model_dir, f"{filename_prefix}_admission_model.joblib")
        occupancy_model_path = os.path.join(self.model_dir, f"{filename_prefix}_occupancy_model.joblib")
        metadata_path = os.path.join(self.model_dir, f"{filename_prefix}_metadata.json")
        
        # Sauvegarder les modèles
        joblib.dump(self.admission_model, admission_model_path)
        joblib.dump(self.occupancy_model, occupancy_model_path)
        
        # Sauvegarder les métadonnées
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance_admission': self.feature_importance_admission,
            'feature_importance_occupancy': self.feature_importance_occupancy,
            'admission_threshold': self.admission_threshold,
            'occupancy_threshold': self.occupancy_threshold,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Modèles et métadonnées sauvegardés avec le préfixe '{filename_prefix}'")
        
        return {
            'admission_model': admission_model_path,
            'occupancy_model': occupancy_model_path,
            'metadata': metadata_path
        }
    
    def load(self, admission_model_path, occupancy_model_path, metadata_path=None):
        """
        Charger des modèles entraînés
        
        Args:
            admission_model_path: Chemin vers le modèle d'admissions sauvegardé
            occupancy_model_path: Chemin vers le modèle de taux d'occupation sauvegardé
            metadata_path: Chemin vers les métadonnées (optionnel)
        """
        logger.info(f"Chargement des modèles depuis {admission_model_path} et {occupancy_model_path}")
        
        # Charger les modèles
        self.admission_model = joblib.load(admission_model_path)
        self.occupancy_model = joblib.load(occupancy_model_path)
        
        # Charger les métadonnées si disponibles
        if metadata_path and os.path.exists(metadata_path):
            logger.info(f"Chargement des métadonnées depuis {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata.get('feature_names')
            self.feature_importance_admission = metadata.get('feature_importance_admission')
            self.feature_importance_occupancy = metadata.get('feature_importance_occupancy')
            self.admission_threshold = metadata.get('admission_threshold')
            self.occupancy_threshold = metadata.get('occupancy_threshold')
            self.metrics = metadata.get('metrics', {'admissions': {}, 'occupancy': {}})
        
        logger.info("Modèles chargés avec succès")
    
    def plot_feature_importance(self, model_type='both', top_n=10):
        """
        Génération du graphique d'importance des features
        
        Args:
            model_type: Type de modèle ('admissions', 'occupancy', ou 'both')
            top_n: Nombre de features à afficher
            
        Returns:
            Figure matplotlib
        """
        logger.info(f"Génération du graphique d'importance des features (type: {model_type}, top {top_n})")
        
        if model_type not in ['admissions', 'occupancy', 'both']:
            raise ValueError("model_type doit être 'admissions', 'occupancy' ou 'both'")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6)) if model_type == 'both' else plt.subplots(figsize=(10, 6))
        
        if model_type in ['admissions', 'both'] and self.feature_importance_admission:
            sorted_features_admissions = sorted(self.feature_importance_admission.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features_adm, importances_adm = zip(*sorted_features_admissions)
            ax1 = axes[0] if model_type == 'both' else axes
            ax1.barh(range(len(features_adm)), importances_adm, align='center')
            ax1.set_yticks(range(len(features_adm)))
            ax1.set_yticklabels(features_adm)
            ax1.set_title("Importance des features - Admissions")
            ax1.set_xlabel('Importance')
        
        if model_type in ['occupancy', 'both'] and self.feature_importance_occupancy:
            sorted_features_occ = sorted(self.feature_importance_occupancy.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features_occ, importances_occ = zip(*sorted_features_occ)
            ax2 = axes[1] if model_type == 'both' else axes
            ax2.barh(range(len(features_occ)), importances_occ, align='center')
            ax2.set_yticks(range(len(features_occ)))
            ax2.set_yticklabels(features_occ)
            ax2.set_title("Importance des features - Modèle d'occupation")
            ax2.set_xlabel('Importance')
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, X, y_true_admissions=None, y_true_occupancy=None, dates=None, future_days=0):
        """
        Visualiser les prédictions par rapport aux valeurs réelles
        
        Args:
            X: Features pour la prédiction
            y_true_admissions: Valeurs réelles des admissions (optionnel)
            y_true_occupancy: Valeurs réelles du taux d'occupation (optionnel)
            dates: Liste des dates correspondant aux prédictions (optionnel)
            future_days: Nombre de jours futurs inclus dans les prédictions (par défaut: 0)
            
        Returns:
            Figure matplotlib
        """
        logger.info("Génération des graphiques de prédiction")
        
        # Faire les prédictions
        y_pred_admissions, y_pred_occupancy, alerts = self.predict(X)
        
        # Créer une figure avec deux sous-graphiques
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Créer un index pour l'axe x
        x_index = range(len(y_pred_admissions))
        if dates is not None:
            if isinstance(dates[0], str):
                dates = [pd.to_datetime(d) for d in dates]
            x_index = dates
        
        # Graphique des admissions
        ax1.plot(x_index, y_pred_admissions, 'b-', label='Prédictions')
        
        if y_true_admissions is not None:
            if future_days > 0:
                # Les données historiques n'incluent pas les jours futurs
                ax1.plot(x_index[:-future_days], y_true_admissions, 'g-', label='Valeurs réelles')
            else:
                ax1.plot(x_index, y_true_admissions, 'g-', label='Valeurs réelles')
        
        # Ajouter une ligne pour le seuil d'alerte
        if self.admission_threshold is not None:
            ax1.axhline(y=self.admission_threshold, color='r', linestyle='--', label=f'Seuil d\'alerte ({self.admission_threshold:.1f})')
        
        # Marquer les alertes
        if alerts:
            alert_indices = [alert['index'] for alert in alerts if alert['type'] == 'admissions']
            if alert_indices and future_days > 0:
                # Mettre en évidence les alertes futures
                future_alerts = [idx for idx in alert_indices if idx >= len(x_index) - future_days]
                if future_alerts:
                    future_x = [x_index[idx] for idx in future_alerts]
                    future_y = [y_pred_admissions[idx] for idx in future_alerts]
                    ax1.scatter(future_x, future_y, color='red', s=50, label='Alertes futures')
        
        ax1.set_title('Prédiction du nombre d\'admissions quotidiennes')
        ax1.set_ylabel('Nombre d\'admissions')
        ax1.legend()
        ax1.grid(True)
        
        # Si nous avons des dates, améliorer le formatage de l'axe x
        if dates is not None:
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Graphique du taux d'occupation
        ax2.plot(x_index, y_pred_occupancy, 'b-', label='Prédictions')
        
        if y_true_occupancy is not None:
            if future_days > 0:
                # Les données historiques n'incluent pas les jours futurs
                ax2.plot(x_index[:-future_days], y_true_occupancy, 'g-', label='Valeurs réelles')
            else:
                ax2.plot(x_index, y_true_occupancy, 'g-', label='Valeurs réelles')
        
        # Ajouter une ligne pour le seuil d'alerte
        if self.occupancy_threshold is not None:
            ax2.axhline(y=self.occupancy_threshold, color='r', linestyle='--', label=f'Seuil critique ({self.occupancy_threshold:.1f}%)')
        
        # Marquer les alertes
        if alerts:
            alert_indices = [alert['index'] for alert in alerts if alert['type'] == 'occupancy']
            if alert_indices and future_days > 0:
                # Mettre en évidence les alertes futures
                future_alerts = [idx for idx in alert_indices if idx >= len(x_index) - future_days]
                if future_alerts:
                    future_x = [x_index[idx] for idx in future_alerts]
                    future_y = [y_pred_occupancy[idx] for idx in future_alerts]
                    ax2.scatter(future_x, future_y, color='red', s=50, label='Alertes futures')
        
        ax2.set_title('Prédiction du taux d\'occupation des lits')
        ax2.set_ylabel('Taux d\'occupation (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Si nous avons des dates, améliorer le formatage de l'axe x
        if dates is not None:
            ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        logger.info("Graphiques de prédiction générés")
        
        return fig
    
    def get_seasonal_patterns(self, df, target_col_admissions, target_col_occupancy):
        """
        Analyser les motifs saisonniers dans les données
        
        Args:
            df: DataFrame contenant les données
            target_col_admissions: Nom de la colonne des admissions
            target_col_occupancy: Nom de la colonne du taux d'occupation
            
        Returns:
            Dict contenant les analyses de saisonnalité
        """
        logger.info("Analyse des motifs saisonniers dans les données")
        
        # Vérifier que df est un DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        
        # S'assurer que les colonnes nécessaires existent
        required_cols = ['month', 'dayOfWeek', target_col_admissions, target_col_occupancy]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans le DataFrame: {missing_cols}")
        
        # Analyse par mois
        monthly_patterns = df.groupby('month').agg({
            target_col_admissions: ['mean', 'std', 'max'],
            target_col_occupancy: ['mean', 'std', 'max']
        })
        
        # Analyse par jour de la semaine
        weekly_patterns = df.groupby('dayOfWeek').agg({
            target_col_admissions: ['mean', 'std', 'max'],
            target_col_occupancy: ['mean', 'std', 'max']
        })
        
        # Analyse par saison
        season_mapping = {
            'Hiver': [12, 1, 2],
            'Printemps': [3, 4, 5],
            'Été': [6, 7, 8],
            'Automne': [9, 10, 11]
        }
        
        df['season'] = df['month'].apply(
            lambda m: next(season for season, months in season_mapping.items() if m in months)
        )
        
        seasonal_patterns = df.groupby('season').agg({
            target_col_admissions: ['mean', 'std', 'max', 'count'],
            target_col_occupancy: ['mean', 'std', 'max']
        })
        
        # Analyse weekends vs jours de semaine
        df['is_weekend'] = df['dayOfWeek'].apply(lambda d: d >= 5)
        weekend_patterns = df.groupby('is_weekend').agg({
            target_col_admissions: ['mean', 'std', 'max', 'count'],
            target_col_occupancy: ['mean', 'std', 'max']
        })
        
        return {
            'monthly': monthly_patterns,
            'weekly': weekly_patterns,
            'seasonal': seasonal_patterns,
            'weekend': weekend_patterns
        }


# Fonctions utilitaires pour faciliter l'utilisation de la classe

def train_hospital_model(data_path, test_size=0.2, random_state=42):
    """
    Fonction utilitaire pour entraîner le modèle hospitalier à partir de données CSV
    
    Args:
        data_path: Chemin vers le fichier CSV ou DataFrame contenant les données
        test_size: Proportion des données à utiliser pour le test (par défaut: 0.2)
        random_state: Graine aléatoire pour la reproductibilité (par défaut: 42)
        
    Returns:
        Tuple (modèle entraîné, métriques d'évaluation, figure d'importance des features, chemins des modèles)
    """
    logger.info(f"Entraînement du modèle hospitalier avec les données de {data_path}")
    
    # Créer une instance du modèle
    model = HospitalPredictionModel()
    
    # Prétraiter les données
    df = model.preprocess_data(data_path)
    
    # Préparer les features et les cibles
    X, y_admissions, y_occupancy = model.prepare_features_targets(
        df, 'totalAdmissions', 'avgOccupancyRate'
    )
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train_admissions, y_test_admissions, y_train_occupancy, y_test_occupancy = train_test_split(
        X, y_admissions, y_occupancy, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Division des données: {len(X_train)} exemples d'entraînement, {len(X_test)} exemples de test")
    
    # Entraîner les modèles
    model.train(X_train, y_train_admissions, y_train_occupancy)
    
    # Évaluer les modèles
    test_metrics = model.evaluate(X_test, y_test_admissions, y_test_occupancy)
    
    # Visualiser l'importance des features
    fig_importance = model.plot_feature_importance()
    
    # Sauvegarder les modèles
    model_paths = model.save()
    
    return model, test_metrics, fig_importance, model_paths


def predict_future_admissions(model, last_data, days=30):
    """
    Fonction utilitaire pour prédire les admissions futures
    
    Args:
        model: Instance entraînée de HospitalPredictionModel
        last_data: Dernières données connues (DataFrame)
        days: Nombre de jours à prédire (par défaut: 30)
        
    Returns:
        DataFrame contenant les prédictions
    """
    logger.info(f"Prédiction des admissions futures sur {days} jours")
    
    # Prédire les admissions et le taux d'occupation pour les jours futurs
    future_df = model.predict_future(last_data, days)
    
    return future_df
