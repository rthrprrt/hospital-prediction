# src/hospital_prediction/data_processor.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Union, List, Dict, Optional, Tuple

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.data_processor')

def load_data_from_excel(file_path: str) -> pd.DataFrame:
    """
    Charge les données depuis un fichier Excel et les prépare pour l'analyse
    
    Args:
        file_path: Chemin vers le fichier Excel
        
    Returns:
        DataFrame prétraité
    """
    logger.info(f"Chargement des données depuis {file_path}...")
    
    # Charger les données
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Données chargées avec succès: {len(df)} enregistrements")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier Excel: {str(e)}")
        raise
    
    # Convertir les colonnes de date en datetime
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'arrivée' in col.lower() or 'départ' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    logger.info(f"Colonnes de date converties: {', '.join(date_columns)}")
    
    return df

def load_data_from_csv(file_path: str, date_column: str = 'date') -> pd.DataFrame:
    """
    Charge les données depuis un fichier CSV et les prépare pour l'analyse
    
    Args:
        file_path: Chemin vers le fichier CSV
        date_column: Nom de la colonne contenant les dates
        
    Returns:
        DataFrame prétraité
    """
    logger.info(f"Chargement des données depuis {file_path}...")
    
    # Charger les données
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Données chargées avec succès: {len(df)} enregistrements")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier CSV: {str(e)}")
        raise
    
    # Convertir la colonne de date en datetime
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        logger.info(f"Colonne de date '{date_column}' convertie")
    
    return df

def preprocess_hospital_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite les données brutes de l'hôpital
    
    Args:
        df: DataFrame contenant les données brutes
    
    Returns:
        DataFrame prétraité
    """
    logger.info("Prétraitement des données hospitalières...")
    
    # Copie du DataFrame pour éviter les modifications en place
    processed_df = df.copy()
    
    # Vérifier les colonnes nécessaires
    required_columns = ['Patient', 'Date d\'arrivée', 'Date de départ', 'Durée d\'attente', 'Taux d\'occupation (lit)']
    for col in required_columns:
        if col not in processed_df.columns:
            logger.warning(f"Colonne '{col}' manquante dans les données")
    
    # Extraire l'heure de la durée d'attente
    if 'Durée d\'attente' in processed_df.columns:
        processed_df['waitingHours'] = processed_df['Durée d\'attente'].str.extract(r'(\d+)').astype(float)
        logger.info("Extraction de la durée d'attente en heures")
    
    # Calculer la durée de séjour en jours
    if 'Date d\'arrivée' in processed_df.columns and 'Date de départ' in processed_df.columns:
        processed_df['stayDuration'] = (processed_df['Date de départ'] - processed_df['Date d\'arrivée']).dt.total_seconds() / (24 * 3600)
        logger.info("Calcul de la durée de séjour en jours")
    
    # Extraire le taux d'occupation numérique
    if 'Taux d\'occupation (lit)' in processed_df.columns:
        processed_df['occupancyRate'] = processed_df['Taux d\'occupation (lit)'].str.rstrip('%').astype(float)
        logger.info("Extraction du taux d'occupation numérique")
    
    # Ajouter des indicateurs pour le type d'admission et les soins intensifs
    if 'Type d\'admissions' in processed_df.columns:
        processed_df['isEmergency'] = (processed_df['Type d\'admissions'] == 'Urgences').astype(int)
        logger.info("Ajout de l'indicateur d'admission aux urgences")
    
    if 'Soins intensifs' in processed_df.columns:
        processed_df['needsICU'] = (processed_df['Soins intensifs'] == 'Oui').astype(int)
        logger.info("Ajout de l'indicateur de soins intensifs")
    
    if 'Lit' in processed_df.columns:
        processed_df['hasBed'] = (processed_df['Lit'] == 'Oui').astype(int)
        logger.info("Ajout de l'indicateur de disponibilité de lit")
    
    # Extraire des caractéristiques de date
    if 'Date d\'arrivée' in processed_df.columns:
        processed_df['arrivalDay'] = processed_df['Date d\'arrivée'].dt.day
        processed_df['arrivalMonth'] = processed_df['Date d\'arrivée'].dt.month
        processed_df['arrivalYear'] = processed_df['Date d\'arrivée'].dt.year
        processed_df['arrivalDayOfWeek'] = processed_df['Date d\'arrivée'].dt.dayofweek
        processed_df['isWeekend'] = processed_df['arrivalDayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        logger.info("Extraction des caractéristiques temporelles")
    
    # Ajouter des indicateurs de saison
    if 'arrivalMonth' in processed_df.columns:
        processed_df['isSummer'] = processed_df['arrivalMonth'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
        processed_df['isWinter'] = processed_df['arrivalMonth'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
        processed_df['isSpring'] = processed_df['arrivalMonth'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
        processed_df['isFall'] = processed_df['arrivalMonth'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
        logger.info("Ajout des indicateurs de saison")
    
    # Traiter les valeurs manquantes
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_count = processed_df[col].isnull().sum()
        if missing_count > 0:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            logger.info(f"Remplacement de {missing_count} valeurs manquantes dans '{col}' par la médiane")
    
    logger.info("Prétraitement des données terminé")
    return processed_df

def aggregate_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données au niveau journalier pour l'analyse de séries temporelles
    
    Args:
        df: DataFrame contenant les données individuelles des patients
        
    Returns:
        DataFrame agrégé par jour
    """
    logger.info("Agrégation des données par jour...")
    
    # Créer une colonne pour la date sans l'heure
    if 'Date d\'arrivée' in df.columns:
        df['arrivalDate'] = df['Date d\'arrivée'].dt.date
    else:
        logger.error("Colonne 'Date d'arrivée' manquante, impossible d'agréger par jour")
        raise ValueError("Colonne 'Date d'arrivée' requise pour l'agrégation journalière")
    
    # Obtenir l'ensemble des dates dans les données
    all_dates = sorted(df['arrivalDate'].unique())
    logger.info(f"Période couverte: du {all_dates[0]} au {all_dates[-1]}")
    
    # Créer un DataFrame avec toutes les dates (pour s'assurer qu'aucun jour n'est manquant)
    date_range = pd.date_range(start=min(all_dates), end=max(all_dates), freq='D')
    daily_df = pd.DataFrame({'date': date_range})
    daily_df['date'] = daily_df['date'].dt.date
    
    # Agréger les métriques par jour
    daily_metrics = df.groupby('arrivalDate').agg({
        'Patient': 'count',  # Nombre total d'admissions
        'isEmergency': 'sum',  # Nombre d'admissions aux urgences
        'needsICU': 'sum',  # Nombre d'admissions nécessitant des soins intensifs
        'occupancyRate': 'mean',  # Taux d'occupation moyen
        'waitingHours': 'mean',  # Temps d'attente moyen
        'stayDuration': 'mean',  # Durée moyenne de séjour
        'arrivalMonth': 'first',  # Mois
        'arrivalYear': 'first',  # Année
        'arrivalDayOfWeek': 'first',  # Jour de la semaine
        'isWeekend': 'first',  # Indicateur weekend
        'isSummer': 'first',  # Indicateur été
        'isWinter': 'first',  # Indicateur hiver
        'isSpring': 'first',  # Indicateur printemps
        'isFall': 'first'  # Indicateur automne
    }).reset_index()
    
    # Renommer les colonnes
    daily_metrics = daily_metrics.rename(columns={
        'arrivalDate': 'date',
        'Patient': 'totalAdmissions',
        'occupancyRate': 'avgOccupancyRate'
    })
    
    logger.info(f"Données agrégées: {len(daily_metrics)} jours")
    
    # Fusionner avec le DataFrame complet de dates pour inclure les jours sans admissions
    daily_df = daily_df.merge(daily_metrics, on='date', how='left')
    logger.info(f"DataFrame complet: {len(daily_df)} jours (après ajout des jours manquants)")
    
    # Remplir les valeurs NaN pour les jours sans admissions
    daily_df['totalAdmissions'] = daily_df['totalAdmissions'].fillna(0)
    daily_df['isEmergency'] = daily_df['isEmergency'].fillna(0)
    daily_df['needsICU'] = daily_df['needsICU'].fillna(0)
    
    # Pour le taux d'occupation et autres métriques, utiliser une moyenne mobile
    daily_df['avgOccupancyRate'] = daily_df['avgOccupancyRate'].fillna(method='ffill')
    daily_df['waitingHours'] = daily_df['waitingHours'].fillna(method='ffill')
    daily_df['stayDuration'] = daily_df['stayDuration'].fillna(method='ffill')
    
    # Remplir les valeurs restantes
    for col in ['arrivalMonth', 'arrivalYear', 'arrivalDayOfWeek', 'isWeekend', 
                'isSummer', 'isWinter', 'isSpring', 'isFall']:
        # Déterminer le type de remplissage en fonction de la colonne
        if col in ['arrivalMonth', 'arrivalDayOfWeek']:
            daily_df[col] = daily_df['date'].apply(
                lambda x: x.month if col == 'arrivalMonth' else x.weekday()
            )
        elif col == 'arrivalYear':
            daily_df[col] = daily_df['date'].apply(lambda x: x.year)
        elif col == 'isWeekend':
            daily_df[col] = daily_df['arrivalDayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        elif col == 'isSummer':
            daily_df[col] = daily_df['arrivalMonth'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
        elif col == 'isWinter':
            daily_df[col] = daily_df['arrivalMonth'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
        elif col == 'isSpring':
            daily_df[col] = daily_df['arrivalMonth'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
        elif col == 'isFall':
            daily_df[col] = daily_df['arrivalMonth'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
    
    logger.info("Calcul des features avancées (lags et moyennes mobiles)")
    
    # Ajouter des features de lag et de moyenne mobile
    for lag in [1, 3, 7]:
        daily_df[f'admissionsLag{lag}'] = daily_df['totalAdmissions'].shift(lag).fillna(0)
        daily_df[f'occupancyLag{lag}'] = daily_df['avgOccupancyRate'].shift(lag).fillna(method='bfill')
    
    # Calculer les moyennes mobiles sur 7 jours
    daily_df['admissionsMA7'] = daily_df['totalAdmissions'].rolling(window=7, min_periods=1).mean()
    daily_df['occupancyMA7'] = daily_df['avgOccupancyRate'].rolling(window=7, min_periods=1).mean()
    
    logger.info("Agrégation des données par jour terminée")
    return daily_df

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, target_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divise les données en ensembles d'entraînement et de test, en respectant l'ordre temporel
    
    Args:
        df: DataFrame à diviser
        test_size: Proportion des données à utiliser pour le test
        target_columns: Liste des colonnes cibles à vérifier avant la division
        
    Returns:
        Tuple (DataFrame d'entraînement, DataFrame de test)
    """
    logger.info(f"Division des données en ensembles d'entraînement ({1-test_size:.0%}) et de test ({test_size:.0%})")
    
    # Vérifier les colonnes cibles
    if target_columns:
        for col in target_columns:
            if col not in df.columns:
                logger.warning(f"Colonne cible '{col}' non trouvée dans le DataFrame")
    
    # Trier par date si disponible
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Calculer l'indice de division
    split_idx = int(len(df) * (1 - test_size))
    
    # Diviser les données
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Division terminée: {len(train_df)} exemples d'entraînement, {len(test_df)} exemples de test")
    return train_df, test_df

def create_feature_matrix(df: pd.DataFrame, target_cols: List[str] = None, exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Crée une matrice de features et des séries cibles à partir d'un DataFrame
    
    Args:
        df: DataFrame source
        target_cols: Liste des colonnes cibles
        exclude_cols: Liste des colonnes à exclure des features
    
    Returns:
        Tuple (DataFrame de features, Dict de séries cibles)
    """
    logger.info("Création de la matrice de features et des séries cibles")
    
    # Par défaut, exclure la date
    if exclude_cols is None:
        exclude_cols = ['date']
    
    # Si aucune colonne cible n'est spécifiée, utiliser les colonnes par défaut
    if target_cols is None:
        target_cols = ['totalAdmissions', 'avgOccupancyRate']
    
    # Vérifier les colonnes cibles
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        logger.error(f"Colonnes cibles manquantes: {', '.join(missing_targets)}")
        raise ValueError(f"Colonnes cibles manquantes: {', '.join(missing_targets)}")
    
    # Créer la liste des colonnes de features
    feature_cols = [col for col in df.columns if col not in target_cols + exclude_cols]
    
    # Extraire les features et les cibles
    X = df[feature_cols]
    targets = {col: df[col] for col in target_cols}
    
    logger.info(f"Matrice de features créée avec {len(feature_cols)} variables et {len(X)} observations")
    return X, targets

def save_processed_data(df: pd.DataFrame, output_dir: str, filename: str = None) -> str:
    """
    Sauvegarde les données prétraitées dans un fichier CSV
    
    Args:
        df: DataFrame à sauvegarder
        output_dir: Répertoire de sortie
        filename: Nom du fichier (par défaut: timestamp)
    
    Returns:
        Chemin vers le fichier sauvegardé
    """
    # Créer le répertoire de sortie si nécessaire
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Répertoire créé: {output_dir}")
    
    # Générer un nom de fichier basé sur le timestamp si non spécifié
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"processed_data_{timestamp}.csv"
    
    # Construire le chemin complet
    output_path = os.path.join(output_dir, filename)
    
    # Sauvegarder le DataFrame
    df.to_csv(output_path, index=False)
    logger.info(f"Données prétraitées sauvegardées dans {output_path}")
    
    return output_path

def generate_synthetic_data(n_days: int = 365, start_date: str = '2022-01-01') -> pd.DataFrame:
    """
    Génère des données synthétiques pour tester le modèle
    
    Args:
        n_days: Nombre de jours à générer
        start_date: Date de début au format 'YYYY-MM-DD'
    
    Returns:
        DataFrame contenant les données synthétiques
    """
    logger.info(f"Génération de {n_days} jours de données synthétiques à partir de {start_date}")
    
    # Créer la séquence de dates
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Initialiser le DataFrame
    df = pd.DataFrame({
        'date': dates,
        'month': dates.month,
        'dayOfMonth': dates.day,
        'dayOfWeek': dates.dayofweek,
        'year': dates.year
    })
    
    # Ajouter des indicateurs
    df['isWeekend'] = df['dayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['isSummer'] = df['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    df['isWinter'] = df['month'].apply(lambda x: 1 if x in [12, 1, 2] else 0)
    df['isSpring'] = df['month'].apply(lambda x: 1 if x in [3, 4, 5] else 0)
    df['isFall'] = df['month'].apply(lambda x: 1 if x in [9, 10, 11] else 0)
    
    # Générer des admissions de base (moyenne de 80 par jour)
    np.random.seed(42)  # Pour la reproductibilité
    base_admissions = 80 + np.random.normal(0, 10, n_days)
    
    # Ajouter des variations saisonnières
    seasonal_factor = df['month'].apply(lambda m: 1.3 if m in [1, 2, 12] else  # Hiver : +30%
                                          0.9 if m in [6, 7, 8] else  # Été : -10%
                                          1.0)  # Printemps/Automne : normal
    
    # Ajouter des variations hebdomadaires
    weekday_factor = df['dayOfWeek'].apply(lambda d: 0.8 if d >= 5 else  # Weekend : -20%
                                            1.1 if d == 0 else  # Lundi : +10%
                                            1.0)  # Autres jours : normal
    
    # Calculer les admissions finales
    df['totalAdmissions'] = (base_admissions * seasonal_factor * weekday_factor).round()
    
    # Générer le taux d'occupation (moyenne de 75%)
    base_occupancy = 75 + np.random.normal(0, 5, n_days)
    
    # Corrélation avec les admissions
    admission_correlation = (df['totalAdmissions'] - df['totalAdmissions'].mean()) / df['totalAdmissions'].std() * 3
    
    # Calculer le taux d'occupation final
    df['avgOccupancyRate'] = np.clip(base_occupancy + admission_correlation, 50, 100).round(1)
    
    # Ajouter des pics d'activité occasionnels (épidémies, etc.)
    n_peaks = int(n_days / 60)  # Environ un pic tous les 2 mois
    peak_indices = np.random.choice(n_days, n_peaks, replace=False)
    peak_durations = np.random.randint(5, 15, n_peaks)  # Durée entre 5 et 14 jours
    
    for idx, duration in zip(peak_indices, peak_durations):
        # Éviter de déborder de la période
        end_idx = min(idx + duration, n_days)
        # Augmenter les admissions et le taux d'occupation pendant le pic
        df.loc[idx:end_idx-1, 'totalAdmissions'] = (df.loc[idx:end_idx-1, 'totalAdmissions'] * 1.5).round()
        df.loc[idx:end_idx-1, 'avgOccupancyRate'] = np.clip(df.loc[idx:end_idx-1, 'avgOccupancyRate'] * 1.2, 0, 100).round(1)
    
    # Calculer les features de lag et moyenne mobile
    for lag in [1, 3, 7]:
        df[f'admissionsLag{lag}'] = df['totalAdmissions'].shift(lag).fillna(0)
        df[f'occupancyLag{lag}'] = df['avgOccupancyRate'].shift(lag).fillna(method='bfill')
    
    # Calculer les moyennes mobiles sur 7 jours
    df['admissionsMA7'] = df['totalAdmissions'].rolling(window=7, min_periods=1).mean()
    df['occupancyMA7'] = df['avgOccupancyRate'].rolling(window=7, min_periods=1).mean()
    
    logger.info(f"Données synthétiques générées avec succès: {len(df)} enregistrements")
    return df

def load_latest_processed_data(data_dir: str) -> pd.DataFrame:
    """
    Charge le fichier de données prétraitées le plus récent
    
    Args:
        data_dir: Répertoire contenant les données prétraitées
        
    Returns:
        DataFrame contenant les données prétraitées
    """
    logger.info(f"Recherche du fichier de données prétraitées le plus récent dans {data_dir}")
    
    # Vérifier si le répertoire existe
    if not os.path.exists(data_dir):
        logger.error(f"Le répertoire {data_dir} n'existe pas")
        raise FileNotFoundError(f"Le répertoire {data_dir} n'existe pas")
    
    # Trouver tous les fichiers CSV dans le répertoire
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.error(f"Aucun fichier CSV trouvé dans {data_dir}")
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {data_dir}")
    
    # Trier par date de modification (le plus récent d'abord)
    csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
    
    # Charger le fichier le plus récent
    latest_file = csv_files[0]
    file_path = os.path.join(data_dir, latest_file)
    
    logger.info(f"Chargement du fichier le plus récent: {latest_file}")
    df = load_data_from_csv(file_path)
    
    return df