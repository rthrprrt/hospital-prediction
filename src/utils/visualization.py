# src/utils/visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import List, Dict, Union, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hospital_prediction.visualization')

# Configuration du style de visualisation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

def set_plotting_style(style: str = 'seaborn-v0_8-whitegrid', context: str = 'notebook', font_scale: float = 1.1) -> None:
    """
    Configure le style global pour les visualisations
    
    Args:
        style: Style matplotlib à utiliser
        context: Contexte seaborn ('paper', 'notebook', 'talk', 'poster')
        font_scale: Échelle de la taille des polices
    """
    plt.style.use(style)
    sns.set_context(context, font_scale=font_scale)
    logger.info(f"Style de visualisation configuré: {style}, contexte: {context}")

def plot_admissions_trend(df: pd.DataFrame, date_col: str = 'date', admissions_col: str = 'totalAdmissions',
                          rolling_window: int = 7, figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Visualise la tendance des admissions au fil du temps
    
    Args:
        df: DataFrame contenant les données
        date_col: Nom de la colonne de date
        admissions_col: Nom de la colonne des admissions
        rolling_window: Fenêtre pour la moyenne mobile
        figsize: Taille de la figure (largeur, hauteur)
        
    Returns:
        Figure matplotlib
    """
    logger.info("Création du graphique de tendance des admissions")
    
    # Convertir la colonne de date en datetime si nécessaire
    df = df.copy()
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tracer les données brutes
    ax.plot(df[date_col], df[admissions_col], 'o-', alpha=0.5, label='Données journalières')
    
    # Ajouter une moyenne mobile
    if rolling_window > 1:
        rolling_mean = df[admissions_col].rolling(window=rolling_window, center=True).mean()
        ax.plot(df[date_col], rolling_mean, 'r-', linewidth=2, 
                label=f'Moyenne mobile ({rolling_window} jours)')
    
    # Ajouter des indications pour les saisons
    if max(df[date_col]) - min(df[date_col]) > timedelta(days=180):
        # Obtenir toutes les années uniques dans les données
        years = df[date_col].dt.year.unique()
        
        for year in years:
            # Hiver (décembre à février)
            winter_start = pd.to_datetime(f"{year-1}-12-01") if year > min(years) else None
            winter_end = pd.to_datetime(f"{year}-02-28")
            
            # Été (juin à août)
            summer_start = pd.to_datetime(f"{year}-06-01")
            summer_end = pd.to_datetime(f"{year}-08-31")
            
            # Ajouter des bandes pour les saisons si elles sont dans la plage de données
            if winter_start and winter_start >= min(df[date_col]) and winter_end <= max(df[date_col]):
                ax.axvspan(winter_start, winter_end, alpha=0.2, color='lightblue', label='_nolegend_')
            
            if summer_start >= min(df[date_col]) and summer_end <= max(df[date_col]):
                ax.axvspan(summer_start, summer_end, alpha=0.2, color='lightyellow', label='_nolegend_')
    
    # Configurer le graphique
    ax.set_title('Évolution des admissions hospitalières', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Nombre d\'admissions', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Formater l'axe des dates
    fig.autofmt_xdate()
    
    # Ajouter des annotations pour les valeurs extrêmes
    max_idx = df[admissions_col].idxmax()
    max_date = df.loc[max_idx, date_col]
    max_value = df.loc[max_idx, admissions_col]
    
    ax.annotate(f'Max: {max_value}',
                xy=(max_date, max_value),
                xytext=(10, 10),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    plt.tight_layout()
    
    return fig

def plot_occupancy_trend(df: pd.DataFrame, date_col: str = 'date', occupancy_col: str = 'avgOccupancyRate',
                          threshold: float = 90.0, rolling_window: int = 7, 
                          figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Visualise la tendance du taux d'occupation au fil du temps
    
    Args:
        df: DataFrame contenant les données
        date_col: Nom de la colonne de date
        occupancy_col: Nom de la colonne du taux d'occupation
        threshold: Seuil d'alerte pour le taux d'occupation
        rolling_window: Fenêtre pour la moyenne mobile
        figsize: Taille de la figure (largeur, hauteur)
        
    Returns:
        Figure matplotlib
    """
    logger.info("Création du graphique de tendance du taux d'occupation")
    
    # Convertir la colonne de date en datetime si nécessaire
    df = df.copy()
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tracer les données brutes
    ax.plot(df[date_col], df[occupancy_col], 'o-', alpha=0.5, label='Données journalières')
    
    # Ajouter une moyenne mobile
    if rolling_window > 1:
        rolling_mean = df[occupancy_col].rolling(window=rolling_window, center=True).mean()
        ax.plot(df[date_col], rolling_mean, 'r-', linewidth=2, 
                label=f'Moyenne mobile ({rolling_window} jours)')
    
    # Ajouter une ligne pour le seuil critique
    ax.axhline(y=threshold, color='red', linestyle='--', 
               label=f'Seuil critique ({threshold}%)')
    
    # Mettre en surbrillance les périodes au-dessus du seuil
    above_threshold = df[df[occupancy_col] >= threshold]
    if not above_threshold.empty:
        ax.scatter(above_threshold[date_col], above_threshold[occupancy_col], 
                  color='red', s=50, label='Périodes critiques')
    
    # Configurer le graphique
    ax.set_title('Évolution du taux d\'occupation des lits', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Taux d\'occupation (%)', fontsize=12)
    ax.set_ylim(0, 105)  # Limite l'axe des y entre 0 et 105%
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Formater l'axe des dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    return fig

def plot_seasonal_patterns(df: pd.DataFrame, value_col: str = 'totalAdmissions', 
                           figsize: Tuple[int, int] = (16, 10)) -> Figure:
    """
    Visualise les motifs saisonniers dans les données
    
    Args:
        df: DataFrame contenant les données
        value_col: Nom de la colonne à analyser
        figsize: Taille de la figure (largeur, hauteur)
        
    Returns:
        Figure matplotlib
    """
    logger.info(f"Analyse des motifs saisonniers pour la colonne '{value_col}'")
    
    # Vérifier que les colonnes nécessaires existent
    required_cols = ['month', 'dayOfWeek', value_col]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Colonne manquante: {col}")
            raise ValueError(f"La colonne {col} est requise pour l'analyse saisonnière")
    
    # Créer la figure avec 4 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Tendance mensuelle
    monthly_data = df.groupby('month')[value_col].agg(['mean', 'std'])
    ax1 = axes[0, 0]
    ax1.bar(monthly_data.index, monthly_data['mean'], yerr=monthly_data['std'], 
            alpha=0.7, capsize=5)
    ax1.set_title(f'Tendance mensuelle - {value_col}', fontsize=12)
    ax1.set_xlabel('Mois', fontsize=10)
    ax1.set_ylabel('Moyenne', fontsize=10)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                          'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Tendance hebdomadaire
    weekly_data = df.groupby('dayOfWeek')[value_col].agg(['mean', 'std'])
    ax2 = axes[0, 1]
    ax2.bar(weekly_data.index, weekly_data['mean'], yerr=weekly_data['std'], 
            alpha=0.7, capsize=5)
    ax2.set_title(f'Tendance hebdomadaire - {value_col}', fontsize=12)
    ax2.set_xlabel('Jour de la semaine', fontsize=10)
    ax2.set_ylabel('Moyenne', fontsize=10)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
    ax2.grid(True, alpha=0.3)
    
    # 3. Comparaison été vs hiver
    df['season'] = df['month'].apply(lambda m: 'Été' if m in [6, 7, 8] else 
                                         'Hiver' if m in [12, 1, 2] else
                                         'Printemps' if m in [3, 4, 5] else 'Automne')
    
    seasonal_data = df.groupby('season')[value_col].agg(['mean', 'std', 'count'])
    ax3 = axes[1, 0]
    seasons_order = ['Hiver', 'Printemps', 'Été', 'Automne']
    season_means = [seasonal_data.loc[s, 'mean'] if s in seasonal_data.index else 0 for s in seasons_order]
    season_stds = [seasonal_data.loc[s, 'std'] if s in seasonal_data.index else 0 for s in seasons_order]
    
    ax3.bar(range(len(seasons_order)), season_means, yerr=season_stds, 
            alpha=0.7, capsize=5)
    ax3.set_title(f'Comparaison par saison - {value_col}', fontsize=12)
    ax3.set_xlabel('Saison', fontsize=10)
    ax3.set_ylabel('Moyenne', fontsize=10)
    ax3.set_xticks(range(len(seasons_order)))
    ax3.set_xticklabels(seasons_order)
    ax3.grid(True, alpha=0.3)
    
    # 4. Weekend vs jours de semaine
    df['is_weekend'] = df['dayOfWeek'].apply(lambda d: 'Weekend' if d >= 5 else 'Semaine')
    weekend_data = df.groupby('is_weekend')[value_col].agg(['mean', 'std', 'count'])
    ax4 = axes[1, 1]
    
    weekend_means = [weekend_data.loc[s, 'mean'] if s in weekend_data.index else 0 
                     for s in ['Semaine', 'Weekend']]
    weekend_stds = [weekend_data.loc[s, 'std'] if s in weekend_data.index else 0 
                    for s in ['Semaine', 'Weekend']]
    
    ax4.bar(['Semaine', 'Weekend'], weekend_means, yerr=weekend_stds, 
            alpha=0.7, capsize=5)
    ax4.set_title(f'Semaine vs Weekend - {value_col}', fontsize=12)
    ax4.set_xlabel('Période', fontsize=10)
    ax4.set_ylabel('Moyenne', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance: Dict[str, float], title: str = 'Importance des features',
                            figsize: Tuple[int, int] = (10, 6), top_n: int = 10) -> Figure:
    """
    Visualise l'importance des features pour un modèle
    
    Args:
        feature_importance: Dictionnaire des importances de features
        title: Titre du graphique
        figsize: Taille de la figure (largeur, hauteur)
        top_n: Nombre de features à afficher
        
    Returns:
        Figure matplotlib
    """
    logger.info(f"Création du graphique d'importance des features (top {top_n})")
    
    if not feature_importance:
        logger.error("Le dictionnaire d'importance des features est vide")
        raise ValueError("Le dictionnaire d'importance des features est vide")
    
    # Trier les features par importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    if top_n:
        sorted_features = sorted_features[:top_n]
    
    # Extraire les noms et valeurs
    feature_names, importance_values = zip(*sorted_features)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Créer le graphique
    y_pos = range(len(feature_names))
    ax.barh(y_pos, importance_values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Le plus important en haut
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(importance_values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    return fig

def plot_correlation_matrix(df: pd.DataFrame, cols: List[str] = None, 
                            figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    Visualise la matrice de corrélation entre les variables
    
    Args:
        df: DataFrame contenant les données
        cols: Liste des colonnes à inclure (toutes par défaut)
        figsize: Taille de la figure (largeur, hauteur)
        
    Returns:
        Figure matplotlib
    """
    logger.info("Création de la matrice de corrélation")
    
    # Sélectionner les colonnes
    if cols:
        df = df[cols]
    else:
        # Par défaut, sélectionner uniquement les colonnes numériques
        df = df.select_dtypes(include=[np.number])
    
    # Calculer la matrice de corrélation
    corr_matrix = df.corr()
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Créer la heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    ax.set_title('Matrice de corrélation', fontsize=14)
    
    plt.tight_layout()
    
    return fig

def plot_prediction_vs_actual(dates: List[Union[str, datetime]], y_pred: List[float], 
                              y_actual: List[float] = None, future_dates: List[Union[str, datetime]] = None, 
                              y_future: List[float] = None, title: str = 'Prédiction vs Réel',
                              y_label: str = 'Valeur', threshold: float = None,
                              figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """
    Compare les prédictions aux valeurs réelles
    
    Args:
        dates: Liste des dates pour les données historiques
        y_pred: Liste des valeurs prédites pour les données historiques
        y_actual: Liste des valeurs réelles (optionnel)
        future_dates: Liste des dates pour les prédictions futures (optionnel)
        y_future: Liste des prédictions futures (optionnel)
        title: Titre du graphique
        y_label: Étiquette de l'axe y
        threshold: Seuil d'alerte à afficher (optionnel)
        figsize: Taille de la figure
        
    Returns:
        Figure matplotlib
    """
    logger.info(f"Création du graphique de comparaison prédiction vs réel: {title}")
    
    # Convertir les dates en datetime si ce sont des chaînes
    if isinstance(dates[0], str):
        dates = [pd.to_datetime(d) for d in dates]
    
    if future_dates and isinstance(future_dates[0], str):
        future_dates = [pd.to_datetime(d) for d in future_dates]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tracer les prédictions historiques
    ax.plot(dates, y_pred, 'b-', label='Prédictions')
    
    # Tracer les valeurs réelles si disponibles
    if y_actual is not None:
        ax.plot(dates, y_actual, 'g-', label='Valeurs réelles')
    
    # Tracer les prédictions futures si disponibles
    if future_dates and y_future:
        # Ajouter une ligne verticale pour séparer historique et prédictions
        last_historical_date = dates[-1]
        ax.axvline(x=last_historical_date, color='r', linestyle='--', label='Aujourd\'hui')
        
        # Tracer les prédictions futures
        ax.plot(future_dates, y_future, 'b--', label='Prédictions futures')
        
        # Ajouter des indications pour les pics d'activité
        if threshold:
            peak_indices = [i for i, val in enumerate(y_future) if val >= threshold]
            if peak_indices:
                peak_dates = [future_dates[i] for i in peak_indices]
                peak_values = [y_future[i] for i in peak_indices]
                ax.scatter(peak_dates, peak_values, color='red', s=50, label='Pics d\'activité')
    
    # Ajouter une ligne pour le seuil si spécifié
    if threshold:
        ax.axhline(y=threshold, color='orange', linestyle='--', 
                   label=f'Seuil d\'alerte ({threshold})')
    
    # Configurer le graphique
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Formater l'axe des dates
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    return fig

def plot_weekly_patterns(df: pd.DataFrame, value_col: str = 'totalAdmissions', 
                          date_col: str = 'date', figsize: Tuple[int, int] = (14, 7)) -> Figure:
    """
    Visualise les tendances hebdomadaires dans les données
    
    Args:
        df: DataFrame contenant les données
        value_col: Nom de la colonne à analyser
        date_col: Nom de la colonne de date
        figsize: Taille de la figure
        
    Returns:
        Figure matplotlib
    """
    logger.info(f"Analyse des tendances hebdomadaires pour la colonne '{value_col}'")
    
    # Préparer les données
    df = df.copy()
    if df[date_col].dtype != 'datetime64[ns]':
        df[date_col] = pd.to_datetime(df[date_col])
    
    df['dayOfWeek'] = df[date_col].dt.dayofweek
    df['weekOfYear'] = df[date_col].dt.isocalendar().week
    df['year'] = df[date_col].dt.year
    
    # Calculer la moyenne par jour de la semaine
    day_means = df.groupby('dayOfWeek')[value_col].mean()
    
    # Créer un heatmap par semaine et jour de la semaine
    pivot_data = df.pivot_table(index='weekOfYear', columns='dayOfWeek', 
                               values=value_col, aggfunc='mean')
    
    # Créer la figure avec 2 sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]})
    
    # 1. Moyenne par jour de la semaine
    ax1.bar(day_means.index, day_means.values, alpha=0.7)
    ax1.set_title('Moyenne par jour de la semaine', fontsize=12)
    ax1.set_xlabel('Jour de la semaine', fontsize=10)
    ax1.set_ylabel(f'Moyenne de {value_col}', fontsize=10)
    ax1.set_xticks(range(7))
    ax1.set_xticklabels(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Heatmap des semaines
    sns.heatmap(pivot_data, cmap='YlGnBu', ax=ax2, cbar_kws={'label': value_col})
    ax2.set_title('Valeurs par semaine et jour', fontsize=12)
    ax2.set_xlabel('Jour de la semaine', fontsize=10)
    ax2.set_ylabel('Semaine de l\'année', fontsize=10)
    ax2.set_xticklabels(['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
    
    plt.tight_layout()
    
    return fig

def save_figure(fig: Figure, output_dir: str, filename: str = None, dpi: int = 300) -> str:
    """
    Sauvegarde une figure matplotlib dans un fichier
    
    Args:
        fig: Figure matplotlib à sauvegarder
        output_dir: Répertoire de sortie
        filename: Nom du fichier (par défaut: timestamp)
        dpi: Résolution de l'image
        
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
        filename = f"figure_{timestamp}.png"
    elif not filename.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
        filename += '.png'
    
    # Construire le chemin complet
    output_path = os.path.join(output_dir, filename)
    
    # Sauvegarder la figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Figure sauvegardée dans {output_path}")
    
    return output_path

def create_dashboard_visualizations(historical_data: pd.DataFrame, future_predictions: pd.DataFrame,
                                   output_dir: str = 'dashboard/assets') -> Dict[str, str]:
    """
    Crée un ensemble de visualisations pour le tableau de bord
    
    Args:
        historical_data: DataFrame contenant les données historiques
        future_predictions: DataFrame contenant les prédictions futures
        output_dir: Répertoire pour sauvegarder les visualisations
        
    Returns:
        Dictionnaire des chemins vers les fichiers générés
    """
    logger.info("Création des visualisations pour le tableau de bord")
    
    # Créer le répertoire de sortie si nécessaire
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    visualizations = {}
    
    # 1. Tendance des admissions
    fig1 = plot_admissions_trend(historical_data)
    visualizations['admissions_trend'] = save_figure(fig1, output_dir, 'admissions_trend.png')
    
    # 2. Tendance du taux d'occupation
    fig2 = plot_occupancy_trend(historical_data)
    visualizations['occupancy_trend'] = save_figure(fig2, output_dir, 'occupancy_trend.png')
    
    # 3. Motifs saisonniers pour les admissions
    fig3 = plot_seasonal_patterns(historical_data, 'totalAdmissions')
    visualizations['admissions_seasonal'] = save_figure(fig3, output_dir, 'admissions_seasonal.png')
    
    # 4. Motifs saisonniers pour le taux d'occupation
    fig4 = plot_seasonal_patterns(historical_data, 'avgOccupancyRate')
    visualizations['occupancy_seasonal'] = save_figure(fig4, output_dir, 'occupancy_seasonal.png')
    
    # 5. Tendances hebdomadaires
    fig5 = plot_weekly_patterns(historical_data)
    visualizations['weekly_patterns'] = save_figure(fig5, output_dir, 'weekly_patterns.png')
    
    # 6. Prédictions futures pour les admissions
    if not future_predictions.empty:
        # Préparer les données
        historical_dates = pd.to_datetime(historical_data['date'].tail(30))
        historical_values = historical_data['totalAdmissions'].tail(30)
        historical_pred = historical_data['totalAdmissions'].tail(30)  # À remplacer par les prédictions réelles
        
        future_dates = pd.to_datetime(future_predictions['date'])
        future_values = future_predictions['predicted_admissions']
        
        # Générer le graphique
        fig6 = plot_prediction_vs_actual(
            dates=historical_dates,
            y_pred=historical_pred,
            y_actual=historical_values,
            future_dates=future_dates,
            y_future=future_values,
            title='Prédiction des admissions hospitalières',
            y_label='Nombre d\'admissions',
            threshold=future_predictions.get('admission_threshold', [None])[0]
        )
        visualizations['admissions_prediction'] = save_figure(fig6, output_dir, 'admissions_prediction.png')
        
        # 7. Prédictions futures pour le taux d'occupation
        fig7 = plot_prediction_vs_actual(
            dates=historical_dates,
            y_pred=historical_data['avgOccupancyRate'].tail(30),
            y_actual=historical_data['avgOccupancyRate'].tail(30),
            future_dates=future_dates,
            y_future=future_predictions['predicted_occupancy'],
            title='Prédiction du taux d\'occupation des lits',
            y_label='Taux d\'occupation (%)',
            threshold=90.0
        )
        visualizations['occupancy_prediction'] = save_figure(fig7, output_dir, 'occupancy_prediction.png')
    
    logger.info(f"{len(visualizations)} visualisations créées dans {output_dir}")
    return visualizations