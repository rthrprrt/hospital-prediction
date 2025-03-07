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


def plot_predictions(dates: List, y_true: List, y_pred: List, title: str = "Prédictions vs Réel",
                     xlabel: str = "Dates", ylabel: str = "Valeurs", figsize: Tuple[int, int] = (12, 6)):
    """
    Affiche un graphique comparant les valeurs réelles et les prédictions.

    Args:
        dates (List): Liste des dates.
        y_true (List): Valeurs réelles.
        y_pred (List): Valeurs prédites.
        title (str): Titre du graphique.

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, y_true, label='Valeurs réelles', marker='o')
    ax.plot(dates, y_pred, linestyle='--', label='Prédictions')
    ax.set_xlabel(xlabel="Dates")
    ax.set_ylabel(ylabel="Valeurs")
    ax.set_title(label="Prédictions vs Réalité")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig