# src/utils/visualization.py

import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from typing import List, Tuple
from matplotlib.figure import Figure

logger = logging.getLogger('hospital_prediction.visualization')

def plot_predictions(dates, y_true, y_pred, title="Prédictions vs Réalité"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, y_true, label='Valeurs réelles', marker='o')
    ax.plot(dates, y_pred, linestyle='--', label='Prédictions')
    ax.set_xlabel("Dates")
    ax.set_ylabel("Valeurs")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig

def save_figure(fig, output_dir: str, filename: str, dpi: int = 300):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
    return file_path
