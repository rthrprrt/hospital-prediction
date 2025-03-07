# src/utils/visualization.py
import matplotlib.pyplot as plt
import os

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

def plot_feature_importance(feature_names, importances, title="Importance des Features"):
    fig, ax = plt.subplots(figsize=(12, 6))
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]

    ax.barh(sorted_features, sorted_importances, align='center')
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig
