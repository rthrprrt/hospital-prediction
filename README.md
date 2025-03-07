# Projet de Prédiction pour l'Hôpital Pitié-Salpêtrière

## 🏥 Contexte du Projet

Ce projet vise à développer un modèle de machine learning avancé pour prédire les admissions hospitalières et le taux d'occupation des lits à la Pitié-Salpêtrière. L'objectif est de fournir des outils d'anticipation pour optimiser la gestion des ressources hospitalières.

### Objectifs Principaux

- Prédire le nombre quotidien d'admissions
- Estimer le taux d'occupation des lits
- Identifier les périodes potentielles de pics d'activité
- Aider à la prise de décision pour l'allocation des ressources

## 🛠 Technologies Utilisées

- Python 3.9+
- Bibliothèques de Machine Learning :
  - Scikit-learn
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
- Conteneurisation :
  - Docker
  - Docker Compose

## 📂 Structure du Projet

```
HOPITAL-PREDICTION/
├── data/                          # Données
│   ├── raw/                       # Données brutes
│   └── processed/                 # Données prétraitées
├── logs/                          # Journaux d'exécution
├── models/                        # Modèles entraînés
├── predictions/                   # Prédictions générées
├── src/
│   ├── hospital_prediction/       # Package principal
│   │   ├── model.py               # Classe du modèle de prédiction
│   │   ├── data_processor.py      # Prétraitement des données
│   │   ├── train.py               # Fonctions d'entraînement
│   │   └── predict.py             # Fonctions de prédiction
│   ├── scripts/                   # Scripts exécutables
│   │   ├── train_model.py         # Script d'entraînement
│   │   ├── generate_predictions.py# Script de génération de prédictions
│   │   └── validate_model.py      # Script de validation du modèle
│   └── utils/                     # Utilitaires
│       ├── visualization.py       # Fonctions de visualisation
│       └── metrics.py             # Métriques de performance
├── docker-compose.yml             # Configuration multi-conteneurs (optionnel)
├── Dockerfile                     # Configuration du conteneur Docker
├── requirements.txt               # Dépendances Python
└── README.md                      # Documentation du projet
```

## 🚀 Installation et Configuration

### Prérequis

- Python 3.9+
- Docker (optionnel mais recommandé)
- pip

### Installation Standard

1. Cloner le dépôt
```bash
git clone https://github.com/votre-utilisateur/hospital-prediction.git
cd hospital-prediction
```

2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### Installation avec Docker

1. Construire l'image
```bash
docker build -t hospital-prediction .
```

2. Utiliser Docker Compose
```bash
docker-compose up train  # Pour entraîner le modèle
docker-compose up predict  # Pour générer des prédictions
```

## 🔍 Utilisation

### Entraînement du Modèle

```bash
# Avec Python
python -m src.scripts.train_model \
    --data-file data/raw/hospital_data.csv \
    --output-dir models \
    --test-size 0.2 \
    --optimize

# Avec Docker
docker run --entrypoint python \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  hospital-prediction \
  -m src.scripts.train_model \
  --data-file "/app/data/raw/hospital_data.csv" \
  --output-dir "/app/models" \
  --test-size 0.2
```

### Génération de Prédictions

```bash
# Avec Python
python -m src.scripts.generate_predictions \
    --data-file data/processed/daily_data.csv \
    --model-dir models \
    --output-dir predictions \
    --days 30

# Avec Docker
docker run --entrypoint python \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/predictions:/app/predictions" \
  hospital-prediction \
  -m src.scripts.generate_predictions \
  --data-file "/app/data/processed/processed_data.csv" \
  --model-dir "/app/models" \
  --output-dir "/app/predictions" \
  --days 7
```

### Validation du Modèle

```bash
# Avec Python
python -m src.scripts.validate_model \
    --model-path models \
    --data-file data/processed/test_data.csv \
    --output-dir validation_results
```

# Avec Docker
docker run --entrypoint python \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  hospital-prediction \
  -m src.scripts.validate_model \
  --model-path "/app/models" \
  --data-file "/app/data/processed/test_data.csv" \
  --output-dir "/app/validation_results"

## 📊 Métriques et Performances

- Métriques de validation :
  - Erreur Absolue Moyenne (MAE)
  - Erreur Quadratique Moyenne (MSE)
  - Coefficient de Détermination (R²)
  - Erreur Absolue Pourcentage Moyenne (MAPE)

## Questions Fréquentes (FAQ)
Comment modifier les paramètres du modèle ?
Les hyperparamètres sont définis dans src/hospital_prediction/model.py dans la méthode train().

Comment adapter la période de prédiction ?
Utilisez l’argument --days dans le script generate_predictions.py.

Comment mettre à jour le modèle en production ?
Re-construisez l’image Docker après chaque mise à jour du code source et utilisez un système de versioning pour vos modèles et métadonnées.

Pourquoi y a-t-il des différences entre VS Code et GitHub ?
Assurez-vous que toutes vos modifications locales sont bien commitées et poussées sur GitHub. Ce README mis à jour reflète la structure et l'utilisation actuelles du projet.

