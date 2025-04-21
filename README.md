# Projet de Prédiction pour l'Hôpital Pitié-Salpêtrière

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🏥 Contexte du Projet

Ce projet vise à développer un modèle de machine learning avancé pour prédire les admissions hospitalières quotidiennes et le taux d'occupation des lits à l'hôpital Pitié-Salpêtrière (ou basé sur des données similaires). L'objectif est de fournir des outils d'anticipation pour optimiser la gestion des ressources hospitalières.

### Objectifs Principaux

-   Prédire le nombre quotidien d'admissions (`totalAdmissions`).
-   Estimer le taux d'occupation moyen des lits (`avgOccupancyRate`).
-   Identifier les périodes potentielles de pics d'activité via des seuils configurables.
-   Aider à la prise de décision pour l'allocation des ressources.

## Modèle Utilisé

Le modèle combine deux approches de régression :

1.  **Admissions Quotidiennes:** Un modèle `GradientBoostingRegressor` est utilisé pour prédire le nombre total d'admissions.
2.  **Taux d'Occupation:** Un modèle `RandomForestRegressor` est utilisé pour prédire le taux d'occupation moyen des lits.

Les features utilisées incluent des informations temporelles (mois, jour, année, jour de la semaine, weekend), des indicateurs saisonniers, ainsi que des valeurs passées (lags) et des moyennes mobiles des admissions et du taux d'occupation.

## 🛠 Technologies Utilisées

-   Python 3.9+
-   Bibliothèques Principales :
    -   Scikit-learn (`sklearn`)
    -   Pandas
    -   NumPy
    -   Matplotlib
    -   Seaborn
    -   Joblib (pour la sauvegarde/chargement des modèles)
    -   Openpyxl (pour lire les fichiers `.xlsx`)
-   Conteneurisation (Optionnel) :
    -   Docker

## 📂 Structure du Projet

```
HOPITAL-PREDICTION/
├── data/                       # Données (ignoré par Git)
│   ├── raw/                    # Données brutes (ex: hospital_data.xlsx)
│   └── processed/              # Données prétraitées (ex: daily_aggregated_data.csv)
├── logs/                       # Journaux d'exécution (si configuré, ignoré par Git)
├── models/                     # Modèles entraînés (ignoré par Git)
├── predictions/                # Prédictions générées (ignoré par Git)
├── validation_results/         # Résultats de validation (ignoré par Git)
├── src/                        # Code source
│   ├── hospital_prediction/    # Package principal de logique métier
│   │   ├── __init__.py
│   │   ├── model.py            # Classe du modèle de prédiction (HospitalPredictionModel)
│   │   ├── data_processor.py   # Prétraitement, agrégation, feature engineering
│   │   ├── train.py            # Fonctions et logique d'entraînement détaillées
│   │   └── predict.py          # Fonctions et logique de prédiction détaillées
│   ├── scripts/                # Scripts exécutables pour les pipelines
│   │   ├── __init__.py
│   │   ├── train_model.py      # Script principal d'entraînement
│   │   ├── generate_predictions.py # Script principal de génération de prédictions
│   │   └── validate_model.py   # Script principal de validation du modèle
│   └── utils/                  # Utilitaires partagés
│       ├── __init__.py
│       ├── visualization.py    # Fonctions de visualisation
│       └── metrics.py          # Métriques de performance
├── docker-compose.yml          # (Optionnel) Configuration multi-conteneurs
├── Dockerfile                  # Configuration du conteneur Docker
├── requirements.txt            # Dépendances Python
├── .gitignore                  # Fichiers et dossiers ignorés par Git
├── LICENSE                     # Licence du projet (MIT)
└── README.md                   # Documentation du projet (ce fichier)
```

## 🚀 Installation et Configuration

### Prérequis

-   Python 3.9 ou supérieur
-   pip (gestionnaire de paquets Python)
-   Git
-   Docker (optionnel, pour la conteneurisation)

### Installation Standard

1.  **Cloner le dépôt :**
    ```bash
    git clone <URL_DU_REPO> # Remplacez par l'URL de votre dépôt
    cd HOPITAL-PREDICTION
    ```

2.  **Créer et activer un environnement virtuel :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

### Installation avec Docker

*(Note : Un `Dockerfile` et éventuellement un `docker-compose.yml` fonctionnels sont nécessaires pour ces commandes.)*

1.  **Construire l'image Docker :**
    ```bash
    docker build -t hospital-prediction-app .
    ```

*(Les exemples `docker run` ci-dessous supposent que les répertoires `data`, `models`, `predictions`, `validation_results` existent localement et seront mappés dans le conteneur sous `/app/`)*

2.  **Utiliser `docker run` (Exemples):**

    *   **Entraînement :**
        ```bash
        # Assurez-vous que data/raw/your_data.csv existe ou utilisez --use-synthetic
        docker run --rm \
          -v "$(pwd)/data:/app/data" \
          -v "$(pwd)/models:/app/models" \
          hospital-prediction-app \
          python -m src.scripts.train_model --data-file /app/data/raw/your_data.csv --model-dir /app/models --output-dir /app/data/processed
          # Ou avec des données synthétiques:
          # docker run --rm -v "$(pwd)/models:/app/models" hospital-prediction-app python -m src.scripts.train_model --use-synthetic --model-dir /app/models
        ```

    *   **Prédiction :**
        ```bash
        # Assurez-vous que data/processed/daily_aggregated_data.csv (ou autre nom) et les modèles existent
        docker run --rm \
          -v "$(pwd)/data:/app/data" \
          -v "$(pwd)/models:/app/models" \
          -v "$(pwd)/predictions:/app/predictions" \
          hospital-prediction-app \
          python -m src.scripts.generate_predictions --data-file /app/data/processed/daily_aggregated_data.csv --model-dir /app/models --output-dir /app/predictions --days 14
        ```

    *   **Validation :**
        ```bash
        # Assurez-vous que data/processed/daily_aggregated_data.csv (ou autre nom) et les modèles existent
        docker run --rm \
          -v "$(pwd)/data:/app/data" \
          -v "$(pwd)/models:/app/models" \
          -v "$(pwd)/validation_results:/app/validation_results" \
          hospital-prediction-app \
          python -m src.scripts.validate_model --data-file /app/data/processed/daily_aggregated_data.csv --model-path /app/models --output-dir /app/validation_results
        ```

## 💾 Données

-   Les données brutes (ex: `hospital_data.xlsx` ou `.csv`) doivent être placées dans `data/raw/`.
-   Le script d'entraînement (`train_model.py`) prétraite les données brutes et sauvegarde les données agrégées par jour dans `data/processed/`.
-   Le format attendu des données brutes (si non synthétiques) doit contenir à minima des colonnes permettant de dériver :
    -   Une date d'événement (ex: 'Date d\'arrivée')
    -   Un identifiant patient/événement pour compter les admissions
    -   Une mesure du taux d'occupation (ex: 'Taux d\'occupation (lit)')
    -   D'autres colonnes optionnelles utilisées pour le feature engineering (ex: 'Durée d\'attente', 'Type d\'admissions', 'Soins intensifs'). Voir `src/hospital_prediction/data_processor.py` pour les détails du prétraitement.
-   Si aucune donnée n'est fournie, l'option `--use-synthetic` du script `train_model.py` peut être utilisée pour générer et entraîner sur des données synthétiques.

## 🔍 Utilisation des Scripts

*(Exécutez ces commandes depuis la racine du projet après avoir activé l'environnement virtuel ou via Docker comme montré précédemment.)*

### 1. Entraînement du Modèle

Entraîne les modèles d'admission et de taux d'occupation et les sauvegarde dans le répertoire `models/`.

```bash
python -m src.scripts.train_model \
    --data-file data/raw/your_hospital_data.xlsx \ # ou .csv
    --model-dir models \
    --output-dir data/processed \
    # --optimize               # Optionnel: pour optimiser les hyperparamètres (plus long)
    # --cross-validation       # Optionnel: pour effectuer une validation croisée temporelle
    # --use-synthetic          # Optionnel: pour forcer l'utilisation de données synthétiques
```

### 2. Génération de Prédictions

Charge le dernier modèle entraîné depuis `models/` et génère des prédictions pour les N prochains jours. Sauvegarde les prédictions (`predictions_*.csv`), les alertes (`alerts_*.json`) et un graphique (`predictions_plot_*.png`) dans `predictions/`.

```bash
python -m src.scripts.generate_predictions \
    --data-file data/processed/daily_aggregated_data.csv \ # Utilise les dernières données agrégées
    --model-dir models \
    --output-dir predictions \
    --days 30 # Nombre de jours futurs à prédire
```

### 3. Validation du Modèle

Évalue le dernier modèle entraîné sur un jeu de données (typiquement les données complètes pour re-splitter ou un jeu de test dédié). Sauvegarde les métriques (`validation_metrics.json`) et des graphiques d'analyse (`predictions_vs_reality.png`, `residuals.png`, `residuals_distribution.png`) dans `validation_results/`.

```bash
python -m src.scripts.validate_model \
    --model-path models \ # Chemin vers le répertoire des modèles
    --data-file data/processed/daily_aggregated_data.csv \ # Données à utiliser pour la validation
    --output-dir validation_results \
    --test-size 0.2 # Proportion des données pour le set de test lors de la validation
```

### 4. Mise à jour des données et Réentraînement

Pour ajouter de nouvelles données et mettre à jour le modèle :

Ajouter/Mettre à jour les Données Brutes : Placez le nouveau fichier de données complet (ou mis à jour) dans `data/raw/`.

Réentraîner le Modèle : Exécutez à nouveau le script d'entraînement. Il re-prétraitera les données et entraînera le modèle sur l'ensemble actualisé.

```bash
python -m src.scripts.train_model --data-file data/raw/new_hospital_data.xlsx --model-dir models
```

(Optionnel) Valider le Nouveau Modèle : Exécutez le script de validation sur les nouvelles données pour vérifier les performances.

```bash
python -m src.scripts.validate_model --model-path models --data-file data/processed/daily_aggregated_data.csv --output-dir validation_results
```

Générer de Nouvelles Prédictions : Utilisez le script de prédiction avec le modèle nouvellement entraîné.

```bash
python -m src.scripts.generate_predictions --data-file data/processed/daily_aggregated_data.csv --model-dir models --output-dir predictions --days 14
```

## 📊 Métriques et Performances

Les scripts de validation et d'entraînement rapportent plusieurs métriques standard de régression :

- **MAE** (Mean Absolute Error): Erreur absolue moyenne. Interprétation directe dans l'unité de la cible.
- **MSE** (Mean Squared Error): Erreur quadratique moyenne. Pénalise davantage les grosses erreurs.
- **RMSE** (Root Mean Squared Error): Racine carrée de la MSE. Dans l'unité de la cible.
- **R²** (Coefficient de Détermination): Proportion de la variance de la cible expliquée par le modèle (entre -∞ et 1, plus proche de 1 est meilleur).
- **MAPE** (Mean Absolute Percentage Error): Erreur absolue moyenne en pourcentage (peut être problématique si les valeurs réelles sont proches de zéro).

Les résultats sont enregistrés dans les fichiers JSON générés par les scripts `validate_model.py` et `train_model.py`.

## ❓ Questions Fréquentes (FAQ)

**Comment modifier les hyperparamètres du modèle ?**
- Les paramètres par défaut sont dans `src/hospital_prediction/model.py`.
- Utilisez l'option `--optimize` du script `train_model.py` pour lancer une recherche par grille (GridSearchCV) pour trouver les meilleurs paramètres (ceci peut être long). Les grilles de recherche sont définies dans `src/hospital_prediction/train.py`.

**Comment adapter la période de prédiction future ?**
- Utilisez l'argument `--days` du script `generate_predictions.py`.

**Comment utiliser un modèle spécifique au lieu du plus récent ?**
- Les scripts `generate_predictions.py` et `validate_model.py` chargent actuellement le modèle le plus récent basé sur le timestamp dans le nom de fichier dans le répertoire `--model-dir` ou `--model-path`. Pour utiliser un modèle spécifique, vous devrez modifier légèrement la logique de chargement dans `src/hospital_prediction/predict.py` (`load_prediction_model`) ou passer le chemin complet vers le fichier .joblib spécifique au script de validation via `--model-path`.

**Comment le modèle gère-t-il les données manquantes ?**
- Le prétraitement dans `src/hospital_prediction/data_processor.py` et `src/hospital_prediction/model.py` impute les valeurs numériques manquantes avec la médiane et les valeurs catégorielles avec le mode. L'agrégation journalière gère les jours manquants en les remplissant avec 0 pour les comptes et en utilisant ffill (forward fill) pour les taux/moyennes.

## 📜 Licence

Ce projet est distribué sous la licence MIT. Voir le fichier LICENSE pour plus de détails.