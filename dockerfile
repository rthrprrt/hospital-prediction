# Dockerfile pour le projet de prédiction hospitalière

# Utiliser une image de base Python officielle
FROM python:3.9-slim-buster

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu du projet
COPY . .

# Créer les répertoires nécessaires
RUN mkdir -p /app/data/raw \
    && mkdir -p /app/data/processed \
    && mkdir -p /app/models \
    && mkdir -p /app/predictions \
    && mkdir -p /app/logs

# Variables d'environnement
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Script par défaut pour lancer les tests ou les prédictions
ENTRYPOINT ["python", "-m", "src.scripts.generate_predictions"]

# Arguments par défaut (peuvent être surchargés)
CMD ["--help"]