# Configuration Docker Optimisée (Taille réduite)

# 1. Utiliser une image de base légère (Debian Slim)
FROM python:3.10-slim

# Éviter les fichiers pyc et buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Installer uniquement les dépendances système minimales
# libgl1/libglib2.0 pour OpenCV, curl pour téléchargement
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Installer PyTorch CPU-only D'ABORD (pour éviter de télécharger la version CUDA lourde)
# Cela réduit l'image de ~2 Go
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# 4. Installer les autres dépendances
# chumpy est installé avec --no-build-isolation pour compatibilité
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation chumpy && \
    pip install --no-cache-dir -r requirements.txt

# Créer le répertoire models
RUN mkdir -p /app/models

# Télécharger le modèle MediaPipe (mise en cache layer)
RUN curl -L -o /app/pose_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task || true

# 5. Copier application (change souvent, donc à la fin)
COPY app.py smpl_engine.py setup_models.py ./
COPY utils/ ./utils/
# On ne copie PAS le dossier models/ local s'il est gros, 
# on préfère le monter via docker-compose ou le télécharger.
# Mais pour l'image autonome, on peut copier s'il contient des fichiers légers.
COPY models/ ./models/

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["sh", "-c", "python setup_models.py && python app.py"]