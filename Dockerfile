# Configuration Docker pour le microservice SMPL

FROM python:3.10

# Installer les dépendances système requises pour OpenCV et MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \                
    libglib2.0-0 \
    libssl-dev \
    libopencv-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt en premier pour profiter du cache Docker
COPY requirements.txt .

# Mettre à jour pip et installer les dépendances Python
RUN pip install --upgrade pip setuptools wheel && \
    pip cache purge && \
    pip install -r requirements.txt --no-cache-dir

# Télécharger le modèle PoseLandmarker de MediaPipe (version "full")
# Le fichier sera placé à la racine de /app, accessible en lecture par l'application
RUN curl -L -o /app/pose_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task

# (Optionnel) Vérifier que le modèle a bien été téléchargé
RUN ls -lh /app/pose_landmarker.task

# Copier le code applicatif
COPY app.py .
COPY smpl_engine.py .
COPY utils/ ./utils/
COPY models/ ./models/

# Exposer le port utilisé par Flask
EXPOSE 5000

# Healthcheck (nécessite curl installé)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Lancer l'application
CMD ["python", "app.py"]