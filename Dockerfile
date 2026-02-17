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
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Nettoyer pip cache et réinstaller proprement
RUN pip install --upgrade pip setuptools wheel && \
    pip cache purge && \
    pip install -r requirements.txt --no-cache-dir

# Copier le code application
COPY app.py .
COPY smpl_engine.py .
COPY utils/ ./utils/
COPY models/ ./models/

# Exposer le port
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Lancer l'application
CMD ["python", "app.py"]
