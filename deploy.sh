#!/bin/bash
# VPS Deployment Script for SMPL Microservice
# Script exécuté côté SERVEUR VPS lors du CI/CD (GitHub Actions)

set -e # Arrête le script à la moindre erreur

echo "======================================"
echo "🚀 Début du déploiement Docker..."
echo "======================================"

# Vérifier si Docker et Docker Compose sont installés
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé sur ce VPS. Installation requise."
    exit 1
fi
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé sur ce VPS. Installation requise."
    exit 1
fi

echo "✅ Environnement Docker vérifié."

# Forcer la reconstruction et le redémarrage (Downtime minimalisé par Compose)
echo "🔄 Construction (Build) et Redémarrage..."
# --build garantit que les modifications du repo sont intégrées dans l'image
docker-compose up -d --build

# Nettoyage des vieilles images Docker inutiles (gain de place sur VPS)
echo "🧹 Nettoyage des anciennes images Docker (Prune)..."
docker image prune -f

echo "======================================"
echo "✅ DEPLOIEMENT REUSSI ! Le microservice tourne sur le port 5000."
echo "======================================"
