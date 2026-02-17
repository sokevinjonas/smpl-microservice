#!/usr/bin/env python3
"""
Script de configuration pour les modèles SMPL.
Utilisation: python setup_models.py
"""

import os
from pathlib import Path
import sys

def setup_smpl_models():
    """Configure les modèles SMPL."""
    
    models_dir = Path('./models')
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Configuration des modèles SMPL")
    print("=" * 60)
    
    try:
        import smplx
        print("✓ smplx importé")
    except ImportError:
        print("❌ smplx non installé. Exécute: pip install smplx")
        return False
    
    # Vérifier si les fichiers existent
    model_path = models_dir / 'SMPL_NEUTRAL.npz'
    
    if model_path.exists():
        print(f"✓ Modèle SMPL trouvé: {model_path}")
        return True
    else:
        print(f"\n⚠️  Fichiers modèles SMPL non trouvés")
        print(f"   Chemin attendu: {model_path.resolve()}\n")
        
        print("📥 Pour obtenir les fichiers modèles:")
        print("   1. Aller sur: https://smpl.is.tue.mpg.de/")
        print("   2. Créer un compte gratuit")
        print("   3. Télécharger 'SMPL v1.0 (Neutral, zip)'")
        print("   4. Extraire et copier SMPL_NEUTRAL.npz dans ./models/")
        print(f"   5. Chemin complet: {models_dir.resolve()}/SMPL_NEUTRAL.npz\n")
        
        print("⏱️  En attendant, l'app démarre avec un modèle synthétique")
        print("   (Les mensurations seront moins précises)")
        
        return False


def setup_mediapipe_models():
    """Configure les modèles MediaPipe."""
    try:
        import mediapipe as mp
        
        print("\n" + "=" * 60)
        print("Configuration de MediaPipe")
        print("=" * 60)
        
        print(f"✓ MediaPipe version: {mp.__version__}")
        print("✓ Les modèles seront téléchargés à la première utilisation")
        
        return True
    except Exception as e:
        print(f"❌ Erreur MediaPipe: {e}")
        return False


def main():
    """Fonction principale."""
    print("\nPréparation de l'environnement SMPL Microservice\n")
    
    # Setup SMPL
    smpl_ok = setup_smpl_models()
    
    # Setup MediaPipe
    mp_ok = setup_mediapipe_models()
    
    print("\n" + "=" * 60)
    if mp_ok:
        if smpl_ok:
            print("✓ Configuration complète!")
        else:
            print("⚠️  Configuration partiellement complète")
        print("  L'application démarre maintenant...")
        print("  http://localhost:5000/health")
        return 0
    else:
        print("❌ Configuration incomplète")
        return 1


if __name__ == '__main__':
    sys.exit(main())

