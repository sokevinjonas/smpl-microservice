#!/usr/bin/env python3
"""
Script de configuration pour télécharger les modèles SMPL.
Utilisation: python setup_models.py
"""

import os
from pathlib import Path
import sys

def setup_smpl_models():
    """Télécharge les modèles SMPL via smplx."""
    
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
    
    # Essayer de télécharger via le script intégré de smplx
    try:
        print("\n⏳ Téléchargement des modèles SMPL...")
        print("   (Cela peut prendre quelques minutes)")
        
        # Le modèle SMPL est téléchargé à la première utilisation
        # On essaie simplement de créer une instance
        smpl = smplx.create(
            model_path=str(models_dir),
            model_type='smpl',
            gender='neutral',
            batch_size=1,
            device='cpu',
            create_transl=True,
            create_expression=False,
            ext='npz'
        )
        
        print("✓ Modèles SMPL téléchargés/chargés avec succès!")
        print(f"✓ Fichiers sauvegardés dans: {models_dir.resolve()}")
        
        # Vérifier les fichiers
        files = list(models_dir.glob("SMPL*"))
        if files:
            print(f"\n✓ Fichiers présents:")
            for f in files:
                size_mb = f.stat().st_size / (1024**2)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Erreur de téléchargement: {e}")
        print("\n📥 Téléchargement manuel:")
        print("   1. Aller sur: https://smpl.is.tue.mpg.de/")
        print("   2. Créer un compte et accepter les conditions")
        print("   3. Télécharger SMPL v1.0 (NEUTRAL, .npz)")
        print(f"   4. Placer dans: {models_dir.resolve()}/")
        print("   5. Relancer ce script ou l'application")
        return False
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("\nSi le problème persiste:")
        print("   1. Télécharge manuellement depuis https://smpl.is.tue.mpg.de/")
        print(f"   2. Place les fichiers .npz dans {models_dir.resolve()}/")
        return False


def setup_mediapipe_models():
    """Configure les modèles MediaPipe."""
    try:
        import mediapipe as mp
        
        print("\n" + "=" * 60)
        print("Configuration de MediaPipe")
        print("=" * 60)
        
        print(f"✓ MediaPipe version: {mp.__version__}")
        
        # MediaPipe télécharge les modèles automatiquement
        print("✓ Les modèles MediaPipe seront téléchargés à la première utilisation")
        
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
    if smpl_ok and mp_ok:
        print("✓ Configuration complète!")
        print("  Tu peux maintenant démarrer l'application:")
        print("  - Local: python app.py")
        print("  - Docker: docker-compose up")
        return 0
    else:
        print("⚠️ Configuration incomplète")
        print("  Résous les erreurs ci-dessus et réessaie")
        return 1


if __name__ == '__main__':
    sys.exit(main())
