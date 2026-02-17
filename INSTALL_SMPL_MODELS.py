#!/usr/bin/env python3
"""
Téléchargeur de modèles SMPL - Guide manuel
"""

import os
from pathlib import Path

def print_setup_guide():
    """Affiche le guide de configuration."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                    CONFIGURATION DES MODÈLES SMPL                          ║
╚════════════════════════════════════════════════════════════════════════════╝

⚠️  Les fichiers modèles SMPL ne sont pas disponibles.

Le modèle SMPL nécessite un authentification. Voici les étapes:

📥 TÉLÉCHARGEMENT MANUEL:
═══════════════════════════════════════════════════════════════════════════

1. Aller sur: https://smpl.is.tue.mpg.de/

2. Créer un compte (gratuit pour la recherche)

3. Accepter les conditions et télécharger:
   - SMPL v1.0 (Neutral, zip)

4. Extraire le fichier ZIP:
   - Vous obtiendrez un dossier "models_smpl_v_1_0_0_nm"

5. Copier les fichiers .npz:
   - Chercher: SMPL_NEUTRAL.npz
   - Copier dans: ./models/

   Exemple Linux/Mac:
   cp models_smpl_v_1_0_0_nm/SMPL_NEUTRAL.npz ./models/

6. Relancer l'application:
   sudo docker-compose up --build

═══════════════════════════════════════════════════════════════════════════

🔗 LIEN DIRECT:
   https://smpl.is.tue.mpg.de/download.php?type=releases&id=1

⏱️  Une fois les fichiers placés dans ./models/, l'app démarre normalement.

═══════════════════════════════════════════════════════════════════════════
""")

if __name__ == '__main__':
    print_setup_guide()
