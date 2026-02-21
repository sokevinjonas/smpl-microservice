# Datasets Avancés pour la Précision Millimétrique

Pour réduire la marge d'erreur (Tolérance +/- 10mm), nous devons entraîner ou calibrer le modèle sur des données disposant de scans 3D de haute qualité (Ground Truth).

---

## 🚀 Recommandations de Datasets

### 1. AGORA (Synthetic & Real humans in 3D)
C'est actuellement l'un des meilleurs datasets pour le fitting de corps complet.
- **Volume** : 14 000 images haute résolution.
- **Ground Truth** : Paramètres SMPL-X ultra-précis générés par des experts.
- **Utilité** : Parfait pour calibrer la corrélation entre les points clés 2D et le volume 3D réel.
- **Lien** : [agora.is.tue.mpg.de](https://agora.is.tue.mpg.de/)

### 2. 3DPW (3D Peoples in the Wild)
Images réelles capturées en extérieur avec des capteurs IMU pour la vérité terrain.
- **Volume** : 60 séquences vidéo (milliers de frames).
- **Ground Truth** : Poses et formes SMPL vérifiées.
- **Utilité** : Tester la robustesse face aux vêtements de tous les jours et aux arrière-plans complexes.
- **Lien** : [virtualhumans.mpi-inf.mpg.de/3DPW/](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

### 3. SURREAL (Synthetic Humans)
Entièrement synthétique mais permet une échelle massive.
- **Volume** : 6 millions de frames.
- **Ground Truth** : Tout est connu (profondeur, segmentation, SMPL).
- **Utilité** : Idéal pour "pré-entraîner" un modèle de correction de vêtements (Clothing Compensation).
- **Lien** : [di.ens.fr/willow/research/surreal/](https://www.di.ens.fr/willow/research/surreal/data/)

### 4. NOMAD (Diverse Poses and Clothing)
Spécifiquement conçu pour l'analyse de personnes habillées.
- **Utilité** : Essentiel pour réduire l'erreur sur le tour de poitrine/taille causée par l'épaisseur des textiles.

---

## 🛠️ Stratégie pour atteindre +/- 10mm

1.  **Calibration de la "Loi de Puissance"** : Utiliser **AGORA** pour vérifier si notre logique de mesure (slicing) sur-estime ou sous-estime systématiquement certains membres.
2.  **Clothing Compensation Model** : Utiliser **SURREAL** pour simuler des épaisseurs de vêtements et apprendre à l'IA à "deviner" le corps sous les habits.
3.  **Cross-Validation** : Faire tourner `evaluate_model.py` sur ces 4 datasets combinés pour obtenir une MAE globale fiable.

---

*Note : La plupart de ces datasets nécessitent une inscription (gratuite pour la recherche) sur les sites officiels des instituts (Max Planck Institute, etc.).*
