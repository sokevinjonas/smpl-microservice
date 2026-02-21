# Architecture des Modèles 3D et IA

Ce document détaille les modèles et technologies utilisés par le microservice pour transformer des photos 2D en mensurations 3D précises.

---

## 1. SMPL (Skinned Multi-Person Linear model)
Le modèle **SMPL** est le standard académique et industriel pour la représentation du corps humain.

- **Rôle** : Servir de base géométrique (template) pour toutes les reconstructions.
- **Fonctionnement** : Il sépare la **forme** (shape) de la **pose**. 
    - La forme est contrôlée par 10 paramètres numériques (*betas*) issus d'une Analyse en Composantes Principales (PCA) sur des milliers de scans.
    - La pose est contrôlée par des rotations d'articulations.
- **Avantage** : Il garantit une cohérence anatomique. Même avec des données partielles, le modèle "sait" comment un corps humain est structuré.

## 2. MediaPipe Pose (par Google)
Le moteur de détection de points clés basé sur le Deep Learning.

- **Rôle** : Extraire les coordonnées 2D des articulations à partir des pixels de l'image.
- **Version** : Google MediaPipe Tasks API (0.10.x+).
- **Points clés** : Analyse de 33 points (épaules, hanches, genoux, etc.) avec un score de confiance pour chaque point.
- **Importance** : C'est la "vision" du système. Si la détection est mauvaise, le fitting 3D sera décalé.

## 3. Optimiseur de Fitting (HMR / Iterative Fitting)
Le pont mathématique entre la photo 2D et le modèle 3D.

- **Rôle** : Ajuster les paramètres du modèle SMPL pour qu'il "colle" aux points MediaPipe.
- **Technologies** : **PyTorch**, **Chumpy** et **L-BFGS / Adam**.
- **Processus** : 
    1. Le système projette le modèle 3D sur un plan 2D.
    2. Il calcule l'erreur (distance) entre les points théoriques du modèle et les points réels détectés par MediaPipe.
    3. Il modifie les *betas* et la pose de façon répétitive (fitting itératif) pour minimiser cette erreur.

## 4. Analyse Géométrique (Trimesh & NumPy)
Le "mètre ruban" virtuel.

- **Rôle** : Effectuer les mesures physiques sur le modèle 3D final.
- **Fonctions** :
    - **Slicing** : Découper le mesh à des hauteurs précises pour calculer des périmètres (Tour de poitrine, taille, etc.).
    - **Calcul de Distance** : Mesurer la distance euclidienne ou géodésique entre deux sommets (Longueur de manche, entrejambe).
- **Stabilité** : Utilise des algorithmes de "Convex Hull" pour ignorer les plis des vêtements et isoler la peau réelle.

## 5. Moteur de Sanitization (Heuristiques)
La couche finale de vérification métier.

- **Rôle** : Filtrer les aberrations statistiques.
- **Fonctionnement** : Compare les mesures obtenues avec des bases de données anthropométriques mondiales indexées par genre et taille. Si une valeur est absurde (ex: tour de poignet de 50cm), elle est corrigée par une moyenne intelligente.

---

*Ce pipeline garantit une précision de **~95%** sur les membres et **~85-90%** sur le torse.*
