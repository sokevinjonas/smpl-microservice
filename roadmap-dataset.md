# Roadmap Évaluation Modèle SMPL

Ce document de référence décrit la stratégie d'évaluation de la fiabilité de notre API d'estimation de mensurations basées sur SMPL. L'objectif est de mesurer avec précision la Marge d'Erreur Moyenne Absolue (MAE) de notre solution en la confrontant à des données terrestres réelles (Ground Truth).

Nous utiliserons trois datasets complémentaires pour tester le modèle sous différents angles.

## 1. Le Dataset Principal : Body Measurements Image Dataset (Kaggle)

Ce dataset sera notre base de référence principale pour l'évaluation empirique des mensurations anthropométriques (circonférences et distances).

*   **Lien / Source :** Kaggle / Hugging Face.
*   **Format :** Plus de 13 000 photographies (face et profil) pour plus de 1 000 individus.
*   **Ground Truth (Vraies Données) :** Mesures réelles incluant taille, tour de poitrine, tour de taille et tour de hanches.
*   **Pourquoi l'utiliser ?**
    *   **Correspondance exacte avec notre pipeline :** Il permet de tester spécifiquement la logique multi-vues de `smpl_engine.py` (qui demande une vue de face et une vue de profil).
    *   **Volume important :** 1 000 individus fournissent une base statistique solide pour calculer la MAE avec précision.
    *   **Variables directes :** Les mesures fournies correspondent directement aux réponses attendues par notre API (ex: `tour_poitrine`, `bassin`, `tour_taille`).
*   **Objectif de test :** Évaluer la précision finale des extractions 3D (la logique de slicing de `utils/mesh_utils.py` sur le mesh final).

## 2. Le Dataset Géométrique : SSP-3D (Sports Shape and Pose 3D)

Ce dataset permet d'isoler et de tester la performance stricte du fitting SMPL (la génération 3D) en s'affranchissant du style vestimentaire.

*   **Contenu :** 311 images d'athlètes portant des vêtements moulants.
*   **Ground Truth :** Paramètres de forme (Shape parameters - `betas`) du modèle SMPL générés à partir de scans 3D de haute qualité.
*   **Pourquoi l'utiliser ?**
    *   **Isoler l'erreur de "Shape" :** En comparant les `betas` générés par notre API avec les vrais `betas` du scan, on peut vérifier si notre optimiseur (dans `fit_model_to_multiple_views`) converge vers le bon volume corporel.
    *   **Pas d'interférence vestimentaire :** Les vêtements moulants permettent de s'assurer que si erreur il y a, elle vient du modèle (pose estimation / fitting) et non d'un t-shirt trop large.
*   **Objectif de test :** Valider mathématiquement la qualité de l'optimisation SMPL (minimisation de la loss) et certifier l'estimation de volume corporel (`betas`).

## 3. Le Dataset "Real-World" : HBW (Human Bodies in the Wild)

Ce dataset sert de stress-test ("crash test") pour évaluer la robustesse du système dans des conditions variables et moins idéales.

*   **Contenu :** Photographies d'individus dans la vie quotidienne (environnements extérieurs "in the wild", éclairages complexes).
*   **Ground Truth :** Squelettes et paramètres de pose alignés manuellement (ou semi-automatiquement) avec le modèle SMPL-X.
*   **Pourquoi l'utiliser ?**
    *   **Robustesse de MediaPipe :** Tester si MediaPipe (dans `utils/pose_estimation.py`) échoue ou est confus face à des arrières-plans chargés ou des postures inhabituelles.
    *   **Généralisation :** S'assurer que le modèle est fiable non seulement en studio ou sur fond blanc, mais aussi avec les photos aléatoires envoyées par de vrais utilisateurs depuis leur domicile.
*   **Objectif de test :** Évaluer le taux de rejet de l'API (erreurs 400 sur la détection de pose) et la dégradation de la précision selon les conditions d'éclairage ou l'arrière-plan.

## Méthodologie Proposée (Benchmarker)

Une fois ces datasets acquis (idéalement téléchargés localement et organisés), la validation se fera via un script de benchmarking automatique (ex: `evaluate_model.py`) :

1.  **Boucle de test :** Le script itère sur chaque individu d'un dataset.
2.  **Inférence API :** Il envoie les images de l'individu à notre API locale (`/estimate`).
3.  **Comparaison :** Il compare la réponse JSON avec la "Ground Truth" du dataset.
4.  **Calcul d'erreur :** Il calcule l'Erreur Absolue Moyenne (MAE) pour chaque type de mensuration (ex: MAE_poitrine = 2.1cm, MAE_taille = 1.8cm).
5.  **Génération de rapport :** Enregistrement des résultats et détection des biais systématiques.
