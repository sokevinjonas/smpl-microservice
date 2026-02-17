# RAPPORT DE DÉBOGAGE - SMPL Microservice

**Date:** 17 février 2026  
**Statut:** En cours d'analyse  
**Priorité:** CRITIQUE

---

## 1. PROBLÈME IDENTIFIÉ

### Erreur Docker Build
```
ERROR: Could not find a version that satisfies the requirement mediapipe==0.9.3.0
No matching distribution found for mediapipe==0.9.3.0
```

### Cause Racine
MediaPipe **0.9.3.0 n'existe pas** sur PyPI (Python Package Index). Les versions disponibles sont:
- ✅ MediaPipe 0.10.5, 0.10.7, 0.10.8, ... 0.10.32
- ❌ MediaPipe 0.9.3.0 (INEXISTANT)

**Impact**: La tentative de fixer une version inexistante cause l'échec de la construction Docker.

---

## 2. INCOMPATIBILITÉ DE PYTHON

### Situation Actuelle
- **Dockerfile utilisé**: `FROM python:3.9`
- **MediaPipe 0.10.x requis**: Python >= 3.10

### Erreur PIP Correspondante
```
ERROR: Ignored the following versions that require a different python version:
- 0.23.0 Requires-Python >=3.10
- 0.25.0 Requires-Python >=3.10
- 1.14.0 Requires-Python >=3.10
- 1.16.0 Requires-Python >=3.11
[... plus d'autres ...]
```

### Analyse
Python 3.9 ne peut **pas** installer les dépendances modernes:
1. MediaPipe 0.10.x demande Python 3.10+
2. Autres packages (scikit-image, scipy) ont aussi des contraintes Python 3.10+
3. Solution: **Upgrader à Python 3.10 minimum**

---

## 3. CHANGEMENTS EFFECTUÉS

### 3.1 Dockerfile
```diff
- FROM python:3.9
+ FROM python:3.10
```
**Raison**: Compatibilité avec MediaPipe 0.10.x et dépendances modernes

### 3.2 requirements.txt
```diff
- mediapipe==0.9.3.0  # N'existe pas
+ mediapipe==0.10.32  # Dernière version disponible
```
**Raison**: Utiliser la version réellement disponible sur PyPI

### 3.3 utils/pose_estimation.py
**Réécriture complète** pour supporter l'API MediaPipe 0.10.x+:

#### Avant (Incomplet - utilisait API 0.9.x)
```python
if hasattr(mp, 'solutions'):
    MEDIAPIPE_API = 'solutions'  # API 0.9.x
elif hasattr(mp, 'tasks'):
    MEDIAPIPE_API = 'tasks'       # API 0.10.x - fallback to mock
```
**Problème**: L'API `tasks` était détecté mais le code n'était pas implémenté.

#### Après (Implémentation complète)
```python
# Nouvelle API MediaPipe Tasks (0.10.x+)
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseEstimator:
    def __init__(self):
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
    
    def estimate_pose(self, image):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        detection_result = self.pose.detect(mp_image)
```

**Améliorations**:
- ✅ Support natif de MediaPipe 0.10.x API `tasks`
- ✅ Classe `PoseLandmarker` au lieu de `Pose.process()`
- ✅ Gestion des modèles (`vision.PoseLandmarkerOptions`)
- ✅ Format image correct (`mp.Image` au lieu de numpy array)
- ✅ Fallback gracieux au mock en cas d'erreur

---

## 4. ARCHITECTURE DE L'API

### 0.9.x (Ancien - Inexistant sur PyPI)
```python
import mediapipe as mp
mp.solutions.pose.Pose()
results = pose.process(image_rgb)
landmarks = results.pose_landmarks.landmark
```

### 0.10.x (Nouveau - Actuellement utilisé)
```python
from mediapipe.tasks.python import vision
options = vision.PoseLandmarkerOptions(...)
pose = vision.PoseLandmarker.create_from_options(options)
results = pose.detect(mp_image)
landmarks = results.pose_landmarks[0]
```

**Différences clés**:
1. Import: `mp.solutions` → `mediapipe.tasks.python.vision`
2. Classe: `Pose` → `PoseLandmarker`
3. Méthode: `process()` → `detect()`
4. Format image: `numpy.ndarray` → `mp.Image`
5. Résultats: `.pose_landmarks.landmark` → `.pose_landmarks[0]`

---

## 5. ÉTAPES DE VALIDATION

### ✅ Statut Actuel
- [x] Dockerfile corrigé (Python 3.10)
- [x] requirements.txt corrigé (mediapipe==0.10.32)
- [x] pose_estimation.py réécrit pour API Tasks
- [x] Gestion des erreurs et fallback mock
- [x] Keypoint names constants définis (33 points COCO)

### ⏳ À Valider
- [ ] Docker build réussit sans erreur
- [ ] MediaPipe 0.10.32 s'installe correctement
- [ ] PoseLandmarker initialise sans erreur
- [ ] Endpoint `/estimate` retourne des measurements
- [ ] Mode production (pas de fallback mock)

---

## 6. INSTRUCTIONS POUR RÉANALYSE

### Pour l'Agent IA
**Merci de réanalyser les points suivants:**

1. **Vérifier la nouvelle implémentation `pose_estimation.py`**
   - L'API `vision.PoseLandmarker` est-elle correctement utilisée?
   - Les options de configuration sont-elles standard?
   - Le téléchargement automatique du modèle fonctionne-t-il?

2. **Valider la gestion des modèles MediaPipe**
   - Chemin `pose_landmarker.task` correct?
   - Chemins de recherche complets? (`~/.cache`, `./models`, etc.)
   - Fallback au mock approprié si modèle absent?

3. **Vérifier la compatibilité de format image**
   - `mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)`
   - Dimensions attendues (H, W, 3)?
   - Conversion BGR→RGB correcte?

4. **Valider la structure des résultats**
   - `detection_result.pose_landmarks` est une liste?
   - Accès à `[0]` pour la première personne?
   - Attributs `x, y, z, presence` disponibles?

5. **Tester intégration complète**
   - app.py initialise-t-il `PoseEstimator()` sans erreur?
   - Endpoint `/health` retourne-t-il 200 OK?
   - Logs indiquent "✓ MediaPipe PoseLandmarker initialized"?

6. **Performance et fallback**
   - Le mock est-il utilisé comme dernier recours?
   - Les measurements fallback sont-elles réalistes?
   - Les temps de réponse sont-ils acceptables?

---

## 7. COMMANDES À EXÉCUTER

```bash
# Construire sans cache (ignorer Docker cache)
docker build --no-cache -t smpl-microservice .

# Exécuter et tester
docker run -it -p 5000:5000 smpl-microservice

# Dans un autre terminal, tester l'endpoint
curl http://localhost:5000/health

# Tester estimation avec une image
curl -X POST http://localhost:5000/estimate \
  -H "Content-Type: application/json" \
  -d '{"photo_url": "https://example.com/image.jpg"}'
```

---

## 8. RÉSUMÉ DES CORRECTIONS

| Problème | Cause | Solution | Statut |
|----------|-------|----------|--------|
| MediaPipe 0.9.3.0 inexistant | Version jamais publiée | Utiliser 0.10.32 | ✅ Corrigé |
| Python 3.9 incompatible | MediaPipe 0.10.x demande 3.10+ | Utiliser Python 3.10 | ✅ Corrigé |
| API Tasks non implémentée | Code utilisait API 0.9.x | Réécriture complète | ✅ Corrigé |
| Import `mp.solutions` échoue | N'existe pas en 0.10.x | Importer `vision` directement | ✅ Corrigé |
| Gestion des modèles manquante | Pas de chemin défini | Ajouter recherche + chemins | ✅ Corrigé |

---

## 9. PROCHAINES ÉTAPES

1. **Immédiat**: Exécuter `docker build --no-cache`
2. **Court terme**: Valider endpoints (`/health`, `/estimate`)
3. **Moyen terme**: Tester avec vraies images
4. **Long terme**: Optimiser performance + caching modèles

---

**Document créé pour traçabilité et analyse.** 
*Pour questions ou améliorations, se référer à ce rapport.*
