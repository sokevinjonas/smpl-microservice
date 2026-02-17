
# README – Microservice Python SMPL + HMR/SPIN pour estimation de mensurations

## 1️⃣ Objectif

Ce microservice Python reçoit une **photo d’une personne** et une liste de mesures à extraire, puis :

1. Détecte la personne et ses **points clés du corps**.
2. Génère un **mesh 3D du corps** via **SMPL + HMR/SPIN**.
3. Extrait les **mensurations demandées**.
4. Retourne un JSON avec les mesures pour intégration dans le backend Laravel.

---

## 2️⃣ Prérequis

* **Python 3.10+**
* **GPU recommandé** pour vitesse de traitement (CUDA 11+)
* Librairies Python :

  ```bash
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
  pip install flask flask-cors requests trimesh pytorch3d opencv-python numpy
  pip install mediapipe # si tu utilises MediaPipe pour pose estimation
  ```
* Téléchargement des modèles pré-entraînés :

  * **SMPL model files** : [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
  * **HMR/SPIN pré-entraîné** (PyTorch) : [https://github.com/nkolot/3DHumanPose](https://github.com/nkolot/3DHumanPose)
  * **Pose estimation** (OpenPose ou MediaPipe Pose)

---

## 3️⃣ Structure du projet

```
smpl-microservice/
├─ app.py               # Point d’entrée Flask
├─ smpl_engine.py       # Contient HMR/SPIN + extraction mesh et mensurations
├─ utils/
│  ├─ pose_estimation.py # Wrapper OpenPose / MediaPipe
│  ├─ mesh_utils.py      # Mesures sur le mesh SMPL
├─ models/              # Modèles pré-entraînés (SMPL, HMR/SPIN)
├─ requirements.txt     # Liste des dépendances
└─ README.md
```

---

## 4️⃣ Endpoint REST

### POST `/estimate`

**Paramètres :**

```json
{
  "photo_url": "https://monsite.com/tmp/photo123.jpg",
  "measures_table": ["tour_poitrine", "taille", "hanche", "longueur_bras"]
}
```

* `photo_url` : URL ou chemin temporaire de la photo
* `measures_table` : liste des mesures à calculer

**Réponse :**

```json
{
  "tour_poitrine": 92,
  "taille": 70,
  "hanche": 98,
  "longueur_bras": 62
}
```

---

## 5️⃣ Pipeline interne

1. **Récupération de la photo**

   * Téléchargement temporaire depuis `photo_url`
   * Vérification format (jpg/png)

2. **Détection de la personne**

   * MediaPipe Pose ou OpenPose
   * Extraction des keypoints nécessaires pour HMR/SPIN

3. **Reconstruction 3D via HMR/SPIN**

   * Génération des paramètres SMPL
   * Création du mesh 3D du corps

4. **Extraction des mensurations**

   * Pour chaque mesure dans `measures_table`, calcul sur le mesh :

     * Tour de poitrine → distance horizontale autour du thorax
     * Taille → distance autour du nombril
     * Hanches → distance autour des hanches
     * Longueur bras / jambes → distance entre joints
   * Retour JSON

5. **Nettoyage**

   * Suppression du fichier photo temporaire
   * Libération de la mémoire GPU si nécessaire

---

## 6️⃣ Exemple minimal Flask (`app.py`)

```python
from flask import Flask, request, jsonify
from smpl_engine import estimate_measures

app = Flask(__name__)

@app.route("/estimate", methods=["POST"])
def estimate():
    data = request.json
    photo_url = data.get("photo_url")
    measures_table = data.get("measures_table", [])

    if not photo_url or not measures_table:
        return jsonify({"error": "photo_url and measures_table required"}), 400

    try:
        result = estimate_measures(photo_url, measures_table)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

---

## 7️⃣ Exemple d’interface SMPL (`smpl_engine.py`)

```python
import cv2
import numpy as np
from utils.pose_estimation import get_keypoints
from utils.mesh_utils import extract_measurements

def estimate_measures(photo_url, measures_table):
    # 1. Charger l'image
    img = cv2.imread(photo_url)

    # 2. Extraire les keypoints via MediaPipe/OpenPose
    keypoints = get_keypoints(img)

    # 3. Générer mesh SMPL via HMR/SPIN
    smpl_mesh = generate_smpl_mesh(img, keypoints)  # fonction interne SPIN/HMR

    # 4. Extraire les mesures demandées
    measures = extract_measurements(smpl_mesh, measures_table)

    return measures
```

> Le microservice Python peut être déployé en **Docker** pour faciliter l’intégration avec Laravel et assurer l’isolation GPU.

---

## 8️⃣ Considérations importantes

* **GPU obligatoire pour production** pour traitement < 5s
* **Supprimer photo temporaire** après traitement
* **Retourner uniquement les mesures demandées** selon `measures_table`
* **Sécurité** : SSL, CORS configuré pour Laravel
* **Logs** : stocker uniquement mesures et erreurs, jamais la photo originale

---

## 9️⃣ Déploiement Docker (optionnel)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

* Utiliser `docker run --gpus all -p 5000:5000 smpl-microservice` pour lancer

---

Si tu veux, je peux te rédiger **la suite complète avec `pose_estimation.py` et `mesh_utils.py` prêt à l’emploi**, de sorte que l’agent IA ou ton développeur ait **un microservice Python complètement fonctionnel dès le départ**, prêt à recevoir les photos et retourner les mensurations.

Veux‑tu que je fasse ça ?
