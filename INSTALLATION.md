# Microservice SMPL - Guide d'installation et utilisation

## Installation

### 1. Cloner le projet et installer les dépendances

```bash
cd smpl-microservice
pip install -r requirements.txt
```

### 2. Télécharger les modèles pré-entraînés

#### Modèle SMPL
- Visiter: https://smpl.is.tue.mpg.de/
- Télécharger `SMPL_python_v.1.1.0.zip`
- Extraire et placer `SMPL_NEUTRAL.pkl` dans le dossier `models/`

#### Modèle HMR/SPIN
- Cloner le repo: https://github.com/nkolot/3DHumanPose
- Télécharger les checkpoints pré-entraînés
- Placer dans `models/`

## Lancer le serveur

```bash
python app.py
```

Le serveur démarre sur `http://localhost:5000`

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Estimation des mensurations (Principal)
```bash
POST /estimate

Body JSON:
{
  "photo_url": "https://...",
  "measures_table": ["tour_poitrine", "taille", "hanche"]
}
```

Réponse:
```json
{
  "measurements": {
    "tour_poitrine": 92.5,
    "taille": 70.2,
    "hanche": 98.1
  },
  "metadata": {
    "image_shape": [1080, 720],
    "num_keypoints": 33,
    "mesh_vertices": 6890
  }
}
```

### 3. Estimation en batch
```bash
POST /estimate/batch

Body JSON:
{
  "images": [
    {"photo_url": "...", "measures_table": [...]},
    ...
  ]
}
```

### 4. Référence des mensurations
```bash
GET /measurements/reference
```

### 5. Statut des modèles
```bash
GET /models/status
```

## Mensurations disponibles

- `tour_poitrine` / `chest_circumference`
- `taille` / `waist`
- `hanche` / `hip`
- `longueur_bras` / `arm_length`
- `longueur_jambe` / `leg_length`
- `largeur_epaules` / `shoulder_width`

## Tests

```bash
# Test simple
python test_api.py

# Test avec image locale
python test_api.py /chemin/vers/image.jpg
```

## Structure du projet

```
smpl-microservice/
├─ app.py                    # Application Flask principale
├─ smpl_engine.py           # Moteur SMPL + HMR/SPIN
├─ requirements.txt         # Dépendances Python
├─ test_api.py             # Tests API
├─ utils/
│  ├─ __init__.py
│  ├─ pose_estimation.py   # Détection de pose (MediaPipe)
│  └─ mesh_utils.py        # Calcul des mensurations
├─ models/                 # Modèles pré-entraînés
└─ README.md
```

## Configuration

Modifier les variables d'environnement ou `app.py`:
- `PORT`: Port du serveur (défaut: 5000)
- `DEVICE`: CPU/GPU (auto-détection par défaut)
- `MODEL_DIR`: Chemin des modèles

## Performance

- **GPU recommandé**: ~200ms par image
- **CPU**: ~1-2s par image

## Intégration Laravel

Exemple d'intégration backend:

```php
$client = new \GuzzleHttp\Client();
$response = $client->post('http://localhost:5000/estimate', [
    'json' => [
        'photo_url' => $imageUrl,
        'measures_table' => ['tour_poitrine', 'taille', 'hanche']
    ]
]);

$measurements = json_decode($response->getBody(), true)['measurements'];
```

## Dépannage

### Erreur: "Aucune personne détectée"
- Assurez-vous que l'image contient une personne claire
- Essayez une image avec meilleur éclairage

### Erreur: "Modèle non trouvé"
- Vérifiez que les modèles sont bien téléchargés dans `models/`
- Les chemins doivent correspondre à ceux configurés

### GPU non détecté
- Vérifier CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Vérifier PyTorch GPU: consulter https://pytorch.org

## Licence

Ce projet utilise les modèles suivants sous leurs licenses respectives:
- SMPL: Licence Max Planck Institute
- MediaPipe: Apache 2.0

## Support

Pour les issues ou questions, ouvrir une issue dans le repo.
