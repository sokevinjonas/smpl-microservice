# 📘 Microservice SMPL - Guide Complet d'Utilisation

## 📋 Table des matières
1. [Installation](#installation)
2. [Démarrage](#démarrage)
3. [Endpoints API](#endpoints-api)
4. [Mensurations disponibles](#mensurations-disponibles)
5. [Exemples d'utilisation](#exemples-dutilisation)
6. [Intégration Laravel](#intégration-laravel)
7. [Tests](#tests)

---

## 🚀 Installation

### Étape 1: Créer un environnement virtuel Python

```bash
# Se placer dans le répertoire du projet
cd ~/Bureau/SassApp/smpl-microservice

# Créer l'environnement virtuel
python3 -m venv venv

# Activer l'environnement (Linux/Mac)
source venv/bin/activate

# OU sur Windows:
# venv\Scripts\activate
```

### Étape 2: Installer les dépendances

```bash
# Vérifier que le venv est bien activé (vous verrez (venv) au début de votre terminal)
pip install --upgrade pip

# Installer toutes les dépendances
pip install -r requirements.txt
```

### Étape 3: Télécharger les modèles pré-entraînés (Optionnel pour les tests)

Pour l'utilisation complète, téléchargez les modèles SMPL:
- https://smpl.is.tue.mpg.de/
- Placer `SMPL_NEUTRAL.pkl` dans le dossier `models/`

---

## ▶️ Démarrage

### Lancer le serveur (avec venv activé)

```bash
# S'assurer que venv est activé
source venv/bin/activate

# Lancer l'application Flask
python app.py
```

**Résultat attendu:**
```
 * Serving Flask app 'app'
 * Debug mode: off
 * WARNING: This is a development server...
 * Running on http://0.0.0.0:5000
```

Le serveur sera accessible sur: **http://localhost:5000**

### Désactiver l'environnement (quand vous avez fini)

```bash
deactivate
```

---

## 📡 Endpoints API

### 1️⃣ Health Check
Vérifier que le serveur fonctionne

```http
GET http://localhost:5000/health
```

**Réponse (200):**
```json
{
  "status": "ok",
  "message": "Microservice SMPL est opérationnel"
}
```

---

### 2️⃣ Estimation des Mensurations ⭐ (PRINCIPAL)
Estimer les mensurations corporelles à partir d'une photo

```http
POST http://localhost:5000/estimate
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "photo_url": "https://example.com/photo.jpg",
  "measures_table": ["tour_poitrine", "taille", "hanche", "longueur_bras"]
}
```

**Paramètres:**
| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `photo_url` | string | Oui* | URL de l'image à analyser |
| `photo_path` | string | Oui* | OU chemin local de l'image |
| `measures_table` | array | Oui | Liste des mensurations à calculer |

*Au moins l'un des deux est requis

**Réponse réussie (200):**
```json
{
  "measurements": {
    "tour_poitrine": 92.5,
    "taille": 70.2,
    "hanche": 98.1,
    "longueur_bras": 62.4
  },
  "metadata": {
    "image_shape": [1080, 720],
    "num_keypoints": 33,
    "mesh_vertices": 6890,
    "validation_errors": []
  }
}
```

**Erreur - Aucune personne détectée (400):**
```json
{
  "error": "Aucune personne détectée dans l'image",
  "code": "NO_PERSON_DETECTED"
}
```

**Erreur - Paramètre manquant (400):**
```json
{
  "error": "photo_url ou photo_path requis"
}
```

---

### 3️⃣ Traitement en Batch
Traiter plusieurs images en une seule requête

```http
POST http://localhost:5000/estimate/batch
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "images": [
    {
      "photo_url": "https://example.com/photo1.jpg",
      "measures_table": ["tour_poitrine", "taille"]
    },
    {
      "photo_url": "https://example.com/photo2.jpg",
      "measures_table": ["hanche", "longueur_bras"]
    }
  ]
}
```

**Réponse (200):**
```json
{
  "results": [
    {
      "index": 0,
      "status": "success",
      "data": {
        "measurements": {...},
        "metadata": {...}
      }
    },
    {
      "index": 1,
      "status": "success",
      "data": {
        "measurements": {...},
        "metadata": {...}
      }
    }
  ]
}
```

---

### 4️⃣ Référence des Mensurations
Voir toutes les mensurations disponibles

```http
GET http://localhost:5000/measurements/reference
```

**Réponse (200):**
```json
{
  "available_measurements": [
    "tour_poitrine",
    "chest_circumference",
    "poitrine",
    "taille",
    "waist",
    "hanche",
    "hip",
    "hanches",
    "longueur_bras",
    "arm_length",
    "longueur_jambe",
    "leg_length",
    "largeur_epaules",
    "shoulder_width"
  ],
  "body_parts": [
    "chest",
    "waist",
    "hip",
    "arm_length",
    "leg_length",
    "shoulder_width"
  ],
  "example_request": {
    "photo_url": "https://...",
    "measures_table": ["tour_poitrine", "taille", "hanche"]
  }
}
```

---

### 5️⃣ Statut des Modèles
Vérifier l'état des modèles chargés

```http
GET http://localhost:5000/models/status
```

**Réponse (200):**
```json
{
  "pose_estimator": "loaded",
  "smpl_engine": "loaded",
  "device": "cuda"
}
```

*(device peut être "cuda" pour GPU ou "cpu" pour CPU)*

---

## 📏 Mensurations disponibles

### Noms acceptés (Français)

| Code | Description | Équivalent anglais |
|------|-------------|-------------------|
| `tour_poitrine` | Tour de poitrine | chest_circumference |
| `poitrine` | Tour de poitrine (court) | - |
| `taille` | Taille/Ceinture | waist |
| `hanche` | Tour de hanches | hip |
| `hanches` | Tour de hanches (pluriel) | - |
| `longueur_bras` | Longueur du bras | arm_length |
| `longueur_jambe` | Longueur de la jambe | leg_length |
| `largeur_epaules` | Largeur des épaules | shoulder_width |

### Noms acceptés (Anglais)

| Code | Description |
|------|-------------|
| `chest_circumference` | Tour de poitrine |
| `waist` | Taille |
| `hip` | Hanches |
| `arm_length` | Longueur du bras |
| `leg_length` | Longueur de la jambe |
| `shoulder_width` | Largeur des épaules |

**Les mensurations retournées sont en millimètres (mm)**

---

## 💡 Exemples d'utilisation

### Exemple 1: cURL - Requête simple

```bash
curl -X POST http://localhost:5000/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "photo_url": "https://example.com/photo.jpg",
    "measures_table": ["tour_poitrine", "taille", "hanche"]
  }'
```

### Exemple 2: cURL - Avec chemin local

```bash
curl -X POST http://localhost:5000/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "photo_path": "/tmp/mon_image.jpg",
    "measures_table": ["tour_poitrine", "taille"]
  }'
```

### Exemple 3: Python - Requête simple

```python
import requests
import json

url = 'http://localhost:5000/estimate'
payload = {
    'photo_url': 'https://example.com/photo.jpg',
    'measures_table': ['tour_poitrine', 'taille', 'hanche', 'longueur_bras']
}

response = requests.post(url, json=payload)
data = response.json()

if response.status_code == 200:
    measurements = data['measurements']
    print(f"Tour de poitrine: {measurements['tour_poitrine']} mm")
    print(f"Taille: {measurements['taille']} mm")
else:
    print(f"Erreur: {data['error']}")
```

### Exemple 4: Python - Traitement batch

```python
import requests

url = 'http://localhost:5000/estimate/batch'
payload = {
    'images': [
        {
            'photo_url': 'https://example.com/photo1.jpg',
            'measures_table': ['tour_poitrine']
        },
        {
            'photo_url': 'https://example.com/photo2.jpg',
            'measures_table': ['taille', 'hanche']
        }
    ]
}

response = requests.post(url, json=payload)
results = response.json()['results']

for result in results:
    if result['status'] == 'success':
        print(f"Image {result['index']}: {result['data']['measurements']}")
    else:
        print(f"Image {result['index']}: Erreur")
```

### Exemple 5: JavaScript/Fetch

```javascript
const estimate = async () => {
  const response = await fetch('http://localhost:5000/estimate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      photo_url: 'https://example.com/photo.jpg',
      measures_table: ['tour_poitrine', 'taille', 'hanche']
    })
  });

  const data = await response.json();
  
  if (response.ok) {
    console.log('Mensurations:', data.measurements);
  } else {
    console.error('Erreur:', data.error);
  }
};

estimate();
```

---

## 🔗 Intégration Laravel

### Installation de la dépendance

```bash
composer require guzzlehttp/guzzle
```

### Exemple 1: Service simple

```php
<?php

namespace App\Services;

use GuzzleHttp\Client;
use Exception;

class SmplEstimationService
{
    private $client;
    private $baseUrl = 'http://localhost:5000';

    public function __construct()
    {
        $this->client = new Client();
    }

    /**
     * Estimer les mensurations à partir d'une URL d'image
     */
    public function estimateFromUrl(string $imageUrl, array $measures): array
    {
        try {
            $response = $this->client->post("{$this->baseUrl}/estimate", [
                'json' => [
                    'photo_url' => $imageUrl,
                    'measures_table' => $measures
                ],
                'timeout' => 30
            ]);

            $data = json_decode($response->getBody(), true);
            return $data['measurements'];

        } catch (Exception $e) {
            throw new Exception("Erreur SMPL: " . $e->getMessage());
        }
    }

    /**
     * Estimer les mensurations à partir d'un chemin local
     */
    public function estimateFromPath(string $imagePath, array $measures): array
    {
        try {
            $response = $this->client->post("{$this->baseUrl}/estimate", [
                'json' => [
                    'photo_path' => $imagePath,
                    'measures_table' => $measures
                ],
                'timeout' => 30
            ]);

            $data = json_decode($response->getBody(), true);
            return $data['measurements'];

        } catch (Exception $e) {
            throw new Exception("Erreur SMPL: " . $e->getMessage());
        }
    }

    /**
     * Traitement batch
     */
    public function estimateBatch(array $images): array
    {
        try {
            $response = $this->client->post("{$this->baseUrl}/estimate/batch", [
                'json' => ['images' => $images],
                'timeout' => 60
            ]);

            $data = json_decode($response->getBody(), true);
            return $data['results'];

        } catch (Exception $e) {
            throw new Exception("Erreur SMPL Batch: " . $e->getMessage());
        }
    }

    /**
     * Vérifier la santé du microservice
     */
    public function isHealthy(): bool
    {
        try {
            $response = $this->client->get("{$this->baseUrl}/health", [
                'timeout' => 5
            ]);
            return $response->getStatusCode() === 200;
        } catch (Exception $e) {
            return false;
        }
    }
}
```

### Exemple 2: Utilisation dans un Controller

```php
<?php

namespace App\Http\Controllers;

use App\Services\SmplEstimationService;
use App\Models\Product;
use Illuminate\Http\Request;

class ProductController extends Controller
{
    private $smplService;

    public function __construct(SmplEstimationService $smplService)
    {
        $this->smplService = $smplService;
    }

    /**
     * Estimer les mensurations d'une robe
     */
    public function estimateMeasurements(Request $request, Product $product)
    {
        try {
            // Vérifier la santé du microservice
            if (!$this->smplService->isHealthy()) {
                return response()->json([
                    'error' => 'Microservice SMPL indisponible'
                ], 503);
            }

            // Estimer les mensurations
            $measurements = $this->smplService->estimateFromUrl(
                $product->image_url,
                ['tour_poitrine', 'taille', 'hanche', 'longueur_bras']
            );

            // Sauvegarder les résultats
            $product->update([
                'estimated_chest' => $measurements['tour_poitrine'],
                'estimated_waist' => $measurements['taille'],
                'estimated_hip' => $measurements['hanche'],
                'estimated_arm_length' => $measurements['longueur_bras']
            ]);

            return response()->json([
                'success' => true,
                'measurements' => $measurements
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Estimer en batch (plusieurs produits)
     */
    public function estimateBatchMeasurements(Request $request)
    {
        try {
            // Préparer les images
            $images = [];
            foreach ($request->input('product_ids', []) as $productId) {
                $product = Product::find($productId);
                $images[] = [
                    'photo_url' => $product->image_url,
                    'measures_table' => ['tour_poitrine', 'taille', 'hanche']
                ];
            }

            // Traiter en batch
            $results = $this->smplService->estimateBatch($images);

            // Sauvegarder les résultats
            foreach ($results as $index => $result) {
                if ($result['status'] === 'success') {
                    $measurements = $result['data']['measurements'];
                    // Mettre à jour le produit...
                }
            }

            return response()->json([
                'success' => true,
                'results' => $results
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'error' => $e->getMessage()
            ], 500);
        }
    }
}
```

### Exemple 3: Route

```php
// routes/api.php

Route::post('/products/{product}/estimate-measurements', 
    [ProductController::class, 'estimateMeasurements']
);

Route::post('/products/estimate-batch', 
    [ProductController::class, 'estimateBatchMeasurements']
);
```

### Exemple 4: Modèle Product

```php
<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Product extends Model
{
    protected $fillable = [
        'name',
        'image_url',
        'estimated_chest',
        'estimated_waist',
        'estimated_hip',
        'estimated_arm_length',
        'estimated_leg_length',
        'estimated_shoulder_width'
    ];

    protected $casts = [
        'estimated_chest' => 'float',
        'estimated_waist' => 'float',
        'estimated_hip' => 'float',
        'estimated_arm_length' => 'float',
        'estimated_leg_length' => 'float',
        'estimated_shoulder_width' => 'float'
    ];
}
```

---

## 🧪 Tests

### Test simple avec Python

```bash
# Activer venv
source venv/bin/activate

# Lancer les tests
python test_api.py
```

### Test avec image locale

```bash
python test_api.py /chemin/vers/image.jpg
```

### Test avec cURL

```bash
# Health check
curl http://localhost:5000/health

# Estimation simple
curl -X POST http://localhost:5000/estimate \
  -H "Content-Type: application/json" \
  -d '{"photo_url":"https://example.com/photo.jpg","measures_table":["tour_poitrine"]}'

# Référence
curl http://localhost:5000/measurements/reference
```

---

## ⚙️ Configuration avancée

### Variables d'environnement (optionnel)

Créer un fichier `.env`:

```bash
FLASK_ENV=production
FLASK_PORT=5000
MODEL_DIR=./models
DEVICE=cuda  # ou cpu
MAX_IMAGE_SIZE=5242880  # 5MB
```

### Démarrage avec gunicorn (Production)

```bash
# Installer gunicorn
pip install gunicorn

# Lancer avec 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Avec Docker

```bash
# Construire l'image
docker build -t smpl-microservice .

# Lancer le conteneur
docker run -p 5000:5000 smpl-microservice

# Ou avec docker-compose
docker-compose up
```

---

## ⚠️ Points importants

✅ **Venv obligatoire** - Utilisez toujours l'environnement virtuel  
✅ **Format des mesures** - Les valeurs sont en **millimètres (mm)**  
✅ **Détection** - Une personne claire et visible est nécessaire  
✅ **GPU optionnel** - Plus rapide avec GPU (~200ms) qu'avec CPU (~1-2s)  
✅ **CORS activé** - Accessible depuis n'importe quel domaine  
✅ **Timeout** - Prévoir 30 secondes de timeout côté client  

---

## 🆘 Dépannage

### Erreur: "No module named 'flask'"
```bash
# Vérifier que venv est activé
source venv/bin/activate

# Réinstaller les dépendances
pip install -r requirements.txt
```

### Erreur: "Aucune personne détectée"
- Assurez-vous que l'image contient une personne clairement visible
- Essayez avec une image mieux éclairée
- Vérifiez que l'image n'est pas trop petite ou floue

### GPU non détecté
```bash
# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Réinstaller PyTorch avec CUDA
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Le serveur démarre mais les requests échouent
```bash
# Vérifier que le serveur écoute
netstat -tlnp | grep 5000

# Vérifier les logs du serveur pour les erreurs
```

---

## 📞 Support

Pour toute question ou problème, consultez:
- README.md - Documentation générale
- INSTALLATION.md - Guide d'installation détaillé
- Les logs du serveur Flask pour déboguer

Bonne utilisation! 🚀
