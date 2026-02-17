# 📚 Guide d'Intégration - Microservice SMPL pour Développeurs Backend

**Adressé aux:** Développeurs Backend Laravel/PHP  
**Date:** 17 février 2026  
**Version:** 1.0  

---

## 📖 Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Endpoints disponibles](#endpoints-disponibles)
4. [Implémentation en Laravel](#implémentation-en-laravel)
5. [Gestion des erreurs](#gestion-des-erreurs)
6. [Intégration base de données](#intégration-base-de-données)
7. [Cas d'usage réels](#cas-dusage-réels)
8. [Bonnes pratiques](#bonnes-pratiques)
9. [FAQ](#faq)

---

## 🎯 Vue d'ensemble

### Qu'est-ce que le microservice SMPL?

Le microservice SMPL est un **service indépendant** qui analyse des photos de personnes et extrait automatiquement leurs **mensurations corporelles** (tour de poitrine, taille, hanches, etc.).

### Pourquoi l'utiliser?

✅ **Automatisation** - Pas besoin de mesurer manuellement  
✅ **Précision** - Utilise l'IA (pose detection + reconstruction 3D)  
✅ **Scalabilité** - Service séparé, ne ralentit pas votre backend  
✅ **Flexibilité** - Choisissez quelles mesures récupérer  

### Flux de travail typique

```
Utilisateur upload une photo
          ↓
Backend envoie la photo au microservice SMPL
          ↓
SMPL retourne les mensurations (JSON)
          ↓
Backend sauvegarde dans la BD
          ↓
Utilisateur voit ses mensurations
```

---

## 🏗️ Architecture

### Composants

```
┌─────────────────────────────────────────┐
│         Votre Backend Laravel           │
│    (Reçoit requêtes des utilisateurs)   │
└──────────────────┬──────────────────────┘
                   │
                   │ HTTP POST /estimate
                   ↓
┌─────────────────────────────────────────┐
│    Microservice SMPL (Python/Flask)     │
│  (Détecte pose + génère mesh SMPL)      │
└──────────────────┬──────────────────────┘
                   │
                   │ Retourne JSON
                   ↓
┌─────────────────────────────────────────┐
│    Backend sauvegarde en BD             │
│    et retourne à l'utilisateur          │
└─────────────────────────────────────────┘
```

### Points importants

- ⏱️ **Timeout**: Prévoir 30-60 secondes (l'IA prend du temps)
- 🔄 **Asynchrone recommandé**: Si vous avez beaucoup de uploads
- 📍 **Localisation**: SMPL peut tourner sur le même serveur ou distant
- 🔌 **Reconnexion**: Gérer les cas où SMPL est indisponible

---

## 📡 Endpoints disponibles

### 1️⃣ Health Check
Vérifier que le microservice est actif

```http
GET http://localhost:5000/health
```

**Réponse réussie (200):**
```json
{
  "status": "ok",
  "message": "Microservice SMPL est opérationnel"
}
```

**Utilité:** Avant de faire une requête d'estimation, vérifiez que le service est actif

---

### 2️⃣ Estimation des Mensurations ⭐ (PRINCIPAL)

Analyser une photo et retourner les mensurations

```http
POST http://localhost:5000/estimate
Content-Type: application/json
```

#### Body (Paramètres)

```json
{
  "photo_url": "https://example.com/uploads/photo123.jpg",
  "measures_table": [
    "tour_poitrine",
    "taille",
    "hanche",
    "longueur_bras"
  ]
}
```

#### Paramètres détaillés

| Paramètre | Type | Obligatoire | Description |
|-----------|------|-------------|-------------|
| `photo_url` | string | Oui* | URL absolue de l'image à analyser |
| `photo_path` | string | Oui* | OU chemin local de l'image (si SMPL sur même serveur) |
| `measures_table` | array | Oui | Liste des mensurations à calculer |

*Au moins l'un des deux est requis (photo_url OU photo_path)

#### Mensurations disponibles

```
"tour_poitrine"      → Tour de poitrine (mm)
"taille"             → Taille/Ceinture (mm)
"hanche"             → Tour de hanches (mm)
"longueur_bras"      → Longueur du bras (mm)
"longueur_jambe"     → Longueur de la jambe (mm)
"largeur_epaules"    → Largeur des épaules (mm)
```

**Tous les noms acceptés:**
```
Poitrine: "tour_poitrine", "chest_circumference", "poitrine"
Taille:   "taille", "waist"
Hanches:  "hanche", "hip", "hanches"
Bras:     "longueur_bras", "arm_length"
Jambe:    "longueur_jambe", "leg_length"
Épaules:  "largeur_epaules", "shoulder_width"
```

#### Réponse réussie (200)

```json
{
  "measurements": {
    "tour_poitrine": 925.5,
    "taille": 702.3,
    "hanche": 981.7,
    "longueur_bras": 624.2
  },
  "metadata": {
    "image_shape": [1080, 720],
    "num_keypoints": 33,
    "mesh_vertices": 6890,
    "validation_errors": []
  }
}
```

**Explication des champs:**
- `measurements` → **Les résultats!** En millimètres
- `metadata.image_shape` → Dimensions de l'image (hauteur, largeur)
- `metadata.num_keypoints` → Points clés du corps détectés (33 pour MediaPipe)
- `metadata.mesh_vertices` → Vertices du mesh SMPL généré
- `metadata.validation_errors` → Avertissements (e.g., mensurations incohérentes)

#### Erreur - Aucune personne détectée (400)

```json
{
  "error": "Aucune personne détectée dans l'image",
  "code": "NO_PERSON_DETECTED"
}
```

**Quand ça arrive:** Image floue, trop sombre, personne trop petite, ou pas de personne du tout

#### Erreur - Paramètre manquant (400)

```json
{
  "error": "photo_url ou photo_path requis"
}
```

#### Erreur - Image invalide (400)

```json
{
  "error": "Image invalide"
}
```

#### Erreur interne (500)

```json
{
  "error": "Erreur serveur: [description]",
  "code": "INTERNAL_ERROR"
}
```

---

### 3️⃣ Traitement en Batch

Traiter plusieurs images en une seule requête (plus efficace)

```http
POST http://localhost:5000/estimate/batch
Content-Type: application/json
```

#### Body

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
    },
    {
      "photo_url": "https://example.com/photo3.jpg",
      "measures_table": ["tour_poitrine"]
    }
  ]
}
```

#### Réponse (200)

```json
{
  "results": [
    {
      "index": 0,
      "status": "success",
      "data": {
        "measurements": {
          "tour_poitrine": 925.5,
          "taille": 702.3
        },
        "metadata": {...}
      }
    },
    {
      "index": 1,
      "status": "success",
      "data": {
        "measurements": {
          "hanche": 981.7,
          "longueur_bras": 624.2
        },
        "metadata": {...}
      }
    },
    {
      "index": 2,
      "status": "error",
      "error": "Aucune personne détectée"
    }
  ]
}
```

**Avantages du batch:**
- ✅ Plus efficace que plusieurs requêtes individuelles
- ✅ Parfait pour les imports massifs
- ✅ Gère les erreurs par image

---

### 4️⃣ Référence des Mensurations

Voir toutes les mensurations disponibles et leurs codes

```http
GET http://localhost:5000/measurements/reference
```

#### Réponse (200)

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

**Utilité:** Afficher dans votre interface quelles mesures sont disponibles

---

### 5️⃣ Statut des Modèles

Vérifier l'état des modèles IA chargés

```http
GET http://localhost:5000/models/status
```

#### Réponse (200)

```json
{
  "pose_estimator": "loaded",
  "smpl_engine": "loaded",
  "device": "cuda"
}
```

**Interprétation:**
- `pose_estimator: "loaded"` → Détection de pose opérationnelle
- `smpl_engine: "loaded"` → Génération mesh SMPL opérationnelle
- `device: "cuda"` → GPU activé (rapide) | `"cpu"` → CPU (lent)

---

## 💻 Implémentation en Laravel

### Installation des dépendances

```bash
composer require guzzlehttp/guzzle
```

### 1️⃣ Service de communication avec SMPL

Créer le fichier `app/Services/SmplService.php`:

```php
<?php

namespace App\Services;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\RequestException;
use Exception;
use Log;

class SmplService
{
    private $client;
    private $baseUrl = 'http://localhost:5000'; // ou l'IP du serveur SMPL

    public function __construct()
    {
        $this->client = new Client([
            'timeout' => 60, // Important: 60 secondes pour le timeout
            'connect_timeout' => 10
        ]);
    }

    /**
     * Vérifier la santé du microservice
     */
    public function isHealthy(): bool
    {
        try {
            $response = $this->client->get("{$this->baseUrl}/health");
            return $response->getStatusCode() === 200;
        } catch (Exception $e) {
            Log::warning("SMPL Service indisponible: " . $e->getMessage());
            return false;
        }
    }

    /**
     * Estimer les mensurations à partir d'une URL d'image
     * 
     * @param string $imageUrl URL de l'image
     * @param array $measures Array des mensurations désirées
     * @return array Associative array avec les mensurations
     * @throws Exception
     */
    public function estimateFromUrl(string $imageUrl, array $measures = []): array
    {
        try {
            // Utiliser les mesures par défaut si aucune spécifiée
            if (empty($measures)) {
                $measures = [
                    'tour_poitrine',
                    'taille',
                    'hanche',
                    'longueur_bras'
                ];
            }

            $response = $this->client->post("{$this->baseUrl}/estimate", [
                'json' => [
                    'photo_url' => $imageUrl,
                    'measures_table' => $measures
                ]
            ]);

            $data = json_decode($response->getBody(), true);

            if ($response->getStatusCode() !== 200) {
                throw new Exception("SMPL Error: " . ($data['error'] ?? 'Unknown error'));
            }

            return $data['measurements'] ?? [];

        } catch (RequestException $e) {
            Log::error("SMPL Request failed: " . $e->getMessage());
            throw new Exception("Impossible de contacter le microservice SMPL");
        } catch (Exception $e) {
            Log::error("SMPL Error: " . $e->getMessage());
            throw $e;
        }
    }

    /**
     * Estimer à partir d'un chemin local (si SMPL sur même serveur)
     */
    public function estimateFromPath(string $imagePath, array $measures = []): array
    {
        try {
            if (empty($measures)) {
                $measures = ['tour_poitrine', 'taille', 'hanche'];
            }

            $response = $this->client->post("{$this->baseUrl}/estimate", [
                'json' => [
                    'photo_path' => $imagePath,
                    'measures_table' => $measures
                ]
            ]);

            $data = json_decode($response->getBody(), true);
            return $data['measurements'] ?? [];

        } catch (Exception $e) {
            Log::error("SMPL Error: " . $e->getMessage());
            throw new Exception("Erreur lors de l'estimation des mensurations");
        }
    }

    /**
     * Traitement batch - plusieurs images à la fois
     * 
     * @param array $images Array of ['photo_url' => '...', 'measures_table' => [...]]
     * @return array Results array
     */
    public function estimateBatch(array $images): array
    {
        try {
            $response = $this->client->post("{$this->baseUrl}/estimate/batch", [
                'json' => ['images' => $images]
            ]);

            $data = json_decode($response->getBody(), true);
            return $data['results'] ?? [];

        } catch (Exception $e) {
            Log::error("SMPL Batch Error: " . $e->getMessage());
            throw new Exception("Erreur lors du traitement batch");
        }
    }

    /**
     * Récupérer les mensurations de référence
     */
    public function getAvailableMeasurements(): array
    {
        try {
            $response = $this->client->get("{$this->baseUrl}/measurements/reference");
            $data = json_decode($response->getBody(), true);
            return $data['available_measurements'] ?? [];

        } catch (Exception $e) {
            Log::error("SMPL Reference Error: " . $e->getMessage());
            return [
                'tour_poitrine', 'taille', 'hanche', 
                'longueur_bras', 'longueur_jambe', 'largeur_epaules'
            ];
        }
    }

    /**
     * Obtenir le statut du microservice
     */
    public function getStatus(): array
    {
        try {
            $response = $this->client->get("{$this->baseUrl}/models/status");
            return json_decode($response->getBody(), true);

        } catch (Exception $e) {
            return [
                'status' => 'unavailable',
                'error' => $e->getMessage()
            ];
        }
    }
}
```

### 2️⃣ Modèle Eloquent pour stocker les mesures

Créer la migration:

```bash
php artisan make:migration create_user_measurements_table
```

Fichier `database/migrations/xxxx_create_user_measurements_table.php`:

```php
<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('user_measurements', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            
            // Mensurations en mm
            $table->float('chest_circumference')->nullable(); // tour_poitrine
            $table->float('waist')->nullable();                // taille
            $table->float('hip_circumference')->nullable();    // hanche
            $table->float('arm_length')->nullable();           // longueur_bras
            $table->float('leg_length')->nullable();           // longueur_jambe
            $table->float('shoulder_width')->nullable();       // largeur_epaules
            
            // Métadonnées
            $table->string('image_url')->nullable();
            $table->json('smpl_metadata')->nullable(); // Stocker les metadata SMPL
            $table->boolean('is_verified')->default(false); // Manuelle ou SMPL?
            
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('user_measurements');
    }
};
```

Exécuter:

```bash
php artisan migrate
```

### 3️⃣ Modèle UserMeasurement

Créer `app/Models/UserMeasurement.php`:

```php
<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class UserMeasurement extends Model
{
    protected $fillable = [
        'user_id',
        'chest_circumference',
        'waist',
        'hip_circumference',
        'arm_length',
        'leg_length',
        'shoulder_width',
        'image_url',
        'smpl_metadata',
        'is_verified'
    ];

    protected $casts = [
        'chest_circumference' => 'float',
        'waist' => 'float',
        'hip_circumference' => 'float',
        'arm_length' => 'float',
        'leg_length' => 'float',
        'shoulder_width' => 'float',
        'smpl_metadata' => 'array',
        'is_verified' => 'boolean'
    ];

    public function user()
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Mapper les résultats SMPL vers la BD
     */
    public static function createFromSmplResponse(int $userId, array $smplData, string $imageUrl): self
    {
        $measurements = $smplData['measurements'] ?? [];
        $metadata = $smplData['metadata'] ?? [];

        return self::create([
            'user_id' => $userId,
            'chest_circumference' => $measurements['tour_poitrine'] ?? null,
            'waist' => $measurements['taille'] ?? null,
            'hip_circumference' => $measurements['hanche'] ?? null,
            'arm_length' => $measurements['longueur_bras'] ?? null,
            'leg_length' => $measurements['longueur_jambe'] ?? null,
            'shoulder_width' => $measurements['largeur_epaules'] ?? null,
            'image_url' => $imageUrl,
            'smpl_metadata' => $metadata,
            'is_verified' => true // SMPL détecte automatiquement
        ]);
    }

    /**
     * Obtenir les mensurations en cm (plus lisible)
     */
    public function getMeasurementsInCm(): array
    {
        return [
            'chest' => round($this->chest_circumference / 10, 1),
            'waist' => round($this->waist / 10, 1),
            'hip' => round($this->hip_circumference / 10, 1),
            'arm' => round($this->arm_length / 10, 1),
            'leg' => round($this->leg_length / 10, 1),
            'shoulder' => round($this->shoulder_width / 10, 1),
        ];
    }
}
```

### 4️⃣ Controller pour gérer les uploads

Créer `app/Http/Controllers/MeasurementController.php`:

```php
<?php

namespace App\Http\Controllers;

use App\Models\UserMeasurement;
use App\Services\SmplService;
use Illuminate\Http\Request;
use Log;

class MeasurementController extends Controller
{
    private $smplService;

    public function __construct(SmplService $smplService)
    {
        $this->smplService = $smplService;
        $this->middleware('auth'); // Utilisateur connecté requis
    }

    /**
     * Estimer les mensurations à partir d'un upload
     */
    public function estimate(Request $request)
    {
        $request->validate([
            'photo' => 'required|image|mimes:jpeg,png,jpg,gif|max:10240', // 10MB max
        ]);

        try {
            // 1️⃣ Vérifier que SMPL est disponible
            if (!$this->smplService->isHealthy()) {
                return response()->json([
                    'error' => 'Service de mesure temporairement indisponible',
                    'code' => 'SERVICE_UNAVAILABLE'
                ], 503);
            }

            // 2️⃣ Sauvegarder l'image uploadée
            $path = $request->file('photo')->store('measurements', 'public');
            $imageUrl = asset('storage/' . $path);

            Log::info("Photo uploadée: {$imageUrl}");

            // 3️⃣ Appeler SMPL
            $smplResponse = $this->smplService->estimateFromUrl($imageUrl, [
                'tour_poitrine',
                'taille',
                'hanche',
                'longueur_bras',
                'longueur_jambe',
                'largeur_epaules'
            ]);

            // 4️⃣ Sauvegarder en BD
            $measurement = UserMeasurement::createFromSmplResponse(
                auth()->id(),
                $smplResponse,
                $imageUrl
            );

            Log::info("Mesures créées pour user " . auth()->id());

            // 5️⃣ Retourner les résultats
            return response()->json([
                'success' => true,
                'measurements' => $measurement->getMeasurementsInCm(), // En cm pour l'utilisateur
                'measurement_id' => $measurement->id,
                'message' => 'Mensurations calculées avec succès'
            ]);

        } catch (\Exception $e) {
            Log::error("Erreur estimation: " . $e->getMessage());
            
            return response()->json([
                'error' => $e->getMessage(),
                'code' => 'ESTIMATION_ERROR'
            ], 500);
        }
    }

    /**
     * Récupérer les mensurations de l'utilisateur
     */
    public function getUserMeasurements()
    {
        $measurements = auth()->user()->measurements()
            ->latest()
            ->first();

        if (!$measurements) {
            return response()->json([
                'error' => 'Aucune mesure trouvée',
                'code' => 'NOT_FOUND'
            ], 404);
        }

        return response()->json([
            'measurements' => $measurements->getMeasurementsInCm(),
            'taken_at' => $measurements->created_at,
            'image_url' => $measurements->image_url
        ]);
    }

    /**
     * Historique des mesures
     */
    public function getMeasurementsHistory()
    {
        $history = auth()->user()->measurements()
            ->orderBy('created_at', 'desc')
            ->paginate(10);

        return response()->json([
            'total' => $history->total(),
            'measurements' => $history->map(fn ($m) => [
                'id' => $m->id,
                'measurements' => $m->getMeasurementsInCm(),
                'date' => $m->created_at->format('Y-m-d H:i:s'),
                'image_url' => $m->image_url
            ])
        ]);
    }

    /**
     * Traitement batch (import massif)
     */
    public function estimateBatch(Request $request)
    {
        $request->validate([
            'images' => 'required|array|min:1|max:50',
            'images.*.photo' => 'required|string|url'
        ]);

        try {
            // Construire les requests pour SMPL
            $batchImages = array_map(fn ($img) => [
                'photo_url' => $img['photo'],
                'measures_table' => ['tour_poitrine', 'taille', 'hanche']
            ], $request->input('images'));

            // Appeler SMPL en batch
            $results = $this->smplService->estimateBatch($batchImages);

            // Traiter les résultats
            $successCount = 0;
            foreach ($results as $index => $result) {
                if ($result['status'] === 'success') {
                    UserMeasurement::createFromSmplResponse(
                        auth()->id(),
                        $result['data'],
                        $request->input('images.' . $index . '.photo')
                    );
                    $successCount++;
                }
            }

            return response()->json([
                'success' => true,
                'processed' => count($results),
                'successful' => $successCount,
                'failed' => count($results) - $successCount
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Vérifier la santé du microservice
     */
    public function checkSmplStatus()
    {
        return response()->json([
            'healthy' => $this->smplService->isHealthy(),
            'status' => $this->smplService->getStatus()
        ]);
    }
}
```

### 5️⃣ Routes

Ajouter à `routes/api.php`:

```php
<?php

use App\Http\Controllers\MeasurementController;
use Illuminate\Support\Facades\Route;

Route::middleware('auth:sanctum')->group(function () {
    // Estimer les mensurations
    Route::post('/measurements/estimate', [MeasurementController::class, 'estimate']);
    
    // Récupérer les dernières mesures
    Route::get('/measurements/current', [MeasurementController::class, 'getUserMeasurements']);
    
    // Historique
    Route::get('/measurements/history', [MeasurementController::class, 'getMeasurementsHistory']);
    
    // Batch
    Route::post('/measurements/batch', [MeasurementController::class, 'estimateBatch']);
});

// Public - Vérifier la santé (sans auth)
Route::get('/measurements/status', [MeasurementController::class, 'checkSmplStatus']);
```

### 6️⃣ Frontend - Upload exemple

```html
<form id="measurementForm" enctype="multipart/form-data">
    <input type="file" id="photoInput" accept="image/*" required>
    <button type="submit">Mesurer</button>
    <div id="result"></div>
</form>

<script>
document.getElementById('measurementForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('photo', document.getElementById('photoInput').files[0]);
    
    try {
        const response = await fetch('/api/measurements/estimate', {
            method: 'POST',
            body: formData,
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('result').innerHTML = `
                <h3>✓ Mensurations:</h3>
                <p>Tour de poitrine: ${data.measurements.chest} cm</p>
                <p>Taille: ${data.measurements.waist} cm</p>
                <p>Hanches: ${data.measurements.hip} cm</p>
            `;
        } else {
            alert(`Erreur: ${data.error}`);
        }
    } catch (error) {
        alert('Erreur réseau: ' + error.message);
    }
});
</script>
```

---

## ⚠️ Gestion des erreurs

### Scénarios d'erreur possibles

#### 1. Image invalide
```php
if (/* image floue ou mauvais format */) {
    // SMPL retourne 400
    // → Redemander une meilleure photo
}
```

#### 2. Aucune personne détectée
```php
// SMPL retourne:
{
    "error": "Aucune personne détectée",
    "code": "NO_PERSON_DETECTED"
}
// Action: Redemander une photo plus claire
```

#### 3. SMPL indisponible
```php
// Service down
// → Code 503
// → Afficher message: "Service temporairement indisponible"
```

#### 4. Timeout
```php
// Prise plus de 60 secondes
// → Guzzle throw RequestException
// → Retry logic?
```

### Code d'erreur à gérer

| Code | HTTP | Action |
|------|------|--------|
| `NO_PERSON_DETECTED` | 400 | Redemander une photo |
| `SERVICE_UNAVAILABLE` | 503 | Réessayer plus tard |
| `INTERNAL_ERROR` | 500 | Log + retry |
| `VALIDATION_ERROR` | 400 | Vérifier paramètres |

---

## 🗄️ Intégration base de données

### Structure recommandée

```sql
users
├── id
├── name
├── email
└── ...

user_measurements
├── id
├── user_id (FK)
├── chest_circumference (mm)
├── waist (mm)
├── hip_circumference (mm)
├── arm_length (mm)
├── leg_length (mm)
├── shoulder_width (mm)
├── image_url
├── smpl_metadata (JSON)
├── is_verified
├── created_at
└── updated_at
```

### Queries utiles

```php
// Dernière mesure de l'utilisateur
$latest = auth()->user()->measurements()->latest()->first();

// Historique complet
$history = auth()->user()->measurements()->get();

// Mesures non vérifiées (SMPL automatique)
$auto = UserMeasurement::where('is_verified', false)->get();

// Évolution dans le temps
$progression = auth()->user()->measurements()
    ->orderBy('created_at')
    ->select(['created_at', 'chest_circumference', 'waist'])
    ->get();
```

---

## 📋 Cas d'usage réels

### 1. E-commerce vêtements
```
Utilisateur upload photo
    ↓
SMPL détecte ses mensurations
    ↓
Système recommande la taille idéale
    ↓
Moins de retours!
```

### 2. Application fitness
```
Utilisateur prend photo chaque mois
    ↓
Mensurations auto-mesurées
    ↓
Graphe de progression
    ↓
Motivation!
```

### 3. Essayage virtuel
```
Mensurations SMPL → Model 3D utilisateur
    ↓
Essayage de vêtements en AR
```

### 4. Santé/Médecine
```
Suivi de patients
    ↓
Mesures objectives
    ↓
Évolution documentée
```

---

## ✨ Bonnes pratiques

### 1️⃣ Vérifier la santé avant chaque requête

```php
if (!$this->smplService->isHealthy()) {
    // Utiliser une fallback ou attendre
}
```

### 2️⃣ Timeout adapté

```php
new Client([
    'timeout' => 60, // SMPL prend du temps!
    'connect_timeout' => 10
])
```

### 3️⃣ Logging complet

```php
Log::info("Estimation lancée pour user: " . auth()->id());
Log::error("SMPL Error: " . $e->getMessage());
```

### 4️⃣ Cache les résultats

```php
$measurements = Cache::remember(
    "measurements:user:{$userId}",
    86400, // 24 heures
    fn () => UserMeasurement::where('user_id', $userId)->latest()->first()
);
```

### 5️⃣ Queue pour les uploads massifs

```php
// dispatch job au lieu d'attendre
dispatch(new ProcessMeasurements($imageUrl));
```

### 6️⃣ Validation côté client

```javascript
// Vérifier taille fichier avant upload
if (file.size > 10 * 1024 * 1024) {
    alert('Image trop volumineuse (max 10MB)');
    return;
}
```

### 7️⃣ Documenter pour l'utilisateur

```php
// Afficher ce qu'on mesure
"Pour les meilleurs résultats:"
- "Photo claire, de face"
- "Bonne lumière"
- "Personne entière visible"
```

---

## ❓ FAQ

### Q: Quelle taille d'image?
**R:** 1-10 MB, formats JPEG/PNG/GIF. Optimiser avant upload (compresser).

### Q: Combien de temps pour une analyse?
**R:** 200-500ms sur GPU, 1-2s sur CPU.

### Q: Les mensurations sont-elles précises à 100%?
**R:** Non, ±5-10% d'erreur est possible. Pas médical. Pour shopping OK.

### Q: Puis-je utiliser les mensurations pour le sizing automatique?
**R:** Oui! Créer une table de mapping taille ↔ mensurations.

### Q: Comment gérer les utilisateurs refusant la détection?
**R:** Fallback input manuel dans la BD (`is_verified=false`).

### Q: Est-ce qu'on peut utiliser SMPL pour du vidéo?
**R:** Actuellement non, image par image seulement.

### Q: Quels formats d'image sont supportés?
**R:** JPEG, PNG, GIF, WebP. Tout ce qu'OpenCV lit.

### Q: Y a-t-il des limitations légales (RGPD)?
**R:** Informer l'utilisateur, stocker les photos de façon sécurisée, droit à l'oubli.

### Q: Comment scaler si trop de requêtes?
**R:** 
- Queue les requests (Laravel Jobs)
- Load balancing du service SMPL
- Cacher les résultats
- Limiter requests par utilisateur (rate limiting)

---

## 📞 Support & Contact

- **Issue?** Vérifier les logs: `/storage/logs/laravel.log`
- **SMPL down?** Vérifier `curl http://localhost:5000/health`
- **Performance?** Activer GPU sur le serveur SMPL
- **Question?** Consulter la doc complète en USAGE.md

---

**Document créé le:** 17 février 2026  
**Version:** 1.0  
**Statut:** Stable ✅
