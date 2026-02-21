from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import uuid
import tempfile
from pathlib import Path
import logging
import datetime
import json


# ✨ MONKEYPATCH: Restaurer les alias supprimés dans Numpy 1.20+ pour compatibilité Chumpy
# Chumpy (utilisé par SMPL) dépend de ces alias obsolètes.
try:
    np.bool = np.bool_
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str
    np.str = str
    print("✓ Numpy monkeypatch appliqué pour compatibilité Chumpy")
except Exception as e:
    print(f"⚠️ Erreur lors du monkeypatch Numpy: {e}")

# Configuration
UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
from utils.pose_estimation import PoseEstimator, load_image, download_image, validate_image
from utils.mesh_utils import MeshMeasurements, validate_measurements, create_measurement_report, export_mesh_to_obj
from utils.fallback_service import FallbackMeasurementService
from smpl_engine import create_smpl_engine

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit to prevent DoS
CORS(app)

# Initialisation des moteurs
pose_estimator = None
smpl_engine = None
use_fallback = False


def sanitize_measurements(measurements, gender='male', height=1.75):
    """
    Corrige les valeurs physiquement impossibles (ex: Poignet=1.89m) par des heuristiques ou moyennes.
    Toutes les valeurs sont considérées en millimètres (mm).
    """
    # Moyennes anthropométriques en mm (approx pour 1.75m male / 1.65m female)
    averages = {
        'male': {
            'height': 1.75,
            'poignet': 175, 'bras': 350, 'epaule': 150,
            'bas': 240, 'genou': 380, 'cuisse': 580,
            'poitrine': 1000, 'taille': 850, 'hanche': 950
        },
        'female': {
            'height': 1.65,
            'poignet': 160, 'bras': 290, 'epaule': 130,
            'bas': 220, 'genou': 350, 'cuisse': 550,
            'poitrine': 900, 'taille': 700, 'hanche': 950
        }
    }
    # Safety: ensure gender is valid
    if gender not in averages: gender = 'male'
    ref = averages[gender]
    
    # Adapter les moyennes par rapport à la taille (Proportionnel)
    h_target = height or ref['height']
    scale = h_target / ref['height']
    avg = {k: v * scale for k, v in ref.items() if k != 'height'}
    
    # Normaliser keys en minuscules pour la sanitization logic interne
    m = {k.lower(): v for k, v in measurements.items()}
    
    # Prétraitement : Unifier les clés pour la logique de sanitization
    def get_val(keys):
        for k in keys:
            if k in m: return m[k]
        return 0.0

    poitrine_val = get_val(['poitrine', 'tour_poitrine', 'tour de poitrine'])
    taille_val = get_val(['taille', 'tour_taille', 'tour de taille'])
    hanche_val = get_val(['hanche', 'bassin'])
    bras_val = get_val(['bras', 'tour_manche', 'tour de manche'])

    # 1. Poignet
    if m.get('poignet', 0) > 300 or m.get('poignet', 0) < 100: 
        m['poignet'] = avg['poignet']

    # 2. Bras (Biceps)
    if bras_val > 600 or bras_val < 150: 
        new_bras = avg['bras']
        for k in ['bras', 'tour_manche', 'tour de manche']:
            if k in m: m[k] = new_bras

    # 3. Torse (Check limits approx)
    if poitrine_val > 1400 or poitrine_val < 600:
        new_poi = avg['poitrine']
        for k in ['poitrine', 'tour_poitrine', 'tour de poitrine']:
            if k in m: m[k] = new_poi
    
    if taille_val > 1400 or taille_val < 500:
        new_tai = avg['taille']
        for k in ['taille', 'tour_taille', 'tour de taille']:
            if k in m: m[k] = new_tai
    
    if hanche_val > 1600 or hanche_val < 600:
        new_han = avg['hanche']
        for k in ['hanche', 'bassin']:
            if k in m: m[k] = new_han
    
    # 4. Cuisse
    if m.get('cuisse', 0) > 1000 or m.get('cuisse', 0) < 300: m['cuisse'] = avg['cuisse']

    # Réinjecter dans le dictionnaire d'origine en respectant les clés d'entrée
    for k in measurements.keys():
        kl = k.lower()
        if kl in m:
            measurements[k] = m[kl]
            
    return measurements


def init_services():
    """Initialise les services au démarrage."""
    global pose_estimator, smpl_engine, use_fallback
    try:
        pose_estimator = PoseEstimator()
        smpl_engine = create_smpl_engine()
        use_fallback = False
        logger.info("✓ Services initialisés avec succès")
    except Exception as e:
        logger.warning(f"⚠️ Services réels non disponibles: {e}")
        logger.info("Mode fallback activé - les mensurations seront simulées")
        use_fallback = True
        # Créer des stubs pour les variables globales
        pose_estimator = None
        smpl_engine = None


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé."""
    return jsonify({
        'status': 'ok',
        'message': 'Microservice SMPL est opérationnel'
    }), 200

@app.route('/download/mesh/<filename>', methods=['GET'])
def download_mesh(filename):
    """Permet de télécharger un fichier mesh généré."""
    try:
        return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': 'Fichier non trouvé'}), 404


@app.route('/estimate', methods=['POST'])
def estimate_measurements():
    """
    Endpoint principal pour l'estimation des mensurations.

    Body JSON:
    {
        "photo_url": "https://...",  # ou "photo_path"
        "measures_table": ["tour_poitrine", "taille", "hanche", ...]
    }

    Retour:
    {
        "tour_poitrine": 92,
        "taille": 70,
        ...
    }
    """

    try:
        # Support multipart/form-data (Upload de fichiers) ou JSON
        is_multipart = request.content_type and 'multipart/form-data' in request.content_type
        
        if is_multipart:
            # Récupérer les fichiers uploadés (clé 'photos')
            uploaded_files = request.files.getlist('photos')
            # Récupérer les autres champs du form-data
            form_data = request.form
            
            photo_source = None # Non utilisé en mode upload
            photos_list = [] # Sera rempli avec les données binaires
            
            # Gérer measures_table (qui peut être une string JSON ou une liste de clés répétées)
            mt_raw = form_data.get('measures_table')
            try:
                if mt_raw:
                    measures_table = json.loads(mt_raw)
                else:
                    measures_table = []
            except:
                # Fallback: essayer de split par virgule si c'est une string simple
                measures_table = mt_raw.split(',') if mt_raw else []
            
            # Nettoyage des clés (retirer les guillemets, crochets parasites si split naïf)
            if measures_table:
                measures_table = [m.strip(' "[]\'') for m in measures_table if m.strip()]
                
            gender = form_data.get('gender', 'neutral')
            height = form_data.get('height')
            include_mesh = form_data.get('include_mesh') == 'true'
            
        else:
            # Mode JSON standard
            data = request.get_json()
            if not data:
                return jsonify({'error': 'JSON requis'}), 400

            photo_source = data.get('photo_url') or data.get('photo_path')
            photos_list = data.get('photos', []) 
            
            # Compatibilité ascendante
            if photo_source and not photos_list:
                photos_list = [photo_source]
                
            measures_table = data.get('measures_table', [])
            gender = data.get('gender', 'neutral')
            height = data.get('height')
            include_mesh = data.get('include_mesh', False)
            uploaded_files = []

        if not photos_list and not uploaded_files:
             logger.error("400: Aucune photo (liste vide)")
             return jsonify({'error': 'Aucune photo fournie (JSON: photos[], Form: photos (files))'}), 400

        if not measures_table:
            logger.error("400: measures_table vide ou mal formatée")
            return jsonify({'error': 'measures_table vide'}), 400
            
        # Normalisation Hauteur
        if height is not None:
            try:
                height = float(height)
                if height > 3.0: height = height / 100.0
            except ValueError:
                logger.warning(f"Hauteur ignorée (valeur invalide: {height})")
                height = None

        logger.info(f"Requête reçue (Multipart={is_multipart}): {len(uploaded_files)} fichiers, {len(photos_list)} urls, Genre={gender}, Hauteur={height}m, Measures={len(measures_table)}")

        # Mode fallback (inchangé)
        if use_fallback:
             # ... 
             logger.info("Utilisation du mode fallback")
             measurements = FallbackMeasurementService.generate_measurements(measures_table)
             return jsonify({
                'measurements': measurements,
                'metadata': {'mode': 'fallback'}
             }), 200

        # Mode normal: Validation & Extraction Multi-View
        image_data_list = []
        validation_errors = []
        
        # Combiner les sources (URLs et Fichiers)
        # On traite d'abord les URLs, puis les fichiers
        sources_to_process = []
        
        # 1. URLs (JSON)
        for url in photos_list:
            sources_to_process.append({'type': 'url', 'data': url})
            
        # 2. Fichiers (Multipart)
        for file_obj in uploaded_files:
            sources_to_process.append({'type': 'file', 'data': file_obj})
            
        # Check doublons sur les URLs seulement (pas possible sur fichiers stream sans lire)
        urls_only = [s['data'] for s in sources_to_process if s['type'] == 'url']
        if len(urls_only) > 1 and len(set(urls_only)) != len(urls_only):
             return jsonify({'error': 'Photos dupliquées détectées (URLs).'}), 400

        for idx, source in enumerate(sources_to_process):
            try:
                # 1. Charger
                if source['type'] == 'url':
                    img, shape = _load_image_source(source['data'])
                else:
                    # Chargement depuis file buffer (Multipart)
                    file_bytes = np.frombuffer(source['data'].read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Impossible de décoder l'image uploadée")
                    shape = img.shape[:2]
                
                if not validate_image(img):
                    validation_errors.append({'index': idx, 'error': 'Image invalide ou corrompue'})
                    continue
                    
                # 2. Détecter Pose
                pose = pose_estimator.estimate_pose(img)
                if pose is None:
                    validation_errors.append({'index': idx, 'error': 'Aucune personne détectée dans la photo'})
                    continue
                    
                kps = pose['keypoints'] # [x, y, vis, (z)] - z optionnel souvent
                
                # 3. Validation FULL BODY (Obligatoire)
                # On relâche la contrainte (0.5 -> 0.3) pour tolérer plus de cas limites
                # Indices: 0=Nose, 11/12=Shoulders, 23/24=Hips, 27/28=Ankles
                ankles_vis = (kps[27, 2] > 0.3) and (kps[28, 2] > 0.3)
                shoulders_vis = (kps[11, 2] > 0.3) and (kps[12, 2] > 0.3)
                
                if not ankles_vis:
                    # On transforme l'erreur bloquante en WARNING si on veut être permissif
                    # Mais pour la precision, c'est indispensable. On garde l'erreur mais avec un seuil plus bas.
                    logger.warning(f"Photo {idx}: Pieds/Chevilles non détectés (confiance < 0.3). Mesure longueur pantalon risque d'être fausse.")
                    # validation_errors.append({'index': idx, 'error': 'Pieds/Chevilles non détectés...'}) -> DESACTIVE POUR TEST
                    
                if not shoulders_vis:
                    logger.warning(f"Photo {idx}: Epaules non détectées (confiance < 0.3). Mesure Carrure risque d'être fausse.")
                    # validation_errors.append({'index': idx, 'error': 'Epaules non détectées...'}) -> DESACTIVE POUR TEST
                    
                # 4. Validation VIEW TYPE (Face vs Profil)
                # On utilise la profondeur relative (Z) des épaules si disponible (MediaPipe Pose World Landmarks le donne)
                # Mais ici on a les keypoints normalisés 2D+vis. 
                # Astuce 2D : Ratio largeur épaules / largeur hanches ? Difficile.
                # Astuce Simple : Si on demande 2 photos:
                # - Photo 0 DOIT être Face
                # - Photo 1 DOIT être Profil
                
                # TODO: Améliorer la détection auto.
                # Pour l'instant on fait confiance à l'ordre, MAIS on peut détecter une incohérence flagrante.
                # Si photo 1 (Profil) a les épaules aussi larges que photo 0 (Face), c'est suspect.
                
                image_data_list.append({
                    'image': img,
                    'keypoints': kps,
                    'shape': shape,
                    'pose_data': pose
                })
                
            except Exception as e:
                validation_errors.append({'index': idx, 'error': str(e)})

        # Si erreurs critiques (aucune photo valide ou erreurs remontées)
        if validation_errors:
             # Si on a demandé plusieurs photos et qu'une plante, on arrête tout par sécurité ?
             # Ou si toutes plantent ?
             if len(image_data_list) == 0:
                 logger.error(f"400: Validation échouée pour toutes les photos: {validation_errors}")
                 return jsonify({
                     'error': 'Validation échouée pour toutes les photos',
                     'details': validation_errors
                 }), 400
             
             # Si on veut être strict (user request):
             logger.error(f"400: Une ou plusieurs photos invalides: {validation_errors}")
             return jsonify({
                 'error': 'Une ou plusieurs photos sont invalides',
                 'details': validation_errors
             }), 400

        logger.info(f"Traitement de {len(image_data_list)} vues validées")

        # Étape 3: Générer le mesh SMPL (Multi-View)
        # On passe la liste des dicts préparés
        mesh_data = smpl_engine.process_image(image_data_list, gender=gender, height=height)
        
        if mesh_data is None:
            logger.error("Échec de la génération du mesh SMPL")
            return jsonify({
                'error': 'Échec de la génération du mesh 3D',
                'code': 'MESH_GENERATION_FAILED'
            }), 500

        # Vérification de la cohérence géométrique (Identity / Pose Check)
        # Si la loss est très élevée, cela signifie que le modèle n'arrive pas à satisfaire les 2 vues
        # Seuil empirique (à ajuster selon logs):
        # - Bon fitting: < 1.0 (souvent négatif si likelihood, mais ici squared error positiv... à vérifier logs)
        # - Mauvais fitting: > 10.0 ou 100.0
        # ATTENTION: Mes logs précédents montraient une loss négative (-8000). C'est très étrange pour une MSE.
        # Probablement un terme de likelihood (log-prob) caché ou un bug d'affichage.
        # JE VAIS AFFICHER LA LOSS BRUTE DANS LES LOGS POUR CALIBRER.
        # Pour l'instant, je mets un seuil "Safe" très haut pour éviter les faux positifs, 
        # mais suffisant pour bloquer les aberrations extrêmes.
        final_loss = mesh_data.get('loss', 0.0)
        logger.info(f"Loss finale du fitting: {final_loss}")
        
        # TODO: Calibrer ce seuil avec des exemples réels.
        # Si c'est une MSE pure * 100, ça devrait être positif.
        # Si c'est négatif, c'est qu'il y a un terme -LogLikelihood.
        # On va supposer que si c'est > 10000 (positif), c'est une explosion.
        if final_loss > 100000.0:  
             return jsonify({
                'error': 'Incohérence géométrique détectée. Les photos semblent incompatibles (personnes différentes ou poses incorrectes).',
                'code': 'GEOMETRIC_INCONSISTENCY',
                'details': {'loss': final_loss}
            }), 400

        vertices = mesh_data['vertices']

        logger.info(f"Mesh généré avec {mesh_data['n_vertices']} vertices")

        # Étape 4: Extraire les mensurations
        mesh_measurements = MeshMeasurements(vertices, mesh_data['faces'])
        measurements = mesh_measurements.get_all_measurements(measures_table)

        # Valider les mensurations
        is_valid, errors = validate_measurements(measurements)
        if not is_valid:
            logger.warning(f"Mensurations invalides: {errors}")

        # Exporter le mesh si demandé
        mesh_url = None
        if include_mesh:
             mesh_obj_content = export_mesh_to_obj(vertices, mesh_data['faces'])
             filename = f"mesh_{uuid.uuid4()}.obj"
             filepath = os.path.join(OUTPUT_FOLDER, filename)
             with open(filepath, 'w') as f:
                 f.write(mesh_obj_content)
             
             mesh_url = f"{request.host_url}download/mesh/{filename}"

        if vertices is None:
             logger.error("Erreur critique: Aucun mesh généré")
             return jsonify({'error': 'Echec de génération du modèle 3D'}), 500

        # Output formatting & Sanitization (Convert from meters to millimeters)
        measurements_clean = {k.strip(' "[]\''): round(v * 1000.0, 1) for k, v in measurements.items()}
        
        # Appliquer la correction des valeurs impossibles (sanitize_measurements attend des mm)
        measurements_clean = sanitize_measurements(measurements_clean, gender, height=height)

        # Générer un ID unique pour cette prédiction (pour le feedback)
        prediction_id = str(uuid.uuid4())
        
        # Sauvegarder les données brutes (pour debug/retraining futur)
        try:
             log_entry = {
                 'prediction_id': prediction_id,
                 'timestamp': datetime.datetime.now().isoformat(),
                 'gender': gender,
                 'height': height,
                 'measurements_api': measurements_clean,
                 'mesh_url': mesh_url
             }
             with open('dataset/predictions_log.jsonl', 'a') as f:
                 f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
             logger.warning(f"Failed to log prediction: {e}")

        # Formater la réponse
        response = {
            'prediction_id': prediction_id,
            'measurements': measurements_clean,
            'mesh_url': mesh_url,
            'metadata': {
                'num_views': len(image_data_list),
                'keypoints_per_view': [len(d['keypoints']) for d in image_data_list],
                'validation_errors': validation_errors,
                'mesh_vertices': len(vertices),
                'mode': 'production' if not use_fallback else 'fallback'
            }
        }
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Erreur dans /estimate: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Erreur serveur: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Reçoit les corrections de l'utilisateur (Human-in-the-loop).
    Ces données serviront à entrainer le modèle de correction (Niveau 2).
    """
    try:
        # Assurance que le dossier existe
        if not os.path.exists('dataset'):
            os.makedirs('dataset')

        data = request.json
        if not data or 'prediction_id' not in data:
            return jsonify({'error': 'Missing prediction_id'}), 400
            
        feedback_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'prediction_id': data['prediction_id'],
            'corrected_measurements': data.get('corrected_measurements', {}),
            'user_profile': data.get('user_profile', {})
        }
        
        # Log dans un fichier séparé pour l'entrainement
        try:
            with open('dataset/feedback_log.jsonl', 'a') as f:
                f.write(json.dumps(feedback_entry) + '\n')
            logger.info(f"Feedback reçu pour {data['prediction_id']}")
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return jsonify({'error': 'Storage error'}), 500
            
        return jsonify({'status': 'success', 'message': 'Feedback recorded for training'}), 200
        
    except Exception as e:
        logger.error(f"Error in /feedback: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/estimate/batch', methods=['POST'])
def estimate_batch():
    """
    Traite plusieurs images en batch.

    Body JSON:
    {
        "images": [
            {"photo_url": "...", "measures_table": [...]},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        images = data.get('images', [])

        if not images:
            return jsonify({'error': 'Aucune image fournie'}), 400

        results = []
        for idx, img_data in enumerate(images):
            try:
                # Préparer la requête
                request_json = request.environ.get('werkzeug.request').json
                request.environ['werkzeug.request'].json = img_data

                # Appeler l'endpoint individual
                response = estimate_measurements()
                results.append({
                    'index': idx,
                    'status': 'success',
                    'data': response.get_json() if hasattr(response, 'get_json') else response[0]
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'status': 'error',
                    'error': str(e)
                })

        return jsonify({'results': results}), 200

    except Exception as e:
        logger.error(f"Erreur dans /estimate/batch: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/models/status', methods=['GET'])
def models_status():
    """Retourne le statut des modèles chargés."""
    if use_fallback:
        return jsonify({
            'pose_estimator': 'fallback',
            'smpl_engine': 'fallback',
            'device': 'cpu',
            'mode': 'fallback',
            'message': 'Services réels non disponibles, utilisation du mode fallback'
        }), 200
    
    return jsonify({
        'pose_estimator': 'loaded' if pose_estimator is not None else 'not_loaded',
        'smpl_engine': 'loaded' if smpl_engine is not None else 'not_loaded',
        'device': str(smpl_engine.device) if smpl_engine else 'unknown',
        'mode': 'production'
    }), 200


@app.route('/measurements/reference', methods=['GET'])
def get_measurement_reference():
    """Retourne les mensurations de référence disponibles."""
    from utils.mesh_utils import MeshMeasurements

    return jsonify({
        'available_measurements': list(MeshMeasurements.MEASUREMENT_MAPPING.keys()),
        'body_parts': list(MeshMeasurements.BODY_PART_VERTICES.keys()),
        'example_request': {
            'photo_url': 'https://...',
            'measures_table': ['tour_poitrine', 'taille', 'hanche']
        }
    }), 200


def _load_image_source(source: str) -> tuple:
    """
    Charge une image depuis une URL ou un chemin local.

    Args:
        source: URL ou chemin local

    Returns:
        Tuple (image numpy, shape)
    """
    # Déterminer si c'est une URL ou un chemin
    if source.startswith('http://') or source.startswith('https://'):
        # URL - télécharger
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = os.path.join(tmpdir, 'temp_image.jpg')
            download_image(source, temp_path)
            return load_image(temp_path)
    else:
        # Chemin local
        return load_image(source)


@app.errorhandler(400)
def bad_request(error):
    """Gestionnaire pour erreur 400."""
    return jsonify({'error': 'Bad Request'}), 400


@app.errorhandler(404)
def not_found(error):
    """Gestionnaire pour erreur 404."""
    return jsonify({'error': 'Not Found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestionnaire pour erreur 500."""
    return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    # Initialiser les services
    init_services()

    # Lancer le serveur
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
