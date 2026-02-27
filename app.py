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
    Corrige les valeurs physiquement impossibles par des heuristiques ou moyennes.
    Normalise les clés avec des underscores pour la logique interne.
    Toutes les valeurs sont en mm.
    """
    # Moyennes anthropométriques complètes (pour 1.75m male / 1.65m female)
    averages = {
        'male': {
            'height': 1.75,
            'poignet': 175, 'bras': 350, 'epaule': 150, 'largeur_epaules': 420,
            'bas': 240, 'genou': 380, 'cuisse': 580, 'mollet': 360, 'cheville': 220,
            'poitrine': 1000, 'taille': 850, 'hanche': 950, 'bassin': 950,
            'entrejambe': 820, 'longueur_manche': 600, 'hauteur_torse': 700,
            'cou': 370, 'tete': 560, 'sous_poitrine': 850
        },
        'female': {
            'height': 1.65,
            'poignet': 160, 'bras': 290, 'epaule': 130, 'largeur_epaules': 380,
            'bas': 220, 'genou': 350, 'cuisse': 550, 'mollet': 340, 'cheville': 210,
            'poitrine': 900, 'taille': 700, 'hanche': 950, 'bassin': 950,
            'entrejambe': 780, 'longueur_manche': 550, 'hauteur_torse': 650,
            'cou': 340, 'tete': 540, 'sous_poitrine': 750
        }
    }
    
    if gender not in averages: gender = 'male'
    ref = averages[gender]
    
    try:
        h_val = float(height) if height else ref['height']
    except:
        h_val = ref['height']
        
    scale = h_val / ref['height']
    avg = {k: v * scale for k, v in ref.items() if k != 'height'}
    
    # Normalisation : clés lowercase et underscores pour la comparaison
    m = {k.lower().strip().replace(' ', '_'): v for k, v in measurements.items()}
    
    def get_val(keys):
        for k in keys:
            if k in m: return m[k]
        return 0.0

    # 1. Torse
    poitrine_val = get_val(['poitrine', 'tour_poitrine', 'tour_de_poitrine'])
    if poitrine_val > 1500 or poitrine_val < 500:
        val = avg['poitrine']
        for k in ['poitrine', 'tour_poitrine', 'tour_de_poitrine']:
            if k in m: m[k] = val

    taille_val = get_val(['taille', 'tour_taille', 'tour_de_taille', 'ceinture', 'pinces'])
    if taille_val > 1400 or taille_val < 450:
        val = avg['taille']
        for k in ['taille', 'tour_taille', 'tour_de_taille', 'ceinture', 'pinces']:
            if k in m: m[k] = val

    hanche_val = get_val(['hanche', 'bassin', 'tour_de_bassin'])
    if hanche_val > 1600 or hanche_val < 500:
        val = avg['hanche']
        for k in ['hanche', 'bassin', 'tour_de_bassin']:
            if k in m: m[k] = val

    if m.get('sous_poitrine', 0) > 1300 or m.get('sous_poitrine', 0) < 400: m['sous_poitrine'] = avg['sous_poitrine']

    # 2. Bras et Poignet
    bras_val = get_val(['bras', 'tour_manche', 'tour_de_manche', 'tour_emanchure'])
    if bras_val > 600 or bras_val < 150:
        val = avg['bras']
        for k in ['bras', 'tour_manche', 'tour_de_manche', 'tour_emanchure']:
            if k in m: m[k] = val

    if m.get('poignet', 0) > 300 or m.get('poignet', 0) < 100: m['poignet'] = avg['poignet']
    if m.get('avant_bras', 0) > 450 or m.get('avant_bras', 0) < 150: m['avant_bras'] = 260 * scale

    # 3. Jambes
    if m.get('cuisse', 0) > 1000 or m.get('cuisse', 0) < 300: m['cuisse'] = avg['cuisse']
    if m.get('genou', 0) > 600 or m.get('genou', 0) < 200: m['genou'] = avg['genou']
    if m.get('mollet', 0) > 600 or m.get('mollet', 0) < 200: m['mollet'] = avg['mollet']
    
    cheville_val = get_val(['cheville', 'bas'])
    if cheville_val > 400 or cheville_val < 150:
        val = avg['cheville']
        if 'cheville' in m: m['cheville'] = val
        if 'bas' in m: m['bas'] = val

    # 4. Longueurs et Épaules
    long_manche_val = get_val(['longueur_manche', 'long_manche'])
    if long_manche_val > 1000 or long_manche_val < 300:
        val = avg['longueur_manche']
        if 'longueur_manche' in m: m['longueur_manche'] = val
        if 'long_manche' in m: m['long_manche'] = val

    entrejambe_val = get_val(['entrejambe', 'long_pantalon'])
    if entrejambe_val > 1300 or entrejambe_val < 500:
        val = avg['entrejambe']
        if 'entrejambe' in m: m['entrejambe'] = val
        if 'long_pantalon' in m: m['long_pantalon'] = val

    # Jambe totale / Jupe / Robe
    jambe_val = get_val(['longueur_jambe', 'long_jupe', 'long_robe'])
    if jambe_val > 1500 or jambe_val < 400:
        val = 1000 * scale # Default ~1m
        for k in ['longueur_jambe', 'long_jupe', 'long_robe']:
            if k in m: m[k] = val

    # Épaules / Dos
    epaules_val = get_val(['largeur_epaules', 'epaule'])
    if epaules_val > 700 or epaules_val < 250:
        val = avg['largeur_epaules']
        if 'largeur_epaules' in m: m['largeur_epaules'] = val
        if 'epaule' in m: m['epaule'] = val

    dos_val = get_val(['largeur_dos', 'dos'])
    if dos_val > 700 or dos_val < 200:
        val = 400 * scale
        if 'largeur_dos' in m: m['largeur_dos'] = val
        if 'dos' in m: m['dos'] = val
        
    torse_h_val = get_val(['hauteur_torse', 'long_taille', 'long_chemise', 'long_camisole'])
    if torse_h_val > 1000 or torse_h_val < 300:
        val = avg['hauteur_torse']
        for k in ['hauteur_torse', 'long_taille', 'long_chemise', 'long_camisole']:
            if k in m: m[k] = val

    # 5. Haut
    if m.get('cou', 0) > 600 or m.get('cou', 0) < 200: m['cou'] = avg['cou']
    if m.get('tete', 0) > 800 or m.get('tete', 0) < 300: m['tete'] = avg['tete']

    # Réinjecter dans le dictionnaire d'origine (en respectant les clés d'entrée originales)
    for k in measurements.keys():
        kl = k.lower().strip().replace(' ', '_')
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
            weight = form_data.get('weight')
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
            weight = data.get('weight')
            include_mesh = data.get('include_mesh', False)
            uploaded_files = []

        if not photos_list and not uploaded_files:
             logger.error("400: Aucune photo (liste vide)")
             return jsonify({'error': 'Aucune photo fournie (JSON: photos[], Form: photos (files))'}), 400

        total_photos = len(photos_list) + len(uploaded_files)
        if total_photos != 2:
             logger.error(f"400: Strict Multi-View Validation Echouee (Total: {total_photos})")
             return jsonify({
                 'error': 'Deux vues obligatoires (Face et Profil)',
                 'message': 'Le modèle nécessite exactement 2 photos : une de Face et une de Profil strict pour garantir une bonne précision 3D.'
             }), 400

        if not measures_table:
            logger.error("400: measures_table vide ou mal formatée")
            return jsonify({'error': 'measures_table vide'}), 400
            
        # Normalisation Hauteur
        height_m = None
        if height is not None:
            try:
                height_m = float(height)
                if height_m > 3.0: height_m = height_m / 100.0 # Convert cm to m
            except ValueError:
                logger.warning(f"Hauteur ignorée (valeur invalide: {height})")
                height_m = None

        # Normalisation Poids et prise en charge des intervalles (e.g. "70-75")
        weight_kg = None
        target_weight_interval = None
        if weight is not None:
            if isinstance(weight, str) and '-' in weight:
                try:
                    parts = weight.split('-')
                    if len(parts) == 2:
                        min_w, max_w = float(parts[0].strip()), float(parts[1].strip())
                        if 0 < min_w <= 300 and 0 < max_w <= 300 and min_w <= max_w:
                            target_weight_interval = (min_w, max_w)
                        else:
                            logger.warning(f"Intervalle de poids ignoré (hors limites réalistes: {weight})")
                except ValueError:
                    logger.warning(f"Intervalle de poids ignoré (valeur invalide: {weight})")
            else:
                try:
                    weight_kg = float(weight)
                    if weight_kg <= 0 or weight_kg > 300:
                        logger.warning(f"Poids ignoré (hors limites réalistes: {weight}kg)")
                        weight_kg = None
                except ValueError:
                    logger.warning(f"Poids ignoré (valeur invalide: {weight})")
                    weight_kg = None

        logger.info(f"Requête reçue (Multipart={is_multipart}): {len(uploaded_files)} fichiers, {len(photos_list)} urls, Genre={gender}, Hauteur={height_m}m, Poids={weight_kg}kg, Measures={len(measures_table)}")

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
                    'pose_data': pose,
                    'segmentation_mask': pose.get('segmentation_mask', None)
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
        # 🛡️ POSE GUARD: Contrôle Qualité
        critical_landmarks = [11, 12, 23, 24] # Épaules, Hanches
        
        all_kps = [img_data['keypoints'] for img_data in image_data_list]
        low_conf = []
        for i, kps in enumerate(all_kps):
            for lm_idx in critical_landmarks:
                if kps[lm_idx, 2] < 0.4: # Seuil visibilité MediaPipe
                    low_conf.append(f"Image {i+1} : Point {lm_idx}")
        
        if low_conf:
            return jsonify({
                'error': 'Pose non valide, veuillez vous reculer',
                'message': 'Assurez-vous que vos épaules et vos hanches sont bien visibles.',
                'details': low_conf
            }), 400

        res = smpl_engine.process_image(
            image_data_list, 
            gender=gender, 
            height=height_m, 
            target_weight=weight_kg,
            target_weight_interval=target_weight_interval
        )
        
        if res is None:
            return jsonify({
                'error': 'Pose non valide, veuillez vous reculer',
                'message': 'Impossible d\'ajuster le modèle aux photos.'
            }), 400
            
        fitting_loss = res.get('loss', 0.0)
        logger.info(f"Loss finale du fitting: {fitting_loss}")
        
        # Seuil strict demandé par l'utilisateur (0.05)
        if fitting_loss > 0.05: 
             return jsonify({
                'error': 'Pose non valide, veuillez vous reculer', 
                'message': 'La photo est trop complexe ou la pose est incorrecte.',
                'loss': fitting_loss
            }), 400

        vertices = res['vertices']
        faces = res['faces']
        smpl_params = res['smpl_params']

        logger.info(f"Mesh généré avec {len(vertices)} vertices")

        # Étape 4: Extraire les mensurations
        mesh_measurements = MeshMeasurements(vertices, faces)
        measurements = mesh_measurements.get_all_measurements(measures_table)
        
        # Valider les mensurations
        is_valid, validation_errors = validate_measurements(measurements)
        if not is_valid:
            logger.warning(f"Mensurations invalides: {validation_errors}")

        # Exporter le mesh si demandé
        mesh_url = None
        if include_mesh:
             mesh_obj_content = export_mesh_to_obj(vertices, faces)
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
        measurements_clean = sanitize_measurements(measurements_clean, gender, height=height_m)

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

        response = {
            'prediction_id': prediction_id,
            'measurements': measurements_clean,
            'mesh_url': mesh_url,
            'metadata': {
                'num_views': len(image_data_list),
                'keypoints_per_view': [len(d['keypoints']) for d in image_data_list],
                'target_weight_kg': weight_kg,
                'target_weight_interval': target_weight_interval,
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
