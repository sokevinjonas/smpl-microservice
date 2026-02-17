from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import logging

from utils.pose_estimation import PoseEstimator, load_image, download_image, validate_image
from utils.mesh_utils import MeshMeasurements, validate_measurements, create_measurement_report
from utils.fallback_service import FallbackMeasurementService
from smpl_engine import create_smpl_engine

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation Flask
app = Flask(__name__)
CORS(app)

# Initialisation des moteurs
pose_estimator = None
smpl_engine = None
use_fallback = False


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
        # Récupérer les paramètres
        data = request.get_json()
        if not data:
            return jsonify({'error': 'JSON requis'}), 400

        photo_source = data.get('photo_url') or data.get('photo_path')
        measures_table = data.get('measures_table', [])

        if not photo_source:
            return jsonify({'error': 'photo_url ou photo_path requis'}), 400

        if not measures_table:
            return jsonify({'error': 'measures_table vide'}), 400

        logger.info(f"Requête reçue pour: {photo_source}")

        # Mode fallback: générer des mensurations simulées
        if use_fallback:
            logger.info("Utilisation du mode fallback (services indisponibles)")
            measurements = FallbackMeasurementService.generate_measurements(measures_table)
            
            return jsonify({
                'measurements': measurements,
                'metadata': {
                    **FallbackMeasurementService.get_fallback_metadata(),
                    'mode': 'fallback',
                    'message': 'Mensurations générées en mode fallback (services réels indisponibles)'
                }
            }), 200

        # Mode normal: utiliser les services réels
        # Étape 1: Télécharger/Charger l'image
        image, image_shape = _load_image_source(photo_source)

        if not validate_image(image):
            return jsonify({'error': 'Image invalide'}), 400

        logger.info(f"Image chargée: {image_shape}")

        # Étape 2: Estimer la pose
        pose_data = pose_estimator.estimate_pose(image)
        if pose_data is None:
            return jsonify({
                'error': 'Aucune personne détectée dans l\'image',
                'code': 'NO_PERSON_DETECTED'
            }), 400

        logger.info(f"Pose estimée avec {pose_data['num_keypoints']} keypoints")

        # Étape 3: Générer le mesh SMPL
        mesh_data = smpl_engine.process_image(image, pose_data['keypoints'])
        vertices = mesh_data['vertices']

        logger.info(f"Mesh généré avec {mesh_data['n_vertices']} vertices")

        # Étape 4: Extraire les mensurations
        mesh_measurements = MeshMeasurements(vertices)
        measurements = mesh_measurements.get_all_measurements(measures_table)

        # Valider les mensurations
        is_valid, errors = validate_measurements(measurements)
        if not is_valid:
            logger.warning(f"Mensurations invalides: {errors}")

        # Formater la réponse
        response = {
            'measurements': measurements,
            'metadata': {
                'image_shape': image_shape,
                'num_keypoints': pose_data['num_keypoints'],
                'mesh_vertices': mesh_data['n_vertices'],
                'validation_errors': errors if not is_valid else [],
                'mode': 'production'
            }
        }

        logger.info(f"Mensurations calculées: {measurements}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Erreur dans /estimate: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Erreur serveur: {str(e)}',
            'code': 'INTERNAL_ERROR'
        }), 500


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
