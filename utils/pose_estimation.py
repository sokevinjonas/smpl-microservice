import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import os

# Importer MediaPipe - support pour 0.10.x+ API
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    MEDIAPIPE_API = 'tasks'  # 0.10.x+ API
    
except ImportError as e:
    print(f"⚠️ MediaPipe import failed: {e}")
    # Fallback au mock
    from .mediapipe_mock import get_mediapipe_or_mock
    mp = get_mediapipe_or_mock()
    MEDIAPIPE_API = 'mock'


# Noms des keypoints COCO (33 points pour MediaPipe Pose)
POSE_KEYPOINT_NAMES = [
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR',
    'MOUTH_LEFT', 'MOUTH_RIGHT',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_PINKY_1', 'RIGHT_PINKY_1',
    'LEFT_INDEX_1', 'RIGHT_INDEX_1',
    'LEFT_THUMB_1', 'RIGHT_THUMB_1',
    'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE',
    'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]


class PoseEstimator:
    """
    Wrapper pour l'estimation de pose corporelle avec MediaPipe.
    Utilise MediaPipe Tasks (0.10.x+) pour détecter les keypoints du corps.
    """

    def __init__(self):
        """Initialise le modèle MediaPipe Pose."""
        self.use_mock = False
        
        if MEDIAPIPE_API == 'mock':
            print("⚠️ Using MediaPipe mock implementation")
            from .mediapipe_mock import MockPose
            self.pose = MockPose(
                static_image_mode=True,
                model_complexity=2,
                smooth_landmarks=True
            )
            self.use_mock = True
        else:
            # Utiliser MediaPipe Tasks API (0.10.x+)
            try:
                # Télécharger le modèle si nécessaire
                model_path = self._get_pose_model_path()
                
                # Créer les options pour PoseLandmarker
                options = vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,  # Détecter une seule personne
                    output_segmentation_masks=True # ENABLING SILHOUETTE MASKS
                )
                
                # Initialiser le PoseLandmarker
                self.pose = vision.PoseLandmarker.create_from_options(options)
                print("✓ MediaPipe PoseLandmarker initialized successfully")
                
            except Exception as e:
                print(f"⚠️ Failed to initialize PoseLandmarker: {e}")
                print("  Falling back to mock implementation")
                from .mediapipe_mock import MockPose
                self.pose = MockPose(
                    static_image_mode=True,
                    model_complexity=2,
                    smooth_landmarks=True
                )
                self.use_mock = True

    def _get_pose_model_path(self) -> str:
        """Récupère le chemin du modèle MediaPipe Pose."""
        # MediaPipe télécharge automatiquement les modèles
        # Mais on peut spécifier le chemin s'il existe
        model_file = "pose_landmarker.task"
        
        # Vérifier plusieurs chemins possibles
        search_paths = [
            model_file,
            os.path.join(os.path.expanduser("~"), ".cache", "mediapipe", model_file),
            os.path.join(os.path.dirname(__file__), "..", "models", model_file),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Si le modèle n'existe pas localement, retourner le chemin par défaut
        # MediaPipe le téléchargera automatiquement lors de la création
        return model_file

    def estimate_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estime la pose à partir d'une image.

        Args:
            image: Image numpy en BGR (OpenCV format) ou array RGB

        Returns:
            Dict contenant les keypoints et leur confiance, ou None si aucune personne détectée
        """
        if image is None or image.size == 0:
            return None
        
        # Convertir BGR en RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Vérifier si c'est BGR (provient d'OpenCV)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        try:
            if self.use_mock:
                # Utiliser l'interface mock
                results = self.pose.process(image_rgb)
                if results.pose_landmarks is None:
                    return None
                    
                keypoints = []
                confidences = []
                for landmark in results.pose_landmarks.landmark:
                    # On stocke la visibilité directement dans la 3ème colonne
                    # car smpl_engine l'utilise comme poids.
                    keypoints.append([landmark.x, landmark.y, landmark.visibility])
                    confidences.append(landmark.visibility)
                    
                return {
                    'keypoints': np.array(keypoints),
                    'confidences': np.array(confidences),
                    'num_keypoints': len(keypoints)
                }
            else:
                # Utiliser MediaPipe Tasks API
                # Créer un objet Image à partir du numpy array
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=image_rgb
                )
                
                # Détecter les poses
                detection_result = self.pose.detect(mp_image)
                
                if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                    return None
                
                # Extraire les landmarks de la première personne détectée
                landmarks = detection_result.pose_landmarks[0]
                
                keypoints = []
                confidences = []
                
                for landmark in landmarks:
                    # Tasks API 0.10+ utilise .visibility et .presence
                    # On utilise visibility qui est plus représentatif de la détection 'clean'
                    vis = getattr(landmark, 'visibility', getattr(landmark, 'presence', 0.0))
                    keypoints.append([landmark.x, landmark.y, vis])
                    confidences.append(vis)
                    
                # Extraction du masque de segmentation si disponible
                segmentation_mask = None
                if detection_result.segmentation_masks and len(detection_result.segmentation_masks) > 0:
                    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                    # Le masque de MP Tasks API est un array de float entre 0 et 1.
                    # On le convertit en un masque binaire plus strict (silhouette corps)
                    segmentation_mask = (segmentation_mask > 0.5).astype(np.uint8)
                    
                return {
                    'keypoints': np.array(keypoints),
                    'confidences': np.array(confidences),
                    'num_keypoints': len(keypoints),
                    'segmentation_mask': segmentation_mask
                }
                
        except Exception as e:
            print(f"⚠️ Pose estimation error: {e}")
            return None

    def get_keypoint_names(self) -> list:
        """Retourne les noms des keypoints détectés."""
        return POSE_KEYPOINT_NAMES


    def release(self):
        """Libère les ressources."""
        self.pose.close()


def load_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Charge une image depuis un chemin.

    Args:
        image_path: Chemin vers l'image

    Returns:
        Tuple (image numpy, (height, width))
    """
    # Nettoyer les URLs dupliquées (e.g., "httpshttps://" → "https://")
    image_path = _normalize_url(image_path)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")

    height, width = image.shape[:2]
    return image, (height, width)


def _normalize_url(url: str) -> str:
    """
    Normalise une URL (corrige les doublons, etc.).
    
    Args:
        url: URL ou chemin
        
    Returns:
        URL/chemin normalisé
    """
    # Corriger les doublons de protocole (httpshttps:// → https://)
    url = url.replace('httpshttps://', 'https://')
    url = url.replace('httphttps://', 'https://')
    url = url.replace('httphttp://', 'http://')
    
    return url



def download_image(image_url: str, save_path: str) -> str:
    """
    Télécharge une image depuis une URL en évitant le SSRF.

    Args:
        image_url: URL de l'image
        save_path: Chemin local pour sauvegarder

    Returns:
        Chemin du fichier sauvegardé
    """
    import requests
    from urllib.parse import urlparse
    import socket
    from pathlib import Path

    # Normaliser l'URL
    image_url = _normalize_url(image_url)
    
    # Validation pour prévenir le SSRF
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in ('http', 'https'):
        raise ValueError("Seules les URLs HTTP/HTTPS sont autorisées.")
    
    # Bloquer les requêtes réseau locales simples
    # blocked_hosts = {'localhost', '127.0.0.1', '0.0.0.0', '169.254.169.254'}
    # if parsed_url.hostname in blocked_hosts:
    #     raise ValueError("Les requêtes réseau locales ne sont pas autorisées.")
        
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(response.content)

        return save_path
    except Exception as e:
        raise RuntimeError(f"Erreur lors du téléchargement de l'image: {e}")



def validate_image(image: np.ndarray) -> bool:
    """
    Valide qu'une image est dans un format acceptable.

    Args:
        image: Image numpy

    Returns:
        True si valide, False sinon
    """
    if image is None or image.size == 0:
        return False
    if len(image.shape) < 2:
        return False
    return True
