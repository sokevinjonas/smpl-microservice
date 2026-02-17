"""
Mock de MediaPipe pour développement/test
Simule la détection de pose sans dépendre de MediaPipe réel
"""

import numpy as np
from typing import List, Optional


class MockLandmark:
    """Simule un landmark MediaPipe"""
    def __init__(self, x: float, y: float, z: float, visibility: float = 0.9):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class MockLandmarks:
    """Simule une liste de landmarks"""
    def __init__(self, num_landmarks: int = 33):
        self.landmark = self._generate_random_landmarks(num_landmarks)
    
    def _generate_random_landmarks(self, num_landmarks: int) -> List[MockLandmark]:
        """Génère des landmarks aléatoires (personne détectée)"""
        landmarks = []
        for i in range(num_landmarks):
            # Générer des coordonnées réalistes (0-1 pour x,y, -1 à 1 pour z)
            x = np.random.uniform(0.2, 0.8)  # Centré dans l'image
            y = np.random.uniform(0.1, 0.9)
            z = np.random.uniform(-0.5, 0.5)
            visibility = np.random.uniform(0.7, 1.0)
            landmarks.append(MockLandmark(x, y, z, visibility))
        return landmarks


class MockResults:
    """Simule les résultats de MediaPipe Pose"""
    def __init__(self, pose_detected: bool = True):
        if pose_detected:
            self.pose_landmarks = MockLandmarks(33)
        else:
            self.pose_landmarks = None


class MockPose:
    """Simule le modèle MediaPipe Pose"""
    def __init__(self, static_image_mode: bool = True, model_complexity: int = 2, smooth_landmarks: bool = True):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        # Probabilité que une personne soit détectée (95% par défaut)
        self.detection_rate = 0.95
    
    def process(self, image_rgb: np.ndarray) -> MockResults:
        """Simule le traitement d'une image"""
        # Simuler une détection avec probabilité
        pose_detected = np.random.random() < self.detection_rate
        return MockResults(pose_detected=pose_detected)
    
    def close(self):
        """Fermer le modèle"""
        pass


class MockPoseLandmark:
    """Enum simulé pour les noms des landmarks"""
    @staticmethod
    def get_names():
        return [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
            'LEFT_EAR', 'RIGHT_EAR',
            'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX',
            'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]


class MockSolutions:
    """Simule le module solutions de MediaPipe"""
    
    class Pose:
        def __init__(self, static_image_mode=True, model_complexity=2, smooth_landmarks=True):
            self.instance = MockPose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks
            )
        
        def __call__(self, *args, **kwargs):
            return self.instance
        
        def __enter__(self):
            return self.instance
        
        def __exit__(self, *args):
            self.instance.close()
    
    class PoseLandmark:
        name = "MockPoseLandmark"
        
        @staticmethod
        def __iter__():
            names = MockPoseLandmark.get_names()
            for name in names:
                yield type('Landmark', (), {'name': name})()


class MockMediaPipe:
    """Mock complet du module MediaPipe"""
    
    solutions = MockSolutions()
    
    @staticmethod
    def get_version():
        return "mock-1.0.0"


def get_mediapipe_or_mock():
    """
    Tente d'importer MediaPipe réel, sinon retourne le mock
    """
    try:
        import mediapipe as mp
        # Vérifier que mp.solutions existe
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
            return mp
    except (ImportError, AttributeError):
        pass
    
    # Retourner le mock
    print("⚠️ MediaPipe réel non disponible, utilisation du mock (détection simulée)")
    return MockMediaPipe()
