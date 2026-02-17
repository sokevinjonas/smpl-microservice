import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import euclidean


class MeshMeasurements:
    """
    Calcule les mensurations corporelles sur un mesh 3D SMPL.
    """

    # Correspondance des keypoints SMPL avec les mesures corporelles
    # Index des vertices SMPL pour différentes parties du corps
    BODY_PART_VERTICES = {
        'chest': {
            'indices': list(range(1300, 1320)) + list(range(4500, 4520)),  # Approximatif
            'description': 'Tour de poitrine'
        },
        'waist': {
            'indices': list(range(800, 820)) + list(range(3900, 3920)),
            'description': 'Taille'
        },
        'hip': {
            'indices': list(range(600, 620)) + list(range(3600, 3620)),
            'description': 'Hanche'
        },
        'arm_length': {
            'start': 1631,  # Épaule droite (approximatif)
            'end': 1659,    # Poignet droit (approximatif)
            'description': 'Longueur du bras'
        },
        'leg_length': {
            'start': 3468,  # Hanche droite (approximatif)
            'end': 3431,    # Cheville droite (approximatif)
            'description': 'Longueur de la jambe'
        },
        'shoulder_width': {
            'left': 1613,   # Épaule gauche
            'right': 1631,  # Épaule droite
            'description': 'Largeur des épaules'
        }
    }

    # Correspondance des noms de mesures demandées
    MEASUREMENT_MAPPING = {
        'tour_poitrine': 'chest',
        'chest_circumference': 'chest',
        'poitrine': 'chest',
        'taille': 'waist',
        'waist': 'waist',
        'hanche': 'hip',
        'hip': 'hip',
        'hanches': 'hip',
        'longueur_bras': 'arm_length',
        'arm_length': 'arm_length',
        'longueur_jambe': 'leg_length',
        'leg_length': 'leg_length',
        'largeur_epaules': 'shoulder_width',
        'shoulder_width': 'shoulder_width'
    }

    def __init__(self, smpl_vertices: np.ndarray):
        """
        Initialise le calculateur de mensurations.

        Args:
            smpl_vertices: Array de vertices du mesh SMPL (n_vertices, 3)
        """
        self.vertices = smpl_vertices
        self.measurements_cache = {}

    def calculate_circumference(self, vertex_indices: List[int]) -> float:
        """
        Calcule une circonférence à partir d'un ensemble de vertices.

        Args:
            vertex_indices: Indices des vertices formant un anneau

        Returns:
            Circonférence en unités (millimètres généralement)
        """
        if len(vertex_indices) < 2:
            return 0.0

        vertices = self.vertices[vertex_indices]
        total_distance = 0.0

        # Calculer la distance entre vertices consécutifs
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            total_distance += euclidean(v1, v2)

        return round(total_distance, 2)

    def calculate_distance(self, start_idx: int, end_idx: int) -> float:
        """
        Calcule la distance entre deux points du mesh.

        Args:
            start_idx: Index du vertex de départ
            end_idx: Index du vertex d'arrivée

        Returns:
            Distance en unités
        """
        start = self.vertices[start_idx]
        end = self.vertices[end_idx]
        return round(euclidean(start, end), 2)

    def get_measurement(self, measurement_name: str) -> float:
        """
        Obtient une mesure corporelle.

        Args:
            measurement_name: Nom de la mesure (ex: 'tour_poitrine')

        Returns:
            Valeur de la mesure en mm
        """
        # Normaliser le nom
        key = measurement_name.lower().strip()

        # Vérifier le cache
        if key in self.measurements_cache:
            return self.measurements_cache[key]

        # Mapper le nom à la mesure
        if key not in self.MEASUREMENT_MAPPING:
            return 0.0

        body_part = self.MEASUREMENT_MAPPING[key]
        part_info = self.BODY_PART_VERTICES.get(body_part, {})

        measurement_value = 0.0

        if 'indices' in part_info:
            # Mesure de circonférence
            measurement_value = self.calculate_circumference(part_info['indices'])
        elif 'start' in part_info and 'end' in part_info:
            # Mesure de distance
            measurement_value = self.calculate_distance(
                part_info['start'],
                part_info['end']
            )
        elif 'left' in part_info and 'right' in part_info:
            # Mesure de largeur
            measurement_value = self.calculate_distance(
                part_info['left'],
                part_info['right']
            )

        # Cacher le résultat
        self.measurements_cache[key] = measurement_value
        return measurement_value

    def get_all_measurements(self, measurement_names: List[str]) -> Dict[str, float]:
        """
        Calcule plusieurs mensurations à la fois.

        Args:
            measurement_names: Liste des noms de mesures

        Returns:
            Dict avec {nom_mesure: valeur}
        """
        measurements = {}
        for name in measurement_names:
            measurements[name] = self.get_measurement(name)
        return measurements

    def scale_measurements(self, scale_factor: float, measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Adapte les mensurations avec un facteur d'échelle.

        Args:
            scale_factor: Facteur d'échelle
            measurements: Dict des mensurations

        Returns:
            Dict des mensurations adaptées
        """
        return {
            name: round(value * scale_factor, 2)
            for name, value in measurements.items()
        }


def create_measurement_report(measurements: Dict[str, float]) -> str:
    """
    Crée un rapport lisible des mensurations.

    Args:
        measurements: Dict des mensurations

    Returns:
        Rapport formaté en texte
    """
    report = "=== RAPPORT DE MENSURATIONS ===\n\n"
    for name, value in measurements.items():
        report += f"{name.upper()}: {value} mm\n"
    return report


def validate_measurements(measurements: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Valide que les mensurations sont réalistes.

    Args:
        measurements: Dict des mensurations

    Returns:
        Tuple (is_valid, list of errors)
    """
    errors = []

    # Vérifications basiques
    for name, value in measurements.items():
        if value <= 0:
            errors.append(f"{name}: valeur négative ou zéro ({value})")
        if value > 10000:  # Plus de 10 mètres ?
            errors.append(f"{name}: valeur trop grande ({value})")

    # Vérifications de cohérence
    chest = measurements.get('tour_poitrine', 0)
    waist = measurements.get('taille', 0)
    hip = measurements.get('hanche', 0)

    if chest > 0 and waist > 0 and waist > chest:
        errors.append("Incohérence: la taille > tour de poitrine")
    if waist > 0 and hip > 0 and hip > waist * 1.5:
        errors.append("Incohérence: les hanches sont disproportionnées")

    return len(errors) == 0, errors
