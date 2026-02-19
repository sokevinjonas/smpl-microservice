import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import euclidean
import trimesh


class MeshMeasurements:
    """
    Calcule les mensurations corporelles sur un mesh 3D SMPL.
    """

    # Correspondance des keypoints SMPL avec les mesures corporelles (Tailoring)
    # Indices approximatifs basés sur la topologie SMPL
    BODY_PART_VERTICES = {
        'dos': { # Largeur carrure dos
            'left': 2865, 'right': 6263,
            'description': 'Largeur carrure dos (acromion à acromion arrière)'
        },
        'epaule': { # Longueur épaule
            'start': 5588, 'end': 4847, # Cou -> Acromion
            'description': 'Longueur épaule (cou à acromion)'
        },
        'poitrine': { # Tour de poitrine
            'indices': list(range(3000, 3020)) + list(range(6500, 6520)), # TODO: Indices précis à valider
            'description': 'Tour de poitrine (niveau mamelons)'
        },
        'long_manche': { # Longueur manche
            'start': 4847, 'end': 5361, # Acromion -> Poignet
            'description': 'Longueur manche (acromion au poignet)'
        },
        'tour_manche': { # Tour de bras (biceps)
            'indices': [4964, 4965, 4966, 4967], # Ring approximatif biceps
            'axis_indices': [4847, 5035], # Acromion -> Coude (pour l'orientation de la coupe)
            'description': 'Tour de manche (biceps)'
        },
        'long_taille': { # Hauteur taille devant (épaule -> taille)
            'start': 5588, 'end': 3500, # Cou -> Nombril/Taille
            'description': 'Longueur taille devant'
        },
        'tour_taille': { # Tour de taille (plus fin)
            'indices': list(range(3500, 3510)) + list(range(6800, 6810)),
            'description': 'Tour de taille'
        },
        'pinces': { # Ecart poitrine
            'left': 3005, 'right': 6505, # Mamelons approx
            'description': 'Ecart poitrine (mamelon à mamelon)'
        },
        'long_camisole': { # Longueur haut (chafop?)
            'start': 5588, 'end': 660, # Cou -> Bassin
            'description': 'Longueur camisole (cou au bassin)'
        },
        'long_robe': { # Longueur totale
            'start': 5588, 'end': 6775, # Cou -> Cheville/Sol
            'description': 'Longueur robe (cou au bas)'
        },
        'long_chemise': { # Longueur chemise
            'start': 5588, 'end': 660, # Cou -> Mi-fesses (similaire camisole pour l'instant)
            'description': 'Longueur chemise'
        },
        'ceinture': { # Tour de ceinture (petites hanches)
            'indices': [3500], # Placeholder ring
            'description': 'Tour de ceinture'
        },
        'bassin': { # Tour de bassin/hanches
            'indices': list(range(600, 620)) + list(range(3600, 3620)),
            'description': 'Tour de bassin'
        },
        'cuisse': { # Tour de cuisse
            'indices': [1000, 1001, 1002, 1003], # Ring cuisse
            'axis_indices': [620, 1100], # Hanche -> Genou (approx)
            'description': 'Tour de cuisse'
        },
        'genou': { # Tour de genou
            'indices': [2000, 2001, 2002, 2003], # Ring genou
            'axis_indices': [1100, 2100], # Haut genou -> Bas genou
            'description': 'Tour de genou'
        },
        'long_jupe': { # Taille -> Genou/Bas
            'start': 3500, 'end': 2000, # Taille -> Genou (pour minijupe? ou sol?)
            'description': 'Longueur jupe (taille au bas)'
        },
        'long_pantalon': { # Taille -> Cheville ext
            'start': 3500, 'end': 6775,
            'description': 'Longueur pantalon (taille à cheville)'
        },
        'bas': { # Tour de cheville
            'indices': [6775, 6776, 6777],
            'axis_indices': [2100, 6700], # Genou -> Cheville (approx) pour orientation
            'description': 'Tour de cheville (bas)'
        },
        'poignet': { # Tour de poignet
            'indices': [5361, 5362, 5363],
            'axis_indices': [5035, 5361], # Coude -> Poignet (pour orientation)
            'description': 'Tour de poignet'
        },
        'tour_emanchure': { # Tour d'emmanchure
            'indices': [4847, 4848, 4849], # Ring épaule
            'description': 'Tour d\'emmanchure'
        }
    }

    # Correspondance des noms de mesures demandées (Exact user strings normalized)
    MEASUREMENT_MAPPING = {
        'dos': 'dos',
        'epaule': 'epaule',
        'poitrine': 'poitrine',
        'long manche': 'long_manche',
        'tour de manche': 'tour_manche',
        'long taille': 'long_taille',
        'tour taille': 'tour_taille',
        'pinces': 'pinces',
        'long camisole': 'long_camisole',
        'long robe': 'long_robe',
        'long chemise': 'long_chemise',
        'ceinture': 'ceinture',
        'bassin': 'bassin',
        'cuisse': 'cuisse',
        'genou': 'genou',
        'long jupe': 'long_jupe',
        'long pantalon': 'long_pantalon',
        'bas': 'bas',
        'poignet': 'poignet',
        'tour emanchure': 'tour_emanchure',
        
        # Legacy/Fallback mapping
        'tour_poitrine': 'poitrine',
        'chest_circumference': 'chest', # Keep legacy just in case
        'taille': 'tour_taille',
        'hanche': 'bassin',
        'longueur_bras': 'long_manche',
        'arm_length': 'arm_length',
        'leg_length': 'leg_length',
        'shoulder_width': 'shoulder_width'
    }

    def __init__(self, smpl_vertices: np.ndarray, smpl_faces: np.ndarray = None):
        """
        Initialise le calculateur de mensurations.

        Args:
            smpl_vertices: Array de vertices du mesh SMPL (n_vertices, 3)
            smpl_faces: Array de faces (n_faces, 3) pour le slicing (optionnel mais recommandé)
        """
        self.vertices = smpl_vertices
        self.faces = smpl_faces
        self.measurements_cache = {}
        
        # Init Trimesh object if faces provided
        self.mesh = None
        if self.faces is not None:
             try:
                 self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
             except Exception as e:
                 print(f"Warning: Could not create Trimesh object: {e}")

    def _get_body_axis(self) -> np.ndarray:
        """
        Calcule l'axe vertical du corps (Vecteur Bassin -> Cou).
        Permet de couper perpendiculairement au corps même si le mesh est incliné.
        """
        # Indices basés sur BODY_PART_VERTICES
        # Cou: 5588 (début épaule)
        neck_idx = 5588
        
        # Bassin: Moyenne des indices de 'bassin'
        pelvis_indices = self.BODY_PART_VERTICES['bassin']['indices']
        pelvis_pos = np.mean(self.vertices[pelvis_indices], axis=0)
        
        neck_pos = self.vertices[neck_idx]
        
        axis = neck_pos - pelvis_pos
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.array([0, 1, 0]) # Fallback Y-up
            
        return axis / norm

    def calculate_slice_circumference(self, landmark_indices: List[int], limb_axis: List[int] = None) -> float:
        """
        Calcule la circonférence en coupant le mesh.
        - Si limb_axis est None: Utilise l'axe vertical du corps (pour Torse).
        - Si limb_axis est [start, end]: Utilise le vecteur (start->end) comme normale (pour Bras/Jambes).
        """
        if self.mesh is None or not landmark_indices:
            return 0.0

        # 1. Définir le plan de coupe
        landmarks = self.vertices[landmark_indices]
        plane_origin = np.mean(landmarks, axis=0) # Centre de la coupe
        
        # Normale du plan
        if limb_axis and len(limb_axis) == 2:
            # Axe local du membre (ex: Coude -> Poignet)
            p1 = self.vertices[limb_axis[0]]
            p2 = self.vertices[limb_axis[1]]
            axis = p2 - p1
            norm = np.linalg.norm(axis)
            plane_normal = axis / norm if norm > 0 else np.array([0, 1, 0])
        else:
            # Axe global du corps (pour le torse)
            plane_normal = self._get_body_axis()
        
        try:
            # 2. Couper le mesh
            slice_section = self.mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            
            if slice_section is None:
                return 0.0
            
            # 3. Gérer les boucles multiples
            # Problème : Si on coupe le bras, la coupe peut aussi traverser le torse (si le plan est infini).
            # Solution : Garder la boucle la plus proche du plane_origin (le centre des landmarks du membre)
            
            try:
                components = slice_section.split()
            except Exception:
                components = [slice_section]

            best_perimeter = 0.0
            min_dist_to_origin = float('inf')
            
            for comp in components:
                # Calcul périmètre
                current_perimeter = 0.0
                if hasattr(comp, 'length'):
                    current_perimeter = comp.length
                else:
                    for entity in comp.entities:
                        pts = comp.vertices[entity.points]
                        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                        current_perimeter += np.sum(dists)
                
                # Vérifier si c'est la bonne composante (la plus proche des landmarks initiaux)
                # On prend le centre de la composante
                if len(comp.vertices) > 0:
                    comp_center = np.mean(comp.vertices, axis=0)
                    dist = np.linalg.norm(comp_center - plane_origin)
                    
                    # Heuristique: Si on coupe un bras, la section du torse sera loin de plane_origin (qui est sur le bras)
                    if dist < min_dist_to_origin:
                        min_dist_to_origin = dist
                        best_perimeter = current_perimeter
            
            return round(best_perimeter, 2)
            
        except Exception as e:
            print(f"Error slicing mesh: {e}")
            return 0.0

    def calculate_circumference(self, vertex_indices: List[int]) -> float:
        """
        Calcule une circonférence. Essaie le slicing d'abord, sinon fallback sur vertices.
        """
        # Tenter le slicing si on a le mesh et assez de points pour définir une hauteur
        if self.mesh is not None and len(vertex_indices) > 0:
            val = self.calculate_slice_circumference(vertex_indices)
            if val > 0:
                return val
        
        # Fallback méthode naïve (périmètre zigzag)
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
        Calcule la distance géodésique (TODO) ou euclidienne.
        Pour l'instant Euclidienne.
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
            # Gestion de l'orientation spécifique (pour bras/jambes)
            axis_indices = part_info.get('axis_indices')
            measurement_value = self.calculate_slice_circumference(part_info['indices'], limb_axis=axis_indices)
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
    chest = measurements.get('poitrine', measurements.get('tour_poitrine', 0))
    waist = measurements.get('tour_taille', measurements.get('taille', 0))
    hip = measurements.get('bassin', measurements.get('hanche', 0))

    if chest > 0 and waist > 0 and waist > chest:
        errors.append("Incohérence: la taille > tour de poitrine")
    if waist > 0 and hip > 0 and hip > waist * 1.5:
        errors.append("Incohérence: les hanches sont disproportionnées")

    return len(errors) == 0, errors


def export_mesh_to_obj(vertices: np.ndarray, faces: np.ndarray) -> str:
    """
    Exporte le mesh au format Wavefront OBJ (string).
    
    Args:
        vertices: (V, 3) float array
        faces: (F, 3) int array
        
    Returns:
        String content of the OBJ file
    """
    obj_lines = []
    obj_lines.append("# SMPL Microservice Export")
    
    # Vertices
    for v in vertices:
        obj_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        
    # Faces (OBJ indices are 1-based)
    for f in faces:
        obj_lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
        
    return "\n".join(obj_lines)
