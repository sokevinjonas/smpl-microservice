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
        'poitrine': { # Tour de poitrine
            'indices': [3015], # Verified Chest Level
            'description': 'Tour de poitrine'
        },
        'taille': { # Tour de taille
            'indices': [3502], # Verified Waist Level
            'description': 'Tour de taille'
        },
        'bassin': { # Tour de bassin/hanches
            'indices': [3170], # Verified Hips Level
            'description': 'Tour de bassin'
        },
        'cuisse': { # Tour de cuisse (Cuisse Gauche)
            'indices': [1010], 
            'axis_indices': [3170, 1010], # Hips -> Knee
            'description': 'Tour de cuisse'
        },
        'bras': { # Tour de bras (Bicep Gauche)
            'indices': [1723], 
            'axis_indices': [636, 1723], # Shoulder -> Elbow
            'description': 'Tour de bras (biceps)'
        },
        'poignet': { # Tour de poignet (Gauche)
            'indices': [2096],
            'axis_indices': [1723, 2096], # Elbow -> Wrist
            'description': 'Tour de poignet'
        }
    }

    # Correspondance des noms de mesures demandées (Exact user strings normalized)
    MEASUREMENT_MAPPING = {
        'poitrine': 'poitrine',
        'tour_poitrine': 'poitrine',
        'taille': 'taille',
        'tour_taille': 'taille',
        'bassin': 'bassin',
        'hanche': 'bassin',
        'cuisse': 'cuisse',
        'bras': 'bras',
        'poignet': 'poignet',
        'tour_manche': 'bras',
        'tour de manche': 'bras'
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
        # Cou: 4788 (Verified Neck Center)
        neck_idx = 4788
        
        # Bassin: 3170 (Verified Hips Center)
        pelvis_idx = 3170
        
        pelvis_pos = self.vertices[pelvis_idx]
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
            
            # Pour le torse, on veut rester proche de l'axe central X=0 (SMPL est centré sur X)
            # plane_origin est le centre des points de repère (landmarks)
            
            for comp in components:
                # 3.1 Calcul périmètre
                current_perimeter = 0.0
                if hasattr(comp, 'length'):
                    current_perimeter = comp.length
                else:
                    for entity in comp.entities:
                        pts = comp.vertices[entity.points]
                        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                        current_perimeter += np.sum(dists)
                
                # Ignorer les bruits (périmètre trop petit)
                if current_perimeter < 0.05: # < 5cm
                    continue

                # 3.2 Vérifier la taille et centrage
                if len(comp.vertices) > 0:
                    comp_center = np.mean(comp.vertices, axis=0)
                    dist = np.linalg.norm(comp_center - plane_origin)
                    
                    # Calculer l'étendue X (largeur)
                    x_min, x_max = np.min(comp.vertices[:, 0]), np.max(comp.vertices[:, 0])
                    width = x_max - x_min
                    
                    if limb_axis:
                        # --- FILTRE MEMBRES ---
                        # On doit être TRES proche du landmark central
                        score = dist 
                        max_allowed_dist = 0.15 
                        max_allowed_width = 0.35 # Une cuisse ou un bras ne fait pas 35cm de large
                        
                        if dist < max_allowed_dist and width < max_allowed_width:
                            if score < min_dist_to_origin:
                                min_dist_to_origin = score
                                best_perimeter = current_perimeter
                    else:
                        # --- FILTRE TORSE ---
                        # Le torse doit être centré en X. On pénalise les bras.
                        off_center_penalty = abs(comp_center[0]) * 5.0 
                        score = dist + off_center_penalty
                        
                        # Limites physiques réalistes pour un torse humain (Slicing SMPL)
                        # Si width > 0.5m, on a probablement inclu les bras/bras collés
                        if width > 0.48: 
                             score += 2.0 # Forte pénalité si trop large
                        
                        # Si on a plusieurs boucles (torse + 2 bras), on veut la plus grosse mais centrée
                        # Heuristique : Score inversement proportionnel à la taille si centré
                        # On cherche le compromisidéal entre "Grosse boucle" et "Centré"
                        if abs(comp_center[0]) < 0.1 and current_perimeter > 0.5:
                             # C'est probablement le torse
                             if current_perimeter < 1.35: # Limite haute raisonnable
                                 score -= 1.0 # Bonus de confiance
                        
                        if dist < 0.35 and score < min_dist_to_origin:
                            min_dist_to_origin = score
                            best_perimeter = current_perimeter
            
            # On retourne la valeur brute en Mètres
            return round(best_perimeter, 4)
            
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

        # Convert m to mm (Actually we remain in meters here, conversion is in app.py)
        return round(total_distance, 4)

    def calculate_distance(self, start_idx: int, end_idx: int) -> float:
        """
        Calcule la distance géodésique (TODO) ou euclidienne.
        Pour l'instant Euclidienne.
        """
        start = self.vertices[start_idx]
        end = self.vertices[end_idx]
        # Convert m to mm (Actually we remain in meters here, conversion is in app.py)
        return round(euclidean(start, end), 4)

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
