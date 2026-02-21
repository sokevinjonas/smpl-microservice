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
        'poitrine': { 
            'indices': [3015, 3016, 3017], # Ring points for better fallback
            'description': 'Tour de poitrine'
        },
        'taille': { 
            'indices': [3502, 3503, 3504], 
            'description': 'Tour de taille'
        },
        'bassin': { 
            'indices': [3170, 3171, 3172],
            'description': 'Tour de bassin'
        },
        'cuisse': { 
            'indices': [1010, 1011], 
            'axis_indices': [3170, 1010],
            'description': 'Tour de cuisse'
        },
        'bras': { 
            'indices': [1723, 1724], 
            'axis_indices': [636, 1723],
            'description': 'Tour de bras (biceps)'
        },
        'poignet': { 
            'indices': [2096, 2097],
            'axis_indices': [1723, 2096],
            'description': 'Tour de poignet'
        },
        'cou': { 
            'indices': [3026, 3027],
            'description': 'Tour de cou'
        },
        'mollet': { 
            'indices': [3292, 3293],
            'axis_indices': [1010, 3292],
            'description': 'Tour de mollet'
        },
        'cheville': { 
            'indices': [3307, 3308],
            'axis_indices': [3292, 3307],
            'description': 'Tour de cheville'
        },
        'avant_bras': { 
            'indices': [2145, 2146],
            'axis_indices': [1723, 2096],
            'description': 'Tour d\'avant-bras'
        },
        'entrejambe': { 
            'type': 'distance',
            'indices': [3500, 3387], 
            'description': 'Longueur entrejambe'
        },
        'longueur_manche': { 
            'type': 'distance',
            'indices': [636, 2096], 
            'description': 'Longueur de manche'
        },
        'largeur_epaules': { 
            'type': 'distance',
            'indices': [636, 4110], 
            'description': 'Largeur d\'épaules'
        },
        'genou': { 
            'indices': [1100, 1101],
            'description': 'Tour de genou'
        },
        'tete': { 
            'indices': [411, 412],
            'description': 'Tour de tête'
        },
        'sous_poitrine': { 
            'indices': [3021, 3022],
            'description': 'Tour sous-poitrine'
        },
        'longueur_jambe': { 
            'type': 'distance',
            'indices': [3170, 3387],
            'description': 'Longueur de jambe totale'
        },
        'largeur_pectoral': { 
            'type': 'distance',
            'indices': [3015, 6500],
            'description': 'Largeur pectorale'
        },
        'largeur_dos': { 
            'type': 'distance',
            'indices': [3021, 6510],
            'description': 'Largeur du dos'
        },
        'hauteur_torse': { 
            'type': 'distance',
            'indices': [636, 3502],
            'description': 'Hauteur du torse'
        }
    }

    # Correspondance des noms de mesures demandées (Synonymes & French labels)
    MEASUREMENT_MAPPING = {
        'poitrine': 'poitrine',
        'tour_poitrine': 'poitrine',
        'tour_de_poitrine': 'poitrine',
        'taille': 'taille',
        'tour_taille': 'taille',
        'tour_de_taille': 'taille',
        'ceinture': 'taille',
        'bassin': 'bassin',
        'hanche': 'bassin',
        'tour_de_bassin': 'bassin',
        'cuisse': 'cuisse',
        'bras': 'bras',
        'tour_manche': 'bras',
        'tour_de_manche': 'bras',
        'poignet': 'poignet',
        'cou': 'cou',
        'mollet': 'mollet',
        'cheville': 'cheville',
        'bas': 'cheville',
        'avant_bras': 'avant_bras',
        'genou': 'genou',
        'tete': 'tete',
        'sous_poitrine': 'sous_poitrine',
        'entrejambe': 'entrejambe',
        'long_pantalon': 'entrejambe',
        'longueur_manche': 'longueur_manche',
        'long_manche': 'longueur_manche',
        'largeur_epaules': 'largeur_epaules',
        'epaule': 'largeur_epaules',
        'longueur_jambe': 'longueur_jambe',
        'long_jupe': 'longueur_jambe',
        'long_robe': 'longueur_jambe',
        'long_chemise': 'hauteur_torse',
        'long_camisole': 'hauteur_torse',
        'hauteur_torse': 'hauteur_torse',
        'long_taille': 'hauteur_torse',
        'largeur_pectoral': 'largeur_pectoral',
        'largeur_dos': 'largeur_dos',
        'dos': 'largeur_dos',
        'tour_emanchure': 'bras',
        'pinces': 'taille'
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
            best_points = []
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
                # 3.2 Vérifier la taille et centrage
                if len(comp.vertices) > 2:
                    comp_center = np.mean(comp.vertices, axis=0)
                    dist = np.linalg.norm(comp_center - plane_origin)
                    
                    # 3.4 Dimensions précises de la boucle
                    x_min, x_max = np.min(comp.vertices[:, 0]), np.max(comp.vertices[:, 0])
                    width = x_max - x_min
                    
                    # Pour la profondeur (Z), le bounding box est faussé si le plan est incliné.
                    # On utilise l'écart type des points par rapport au centre pour estimer le "rayon" Z
                    z_centered = comp.vertices[:, 2] - comp_center[2]
                    depth_approx = np.mean(np.abs(z_centered)) * 4.0 # Diamètre approx
                    depth = max(0.01, depth_approx)
                    
                    if not limb_axis:
                        # --- ALGORITHME DE RÉPARATION TORSE (Ignorer les bras) ---
                        if width > 0.40 or current_perimeter > 1.1:
                            # Plus permissif sur l'avant du corps
                            threshold_x = 0.17 # 17cm max du centre X
                            torso_pts = np.array([p for p in comp.vertices if abs(p[0]) <= threshold_x])
                            
                            if len(torso_pts) > 10:
                                from scipy.spatial import ConvexHull
                                try:
                                    pts_2d = torso_pts[:, [0, 2]]
                                    hull = ConvexHull(pts_2d)
                                    hull_peri = sum(np.linalg.norm(pts_2d[s[0]] - pts_2d[s[1]]) for s in hull.simplices)
                                    current_perimeter = hull_peri
                                    
                                    x_min, x_max = np.min(torso_pts[:, 0]), np.max(torso_pts[:, 0])
                                    width = x_max - x_min
                                    comp_center = np.mean(torso_pts, axis=0)
                                except Exception as e:
                                    pass

                    # 3.5 Compacité (Roundness) recalculée avec les vraies dimensions
                    area_ellipse = (width * depth) * np.pi / 4.0
                    compactness = (4.0 * np.pi * area_ellipse) / (current_perimeter ** 2) if current_perimeter > 0 else 0
                    
                    if limb_axis:
                        # --- FILTRE MEMBRES ---
                        score = dist 
                        # On assouplit légèrement pour ne pas rater les membres
                        if dist < 0.20 and width < 0.35:
                            if score < min_dist_to_origin:
                                min_dist_to_origin = score
                                best_perimeter = current_perimeter
                                best_points = comp.vertices.tolist()
                    else:
                        # --- FILTRE TORSE ---
                        off_center_x = abs(comp_center[0])
                        score = off_center_x * 5.0
                        
                        is_likely_fused = (width > 0.50) or (width / depth > 2.5)
                        if is_likely_fused: score += 20.0
                        
                        if 0.5 < current_perimeter < 1.3:
                             # On demande une compacité raisonnable (0.4 = ellipse assez plate, 1.0 = cercle)
                             if compactness > 0.4:
                                if score < min_dist_to_origin:
                                    min_dist_to_origin = score
                                    best_perimeter = current_perimeter
                                    best_points = comp.vertices.tolist()
            
            # On retourne la valeur brute en Mètres
            return round(best_perimeter, 4), best_points
            
        except Exception as e:
            print(f"Error slicing mesh: {e}")
            return 0.0, []

    def calculate_circumference(self, vertex_indices: List[int], limb_axis: List[int] = None) -> Tuple[float, List]:
        """
        Calcule une circonférence. Essaie le slicing d'abord, sinon fallback sur vertices.
        """
        # Tenter le slicing si on a le mesh et assez de points pour définir une hauteur
        if self.mesh is not None and len(vertex_indices) > 0:
            val, points = self.calculate_slice_circumference(vertex_indices, limb_axis=limb_axis)
            if val > 0:
                return val, points
        
        # Fallback méthode naïve (périmètre zigzag)
        if len(vertex_indices) < 2:
            return 0.0, []

        vertices = self.vertices[vertex_indices]
        total_distance = 0.0

        # Calculer la distance entre vertices consécutifs
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            total_distance += euclidean(v1, v2)

        # Convert m to mm (Actually we remain in meters here, conversion is in app.py)
        return round(total_distance, 4), vertices.tolist()

    def calculate_distance(self, start_idx: int, end_idx: int) -> Tuple[float, List]:
        """
        Calcule la distance géodésique (TODO) ou euclidienne.
        Pour l'instant Euclidienne.
        """
        start = self.vertices[start_idx]
        end = self.vertices[end_idx]
        # Convert m to mm (Actually we remain in meters here, conversion is in app.py)
        dist = round(euclidean(start, end), 4)
        return dist, [start.tolist(), end.tolist()]

    def get_measurement(self, measurement_name: str) -> float:
        """
        Obtient une mesure corporelle.

        Args:
            measurement_name: Nom de la mesure (ex: 'tour_poitrine')

        Returns:
            Valeur de la mesure en mm
        """
        # Normaliser le nom : minuscule, strip et remplacer les espaces par des underscores
        key = measurement_name.lower().strip().replace(' ', '_')

        # Vérifier le cache
        if key in self.measurements_cache:
            return self.measurements_cache[key]

        # Mapper le nom à la mesure
        if key not in self.MEASUREMENT_MAPPING:
            return 0.0

        body_part = self.MEASUREMENT_MAPPING[key]
        part_info = self.BODY_PART_VERTICES.get(body_part, {})

        measurement_value = 0.0
        points = [] # Initialize points list
        part_type = part_info.get('type', 'circumference')

        if part_type == 'circumference' and 'indices' in part_info:
            # Mesure de circonférence
            # NOTE: On utilise calculate_circumference car il gère le fallback si le slicing échoue
            axis_indices = part_info.get('axis_indices')
            measurement_value, _ = self.calculate_circumference(part_info['indices'], limb_axis=axis_indices)
        
        elif part_type == 'distance' and 'indices' in part_info:
            # Mesure de distance entre 2 points (ex: longueur jambe)
            if len(part_info['indices']) >= 2:
                measurement_value, _ = self.calculate_distance(
                    part_info['indices'][0],
                    part_info['indices'][1]
                )
        
        elif 'start' in part_info and 'end' in part_info:
            # Rétrocompatibilité distance
            measurement_value, _ = self.calculate_distance(part_info['start'], part_info['end'])

        # Cacher le résultat
        self.measurements_cache[key] = measurement_value
        return measurement_value

    def get_all_measurements(self, measures_table: List[str] = None) -> Dict:
        """
        Récupère toutes les mesures spécifiées.
        """
        if not measures_table:
            # Mesures par défaut si vide
            measures_table = ['poitrine', 'taille', 'hanche']

        results = {}
        for m_name in measures_table:
            val = self.get_measurement(m_name)
            results[m_name] = val

        return results

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
