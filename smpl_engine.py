import numpy as np

# Patch pour compatibilité NumPy >= 1.24 avec les vieilles libs (chumpy/smplx)
for name, target in [('bool', bool), ('int', int), ('float', float), 
                    ('complex', complex), ('object', object), ('str', str),
                    ('unicode', str), ('long', int)]:
    if not hasattr(np, name):
        setattr(np, name, target)
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

from typing import Dict
from pathlib import Path
import torch
import random
from typing import Dict, Tuple
from smplx import SMPL  # import direct de la classe
from utils.volume_utils import calculate_mesh_volume_tensor

class SMPLEngine:
    def __init__(self, model_dir: str = './models'):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.smpl_model = None
        self.current_gender = None
        print(f"SMPLEngine initialisé sur device: {self.device}")

    def load_smpl_model(self, gender: str = 'neutral') -> bool:
        """
        Charge le modèle SMPL à partir du fichier correspondant au genre.
        Les fichiers attendus dans models/smpl/ :
          - neutral : SMPL_NEUTRAL.npz
          - female  : SMPL_FEMALE.npz
          - male    : SMPL_MALE.npz
          - male    : basicmodel_m_lbs_10_207_0_v1.1.0.pkl
        """
        # Associer le genre au nom de fichier
        filenames = {
            'neutral': 'SMPL_NEUTRAL.npz',
            'female':  'SMPL_FEMALE.npz',
            'male':    'SMPL_MALE.npz'
        }
        if gender not in filenames:
            raise ValueError(f"Genre inconnu : {gender}. Choisissez parmi {list(filenames.keys())}")

        model_file = self.model_dir / 'smpl' / filenames[gender]
        if not model_file.exists():
            # Essayer aussi l'extension .pkl (ancien format)
            pkl_file = model_file.with_name(model_file.stem + '.pkl')
            if pkl_file.exists():
                model_file = pkl_file
            else:
                # Essayer les noms "basicmodel_..."
                alt_names = {
                    'neutral': 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
                    'female':  'basicmodel_f_lbs_10_207_0_v1.1.0.pkl',
                    'male':    'basicmodel_m_lbs_10_207_0_v1.1.0.pkl'
                }
                legacy_file = self.model_dir / 'smpl' / alt_names.get(gender, '')
                if legacy_file.exists():
                    model_file = legacy_file

        if not model_file.exists():
            print(f"⚠️  Fichier modèle non trouvé : {model_file}")
            print("   Utilisation du modèle léger synthétique.")
            self.smpl_model = self._create_lightweight_smpl()
            return False

        try:
            self.smpl_model = SMPL(
                model_path=str(model_file),
                batch_size=1,
                create_transl=True,
                device=self.device
            )
            print(f"✓ Modèle SMPL ({gender}) chargé avec succès depuis {model_file.name}")
            self.current_gender = gender
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle SMPL : {e}")
            print("   Utilisation du modèle léger synthétique.")
            self.smpl_model = self._create_lightweight_smpl()
            return False

    def _create_lightweight_smpl(self):
        class LightweightSMPL:
            def __init__(self, device='cpu'):
                self.device = device
                self.faces = self._get_smpl_faces()
                self.fixed_vertices = torch.zeros(6890, 3, device=device)

            def _get_smpl_faces(self):
                faces = []
                for i in range(0, 6890 - 2, 3):
                    faces.append([i, i + 1, i + 2])
                return np.array(faces, dtype=np.uint32)

            def __call__(self, betas, body_pose, global_orient, transl, return_verts=True):
                batch_size = betas.shape[0]
                vertices = self.fixed_vertices.unsqueeze(0).repeat(batch_size, 1, 1)
                vertices = vertices + transl.unsqueeze(1)
                class Output:
                    pass
                output = Output()
                output.vertices = vertices
                output.faces = self.faces
                return output

        return LightweightSMPL(device=self.device)

    def estimate_smpl_params_from_keypoints(self, keypoints: np.ndarray) -> Dict:
        batch_size = 1
        betas = torch.zeros(batch_size, 10, device=self.device)
        body_pose = torch.zeros(batch_size, 69, device=self.device)
        global_orient = torch.zeros(batch_size, 3, device=self.device)
        transl = torch.zeros(batch_size, 3, device=self.device)

        return {
            'betas': betas.detach().cpu().numpy()[0],
            'body_pose': body_pose.detach().cpu().numpy()[0],
            'global_orient': global_orient.detach().cpu().numpy()[0],
            'translation': transl.detach().cpu().numpy()[0]
        }

    def generate_mesh(self, smpl_params: Dict) -> np.ndarray:
        if self.smpl_model is None:
            if not self.load_smpl_model():
                return None

        try:
            betas = torch.from_numpy(smpl_params['betas']).unsqueeze(0).float().to(self.device)
            body_pose = torch.from_numpy(smpl_params['body_pose']).unsqueeze(0).float().to(self.device)
            global_orient = torch.from_numpy(smpl_params['global_orient']).unsqueeze(0).float().to(self.device)
            transl = torch.from_numpy(smpl_params['translation']).unsqueeze(0).float().to(self.device)

            output = self.smpl_model(
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                transl=transl,
                return_verts=True
            )
            vertices = output.vertices.detach().cpu().numpy()[0]
            print(f"✓ Mesh généré: {len(vertices)} vertices")
            return vertices
        except Exception as e:
            print(f"❌ Erreur lors de la génération du mesh: {e}")
            return None

    def get_metrics_from_betas(self, betas_np: np.ndarray, gender: str = 'neutral') -> Dict:
        """
        Calcule la taille (m) et le poids (kg) théoriques pour un set de betas donné.
        """
        if self.smpl_model is None or self.current_gender != gender:
            self.load_smpl_model(gender)
            
        with torch.no_grad():
            betas = torch.from_numpy(betas_np).unsqueeze(0).float().to(self.device)
            # Pose T-pose (zeros)
            output = self.smpl_model(betas=betas, return_verts=True)
            vertices = output.vertices[0]
            
            # 1. Height
            y_min = torch.min(vertices[:, 1])
            y_max = torch.max(vertices[:, 1])
            height = (y_max - y_min).item()
            
            # 2. Weight
            faces = torch.tensor(np.array(self.smpl_model.faces, dtype=np.int32), dtype=torch.long, device=self.device)
            vol_m3 = calculate_mesh_volume_tensor(vertices, faces)
            weight = vol_m3 * 1010.0 # Masse volumique humaine
            
        return {'height': height, 'weight': weight}

    def get_mesh_faces(self) -> np.ndarray:
        if self.smpl_model is None:
            return None
        try:
            return np.array(self.smpl_model.faces, dtype=np.uint32)
        except:
            return None


    def fit_model_to_multiple_views(self, keypoints_list: list[np.ndarray], image_shapes: list[Tuple[int, int]], target_weight: float = None, target_weight_interval: Tuple[float, float] = None, target_height: float = None, focal_length: float = 1777.8) -> Dict:
        """
        Ajuste le modèle SMPL à plusieurs vues en optimisant pose et shape partagés.
        Suppositions: Vue 0 = Face, Vue 1 = Profil (approx 90 deg)
        Aussi, target_weight (valeur exacte) ou target_weight_interval (min, max) peuvent être fournis pour contraindre le volume 3D.
        target_height (m) est utilisé pour dimensionner correctement le volume lors du calcul du poids.
        Focale: AGORA utilise du 50mm sur 1280x720 (~1777.8 px).
        """
        if self.smpl_model is None or not keypoints_list:
            return None
            
        num_views = len(keypoints_list)
        print(f"🔄 Début fitting Multi-View avec {num_views} vues")
        
        # 1. Préparation des données pour chaque vue
        target_kps_list = []
        weights_list = []
        
        # Indices (identiques pour toutes les vues)
        # Indices (identiques pour toutes les vues)
        # Ajout du nez (MP 0 -> SMPL 24/15?) 
        # Pour SMPL classique (24 joints), le head est joint 15. Nose est souvent approximé par Head.
        mp_indices = [12, 11, 24, 23, 14, 13, 26, 25, 16, 15, 28, 27, 0]
        smpl_indices = [17, 16, 2, 1, 19, 18, 5, 4, 21, 20, 8, 7, 15]
        
        for kps in keypoints_list:
            # Conversion et Inversion Y (MediaPipe -> SMPL Y-up)
            t_kps = torch.tensor(kps[:, :2], dtype=torch.float32, device=self.device)
            t_kps[:, 1] = 1.0 - t_kps[:, 1]
            target_kps_list.append(t_kps)
            
            w = torch.tensor(kps[:, 2], dtype=torch.float32, device=self.device).unsqueeze(1)
            w = torch.clamp(w, 0.0, 1.0) # Sécurité (MediaPipe peut donner >1 ?)
            weights_list.append(w)

        # 2. Paramètres PARTAGÉS (Le corps est unique)
        batch_size = 1
        betas = torch.zeros(batch_size, 10, device=self.device, requires_grad=True)
        body_pose = torch.zeros(batch_size, 69, device=self.device, requires_grad=True)
        # Translation globale du corps (racine)
        transl = torch.zeros(batch_size, 3, device=self.device, requires_grad=True) 
        
        # 3. Paramètres PAR VUE (Rotation globale + Caméra)
        # Chaque vue a sa propre rotation globale du corps (ou position de caméra, c'est équivalent en WeakPersp)
        # Vue 0 (Face) : Rotation ~0
        # Vue 1 (Profil): Rotation ~90 deg sur Y (si profil gauche) ou -90 (profil droit). 
        # On initialise Vue 1 à 90 deg (1.57 rad) sur Y par défaut pour aider la convergence.
        
        global_orients = []
        # Paramètres Perspective : On cherche la translation 3D relative (Z = distance)
        # On garde une liste pour le multi-view même si ici c'est souvent la même caméra
        cam_translations = []
        
        # Focale par défaut pour AGORA (50mm sur 1280px)
        # focal = (50/36)*1280 = 1777.8
        f_px = focal_length
        cx, cy = 0.5, 0.5 # Normalisé
        for i in range(num_views):
            # Rotation
            orient = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
            if i == 1:
                # Initialisation Profil (90 deg autour de Y)
                # Axis-angle: [0, 1.57, 0]
                with torch.no_grad():
                    orient[0, 1] = 1.57
            global_orients.append(orient)
            
            # Caméra Perspective (Translation 3D)
            # On initialise Z à 5.0 mètres (distance typique AGORA)
            c_transl = torch.tensor([[0.0, 0.0, 5.0]], device=self.device, requires_grad=True)
            cam_translations.append(c_transl)
            
        # Optimiseur 1 : Calibration Camera/Position seulement
        optimizer_init = torch.optim.Adam(global_orients + cam_translations + [transl], lr=0.05)
        for i in range(50):
            optimizer_init.zero_grad()
            loss = 0.0
            for v_idx in range(num_views):
                output = self.smpl_model(global_orient=global_orients[v_idx], transl=transl, return_verts=False)
                joints_3d = output.joints[0, smpl_indices, :]
                
                # Projection Perspective
                # On ajoute la translation de caméra spécifique à la vue
                points_cam = joints_3d + cam_translations[v_idx]
                
                # x = (X/Z)*f + cx, y = (Y/Z)*f + cy
                # Ici tout est normalisé [0, 1], donc cx=0.5, cy=0.5
                # La focale doit aussi être normalisée par la largeur de l'image (1280)
                f_norm = f_px / 1280.0
                
                proj_x = (points_cam[:, 0] / points_cam[:, 2]) * f_norm + 0.5
                # Perspective Y: points_cam[:, 1] est SMPL Y (vers le haut).
                # Pour coller au format MediaPipe Y-down (qu'on a déjà inversé en 1-y),
                # on projette directement. Plus Z est grand, plus Y_norm tend vers 0.5.
                proj_y = (points_cam[:, 1] / points_cam[:, 2]) * (f_norm * (1280/720)) + 0.5
                
                pred_2d = torch.stack([proj_x, proj_y], dim=-1)
                loss += torch.mean((pred_2d - target_kps_list[v_idx][mp_indices])**2)
            loss.backward()
            optimizer_init.step()

        # Optimiseur 2 : Tous les paramètres
        optimizer = torch.optim.Adam([
            {'params': [betas], 'lr': 0.05},
            {'params': [body_pose, transl] + global_orients + cam_translations, 'lr': 0.01}
        ])
        
        # Boucle d'optimisation (Perspective)
        for i in range(300):
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Pour chaque vue
            for v_idx in range(num_views):
                # Forward SMPL avec l'orientation de CETTE vue
                # Note: betas/body_pose/transl sont partagés. Seul global_orient change (simule la caméra qui tourne)
                output = self.smpl_model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orients[v_idx],
                    transl=transl,
                    return_verts=False
                )
                joints_3d = output.joints
                
                # Projection Caméra de CETTE vue
                joints_to_fit = joints_3d[0, smpl_indices, :]
                
                # Appliquer params caméra Perspective
                points_cam = joints_to_fit + cam_translations[v_idx]
                f_norm = f_px / 1280.0
                
                proj_x = (points_cam[:, 0] / points_cam[:, 2]) * f_norm + 0.5
                proj_y = (points_cam[:, 1] / points_cam[:, 2]) * (f_norm * (1280/720)) + 0.5
                joints_2d_proj = torch.stack([proj_x, proj_y], dim=-1)
                
                # Loss Reprojection pour CETTE vue
                targets = target_kps_list[v_idx][mp_indices]
                w = weights_list[v_idx][mp_indices]
                
                loss_view = torch.mean(w * (joints_2d_proj - targets)**2)
                total_loss += loss_view * 100.0 # Poids reprojection
            
            # Priors (Régularisation sur les paramètres partagés)
            # On la baisse pour AGORA pour permettre aux betas de s'ajuster
            loss_beta = torch.mean(betas**2) * 0.005
            loss_pose = torch.mean(body_pose**2) * 0.01
            
            # Contrainte de poids (Weight Loss)
            loss_weight = 0.0
            if (target_weight is not None and target_weight > 0) or target_weight_interval is not None:
                # Générer le mesh pour la vue face (0) afin d'évaluer le volume
                # Note: betas/body_pose sont partagés, donc le volume est invariant selon global_orient
                output_vol = self.smpl_model(
                    betas=betas,
                    body_pose=body_pose,
                    global_orient=global_orients[0],
                    transl=transl,
                    return_verts=True
                )
                verts_vol = output_vol.vertices
                faces_vol = torch.tensor(np.array(self.smpl_model.faces, dtype=np.int32), dtype=torch.long, device=self.device)
                
                # Height scaling for correct volume calculation
                vol_scale = 1.0
                if target_height is not None and target_height > 0:
                     y_min = torch.min(verts_vol[:, 1])
                     y_max = torch.max(verts_vol[:, 1])
                     current_height = y_max - y_min
                     if current_height > 0:
                         scale_factor = target_height / current_height
                         vol_scale = scale_factor ** 3
                
                # Calculer le volume matriciel (tensoriel)
                vol_m3 = calculate_mesh_volume_tensor(verts_vol, faces_vol)
                # Estimer le poids à partir du volume basé sur la masse volumique humaine standard (1.01 kg/L)
                estimated_weight = vol_m3 * vol_scale * 1010.0
                
                # Pénalité L2 forte sur l'erreur de poids
                if target_weight_interval is not None:
                     min_w, max_w = target_weight_interval
                     # ReLU loss for interval: 0 penalty if within interval
                     under_loss = torch.relu(min_w - estimated_weight)
                     over_loss = torch.relu(estimated_weight - max_w)
                     loss_weight = (under_loss**2 + over_loss**2) * 10.0
                else:
                     weight_diff = estimated_weight - target_weight
                     loss_weight = torch.mean(weight_diff**2) * 10.0
                
            loss = total_loss + loss_beta + loss_pose + loss_weight
            
            loss.backward()
            optimizer.step()
            
            # Clamping Shape
            with torch.no_grad():
                betas.clamp_(-3.0, 3.0)
                
            if i % 50 == 0:
                print(f"Iter {i}: Loss={loss.item():.4f} (Reproj={total_loss.item():.4f}, Beta={loss_beta.item():.4f}, Pose={loss_pose.item():.4f})")
        
        final_loss = loss.item()
        print(f"✅ Fitting terminé. Loss finale: {final_loss:.4f}")

        # Retourner les paramètres optimisés + LOSS pour validation
        return {
            'betas': betas.detach().cpu().numpy()[0],
            'body_pose': body_pose.detach().cpu().numpy()[0],
            'global_orient': global_orients[0].detach().cpu().numpy()[0], # Vue Face
            'translation': transl.detach().cpu().numpy()[0],
            'loss': final_loss
        }

    def normalize_mesh_height(self, vertices: np.ndarray, target_height: float) -> np.ndarray:
        """
        Redimensionne le mesh pour qu'il ait une hauteur cible exacte.
        """
        if vertices is None or len(vertices) == 0:
            return vertices
            
        # 1. Calculer la hauteur actuelle (Y-axis)
        y_min = np.min(vertices[:, 1])
        y_max = np.max(vertices[:, 1])
        current_height = y_max - y_min
        
        if current_height <= 0:
            return vertices
            
        # 2. Facteur d'échelle
        scale_factor = target_height / current_height
        print(f"📏 Redimensionnement mesh: {current_height:.2f}m -> {target_height:.2f}m (Facteur: {scale_factor:.2f})")
        
        # 3. Appliquer l'échelle (centré sur le bassin ou le sol ? SMPL est centré bassin généralement, mais ici on scale tout simplement)
        # On scale autour de (0,0,0) pour simplifier
        vertices = vertices * scale_factor
        
        return vertices
        
    def process_image(self, image_data_list: list[dict], gender: str = 'neutral', height: float = None, target_weight: float = None, target_weight_interval: Tuple[float, float] = None, focal_length: float = 1777.8) -> Dict:
        """
        Traite une ou plusieurs images (Multi-View).
        image_data_list: Liste de dicts {'image': np.array, 'keypoints': np.array}
        target_weight: Poids corporel cible en kilogrammes.
        target_weight_interval: Intervalle corporel (min, max).
        """
        # Changer de modèle si nécessaire
        if self.current_gender != gender:
            print(f"Changement de modèle vers : {gender}")
            if not self.load_smpl_model(gender):
                return None
        
        # Préparer les listes pour le fitting multi-view
        keypoints_list = [d['keypoints'] for d in image_data_list]
        shapes_list = [d['image'].shape[:2] for d in image_data_list]
        
        try:
            # ⚡ CALIBRATION DE LA TAILLE - Préliminaire pour l'optimisation
            target_height = height
            if target_height is None or target_height <= 0:
                target_height = 1.75 if gender == 'male' else 1.65
                print(f"ℹ️ Pas de taille fournie. Utilisation standard pour {gender} : {target_height}m")
            
            # Appel au nouveau moteur Multi-View avec le paramètre poids et focale
            smpl_params = self.fit_model_to_multiple_views(
                keypoints_list, 
                shapes_list, 
                target_weight=target_weight, 
                target_weight_interval=target_weight_interval,
                target_height=target_height,
                focal_length=focal_length
            )
            
            if smpl_params is None: # Fallback
                 print("⚠️ Echec fitting Multi-View, utilisation paramètres par défaut (basé sur vue 1)")
                 # Fallback: On utilise juste les keypoints de la première vue pour une estimation basique
                 smpl_params = self.estimate_smpl_params_from_keypoints(keypoints_list[0])
                 
        except Exception as e:
            print(f"❌ Erreur Fitting Multi-View: {e}")
            smpl_params = self.estimate_smpl_params_from_keypoints(keypoints_list[0])

        vertices = self.generate_mesh(smpl_params)
        
        # ⚡ CALIBRATION DE LA TAILLE (Appliqué au mesh final)
        if target_height is not None and target_height > 0:
             if vertices is not None:
                  vertices = self.normalize_mesh_height(vertices, target_height)

        if vertices is None:
            return None
            
        return {
            'vertices': vertices,
            'faces': self.get_mesh_faces(),
            'smpl_params': smpl_params,
            'n_vertices': len(vertices),
            'loss': smpl_params.get('loss', 0.0)
        }

    def cleanup(self):
        if self.smpl_model is not None:
            del self.smpl_model
        torch.cuda.empty_cache()


def create_smpl_engine(model_dir: str = './models') -> SMPLEngine:
    engine = SMPLEngine(model_dir)
    engine.load_smpl_model('neutral')  # Charge le modèle neutre par défaut
    return engine