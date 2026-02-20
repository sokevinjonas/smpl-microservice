import numpy as np
from typing import Dict
from pathlib import Path
import torch
import random
from smplx import SMPL  # import direct de la classe

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

    def get_mesh_faces(self) -> np.ndarray:
        if self.smpl_model is None:
            return None
        try:
            return np.array(self.smpl_model.faces, dtype=np.uint32)
        except:
            return None

    def fit_model_to_multiple_views(self, keypoints_list: list[np.ndarray], image_shapes: list[Tuple[int, int]]) -> Dict:
        """
        Ajuste le modèle SMPL à plusieurs vues en optimisant pose et shape partagés.
        Supposition: Vue 0 = Face, Vue 1 = Profil (approx 90 deg)
        """
        if self.smpl_model is None or not keypoints_list:
            return None
            
        num_views = len(keypoints_list)
        print(f"🔄 Début fitting Multi-View avec {num_views} vues")
        
        # 1. Préparation des données pour chaque vue
        target_kps_list = []
        weights_list = []
        
        # Indices (identiques pour toutes les vues)
        mp_indices = [12, 11, 24, 23, 14, 13, 26, 25, 16, 15, 28, 27]
        smpl_indices = [17, 16, 2, 1, 19, 18, 5, 4, 21, 20, 8, 7]
        
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
        cam_scales = []
        cam_transs = []
        
        for i in range(num_views):
            # Rotation
            orient = torch.zeros(batch_size, 3, device=self.device, requires_grad=True)
            if i == 1:
                # Initialisation Profil (90 deg autour de Y)
                # Axis-angle: [0, 1.57, 0]
                with torch.no_grad():
                    orient[0, 1] = 1.57
            global_orients.append(orient)
            
            # Caméra (Scale, Trans)
            # Init: Scale 0.6, Centré (0.5, 0.5)
            scale = torch.tensor([0.6], device=self.device, requires_grad=True)
            trans = torch.tensor([[0.5, 0.5]], device=self.device, requires_grad=True)
            cam_scales.append(scale)
            cam_transs.append(trans)
            
        # Optimiseur : Tous les paramètres
        params = [betas, body_pose, transl] + global_orients + cam_scales + cam_transs
        optimizer = torch.optim.Adam(params, lr=0.02)
        
        # Boucle d'optimisation
        for i in range(150): # Un peu plus d'itérations pour le multi-view
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
                joints_to_fit = joints_3d[:, smpl_indices, :]
                joints_2d_proj = joints_to_fit[:, :, :2]
                
                # Appliquer params caméra de CETTE vue
                s = torch.abs(cam_scales[v_idx])
                t = cam_transs[v_idx]
                joints_2d_proj = joints_2d_proj * s + t.unsqueeze(1)
                
                # Loss Reprojection pour CETTE vue
                targets = target_kps_list[v_idx][mp_indices]
                w = weights_list[v_idx][mp_indices]
                
                loss_view = torch.mean(w * (joints_2d_proj - targets)**2)
                total_loss += loss_view * 100.0 # Poids reprojection
            
            # Priors (Régularisation sur les paramètres partagés)
            loss_beta = torch.mean(betas**2) * 0.1
            loss_pose = torch.mean(body_pose**2) * 0.1
            
            loss = total_loss + loss_beta + loss_pose
            
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
        
    def process_image(self, image_data_list: list[dict], gender: str = 'neutral', height: float = None) -> Dict:
        """
        Traite une ou plusieurs images (Multi-View).
        image_data_list: Liste de dicts {'image': np.array, 'keypoints': np.array}
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
            # Appel au nouveau moteur Multi-View
            smpl_params = self.fit_model_to_multiple_views(keypoints_list, shapes_list)
            
            if smpl_params is None: # Fallback
                 print("⚠️ Echec fitting Multi-View, utilisation paramètres par défaut (basé sur vue 1)")
                 # Fallback: On utilise juste les keypoints de la première vue pour une estimation basique
                 smpl_params = self.estimate_smpl_params_from_keypoints(keypoints_list[0])
                 
        except Exception as e:
            print(f"❌ Erreur Fitting Multi-View: {e}")
            smpl_params = self.estimate_smpl_params_from_keypoints(keypoints_list[0])

        vertices = self.generate_mesh(smpl_params)
        
        # ⚡ CALIBRATION DE LA TAILLE
        # Si une taille cible est fournie (ou une taille par défaut), on redimensionne le mesh.
        # Par défaut : Homme 1.75m, Femme 1.65m
        if height is None or height <= 0:
            height = 1.75 if gender == 'male' else 1.65
            print(f"ℹ️ Pas de taille fournie. Utilisation standard pour {gender} : {height}m")
            
        if vertices is not None:
             vertices = self.normalize_mesh_height(vertices, height)

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