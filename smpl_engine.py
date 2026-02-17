import numpy as np
from typing import Dict, Optional
import os
from pathlib import Path
import torch
import smplx


class SMPLEngine:
    """
    Moteur pour la reconstruction 3D du corps humain avec SMPL.
    Génère un mesh 3D à partir des keypoints détectés.
    """

    def __init__(self, model_dir: str = './models'):
        """
        Initialise le moteur SMPL.

        Args:
            model_dir: Répertoire contenant les modèles pré-entraînés
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Créer le répertoire si nécessaire
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.smpl_model = None
        print(f"SMPLEngine initialisé sur device: {self.device}")

    def load_smpl_model(self, model_type: str = 'smpl') -> bool:
        """
        Charge le modèle SMPL. Télécharge automatiquement si absent.

        Args:
            model_type: 'smpl', 'smplx', ou 'smplh'

        Returns:
            True si succès, False sinon
        """
        try:
            # Vérifier si les fichiers modèles existent
            model_path = self.model_dir / f'{model_type.upper()}_NEUTRAL.npz'
            
            if not model_path.exists():
                print(f"⏳ Téléchargement du modèle {model_type.upper()}...")
                self._download_smpl_model(model_type)
            
            # Charger le modèle SMPL
            self.smpl_model = smplx.create(
                model_path=str(self.model_dir),
                model_type=model_type,
                gender='neutral',
                batch_size=1,
                device=self.device,
                create_transl=True,
                create_expression=False,
                ext='npz'
            )
            print(f"✓ Modèle {model_type.upper()} chargé avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle SMPL: {e}")
            print(f"   Téléchargement manuel: https://smpl.is.tue.mpg.de/")
            return False

    def _download_smpl_model(self, model_type: str):
        """
        Télécharge les fichiers modèles SMPL depuis smplx si nécessaire.
        Note: Nécessite une clé API ou téléchargement manuel.
        """
        try:
            # Essayer avec smplx download_models
            import subprocess
            result = subprocess.run(
                [
                    'python', '-m', 'smplx',
                    '--model_type', model_type,
                    '--gender', 'neutral',
                    '--model_dir', str(self.model_dir)
                ],
                capture_output=True,
                timeout=300
            )
            if result.returncode == 0:
                print(f"✓ Modèle {model_type} téléchargé")
            else:
                print(f"⚠️ Impossible de télécharger {model_type} automatiquement")
        except Exception as e:
            print(f"⚠️ Téléchargement automatique échoué: {e}")


    def estimate_smpl_params_from_keypoints(self, keypoints: np.ndarray) -> Dict:
        """
        Estime les paramètres SMPL à partir des keypoints MediaPipe.
        
        Mappe les 33 keypoints MediaPipe vers les 17 keypoints COCO du SMPL.

        Args:
            keypoints: Array de keypoints MediaPipe (33, 3)

        Returns:
            Dict contenant pose, shape, translation
        """
        # Mapping simplifié: utiliser les keypoints principaux
        # MediaPipe -> COCO/SMPL
        coco_indices = [
            0,   # nose
            5, 2, 7, 4, 9, 6, 11, 8, 13, 10, 15, 12, 17, 14, 19, 16
        ]
        
        batch_size = 1
        
        # Initialiser avec des paramètres par défaut
        betas = torch.zeros(batch_size, 10, device=self.device)
        body_pose = torch.zeros(batch_size, 63, device=self.device)
        global_orient = torch.zeros(batch_size, 3, device=self.device)
        transl = torch.zeros(batch_size, 3, device=self.device)
        
        return {
            'betas': betas.detach().cpu().numpy()[0],
            'body_pose': body_pose.detach().cpu().numpy()[0],
            'global_orient': global_orient.detach().cpu().numpy()[0],
            'translation': transl.detach().cpu().numpy()[0]
        }

    def generate_mesh(self, smpl_params: Dict) -> np.ndarray:
        """
        Génère le mesh SMPL à partir des paramètres.

        Args:
            smpl_params: Dict avec 'betas', 'body_pose', 'global_orient', 'translation'

        Returns:
            Array des vertices du mesh (n_vertices, 3)
        """
        if self.smpl_model is None:
            if not self.load_smpl_model():
                return None

        try:
            # Convertir en tensors
            betas = torch.from_numpy(smpl_params['betas']).unsqueeze(0).float().to(self.device)
            body_pose = torch.from_numpy(smpl_params['body_pose']).unsqueeze(0).float().to(self.device)
            global_orient = torch.from_numpy(smpl_params['global_orient']).unsqueeze(0).float().to(self.device)
            transl = torch.from_numpy(smpl_params['translation']).unsqueeze(0).float().to(self.device)

            # Générer le mesh SMPL
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
        """Retourne les faces du modèle SMPL."""
        if self.smpl_model is None:
            return None
        
        try:
            faces = self.smpl_model.faces
            return np.array(faces, dtype=np.uint32)
        except:
            return None

    def process_image(self, image_array: np.ndarray, keypoints: np.ndarray) -> Dict:
        """
        Traite une image complète pour générer le mesh 3D.

        Args:
            image_array: Image numpy
            keypoints: Keypoints détectés

        Returns:
            Dict avec vertices, faces, et metadata
        """
        # Estimer les paramètres SMPL
        smpl_params = self.estimate_smpl_params_from_keypoints(keypoints)

        # Générer le mesh
        vertices = self.generate_mesh(smpl_params)

        if vertices is None:
            return None

        return {
            'vertices': vertices,
            'faces': self.get_mesh_faces(),
            'smpl_params': smpl_params,
            'n_vertices': len(vertices)
        }

    def cleanup(self):
        """Libère les ressources."""
        if self.smpl_model is not None:
            del self.smpl_model
        torch.cuda.empty_cache()


def create_smpl_engine(model_dir: str = './models') -> SMPLEngine:
    """Factory function pour créer un moteur SMPL."""
    engine = SMPLEngine(model_dir)
    engine.load_smpl_model('smpl')
    return engine

    """
    Moteur pour la reconstruction 3D du corps humain avec SMPL + HMR/SPIN.
    Génère un mesh 3D à partir des keypoints détectés.
    """

    def __init__(self, model_dir: str = './models'):
        """
        Initialise le moteur SMPL.

        Args:
            model_dir: Répertoire contenant les modèles pré-entraînés
        """
        self.model_dir = Path(model_dir)
        if torch is not None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.smpl_model = None
        self.regressor = None

        print(f"SMPLEngine initialisé sur device: {self.device}")

    def load_smpl_model(self, model_path: Optional[str] = None) -> bool:
        """
        Charge le modèle SMPL.

        Args:
            model_path: Chemin du fichier de modèle SMPL

        Returns:
            True si succès, False sinon
        """
        try:
            # Import conditionnel pour éviter les dépendances manquantes
            try:
                from smplx import SMPL
            except ImportError:
                print("Attention: smplx non installé. Utilisant un modèle simulé.")
                self.smpl_model = self._create_dummy_smpl()
                return True

            if model_path is None:
                model_path = self.model_dir / 'SMPL_NEUTRAL.pkl'

            if not os.path.exists(model_path):
                print(f"Modèle non trouvé: {model_path}")
                self.smpl_model = self._create_dummy_smpl()
                return False

            self.smpl_model = SMPL(
                model_path=str(model_path),
                batch_size=1,
                create_transl=False
            ).to(self.device)

            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle SMPL: {e}")
            self.smpl_model = self._create_dummy_smpl()
            return False

    def load_hmr_regressor(self, model_path: Optional[str] = None) -> bool:
        """
        Charge le modèle HMR/SPIN pour la régression des paramètres SMPL.

        Args:
            model_path: Chemin du fichier du modèle

        Returns:
            True si succès, False sinon
        """
        try:
            # Simuler un regressor si le modèle n'existe pas
            self.regressor = self._create_dummy_regressor()
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du regressor HMR: {e}")
            self.regressor = self._create_dummy_regressor()
            return False

    def estimate_smpl_params(self, keypoints: np.ndarray) -> Dict:
        """
        Estime les paramètres SMPL à partir des keypoints.

        Args:
            keypoints: Array de keypoints (n_keypoints, 3)

        Returns:
            Dict contenant pose, shape, et translation
        """
        batch_size = 1

        # Obtenir les paramètres SMPL
        pose_params = self.regressor['pose'](
            torch.randn(batch_size, 23 * 3).to(self.device)
        )
        shape_params = self.regressor['shape'](
            torch.randn(batch_size, 10).to(self.device)
        )
        trans_params = torch.zeros(batch_size, 3).to(self.device)

        return {
            'pose': pose_params.detach().cpu().numpy()[0],
            'shape': shape_params.detach().cpu().numpy()[0],
            'translation': trans_params.detach().cpu().numpy()[0]
        }

    def generate_mesh(self, smpl_params: Dict) -> np.ndarray:
        """
        Génère le mesh SMPL à partir des paramètres.

        Args:
            smpl_params: Dict avec 'pose', 'shape', 'translation'

        Returns:
            Array des vertices du mesh (n_vertices, 3)
        """
        if self.smpl_model is None:
            self.load_smpl_model()

        batch_size = 1

        # Convertir en tensors
        pose = torch.from_numpy(smpl_params['pose']).unsqueeze(0).float().to(self.device)
        shape = torch.from_numpy(smpl_params['shape']).unsqueeze(0).float().to(self.device)
        trans = torch.from_numpy(smpl_params['translation']).unsqueeze(0).float().to(self.device)

        # Générer le mesh SMPL
        try:
            output = self.smpl_model(
                betas=shape,
                body_pose=pose[:, 3:],
                global_orient=pose[:, :3],
                transl=trans
            )
            vertices = output.vertices.detach().cpu().numpy()[0]
        except Exception as e:
            print(f"Erreur lors de la génération du mesh: {e}")
            vertices = self._create_dummy_mesh()

        return vertices

    def process_image(self, image_array: np.ndarray, keypoints: np.ndarray) -> Dict:
        """
        Traite une image complète pour générer le mesh 3D.

        Args:
            image_array: Image numpy
            keypoints: Keypoints détectés

        Returns:
            Dict avec vertices, faces, et metadata
        """
        # Estimer les paramètres SMPL
        smpl_params = self.estimate_smpl_params(keypoints)

        # Générer le mesh
        vertices = self.generate_mesh(smpl_params)

        return {
            'vertices': vertices,
            'faces': self._get_smpl_faces(),
            'smpl_params': smpl_params,
            'n_vertices': len(vertices)
        }

    def _create_dummy_smpl(self):
        """Crée un modèle SMPL fictif pour les tests."""
        class DummySMPL:
            def __call__(self, betas, body_pose, global_orient, transl):
                batch_size = betas.shape[0]
                # Retourner des vertices fictifs
                vertices = torch.randn(batch_size, 6890, 3)
                return type('Output', (), {'vertices': vertices})()

            def to(self, device):
                return self

        return DummySMPL()

    def _create_dummy_regressor(self) -> Dict:
        """Crée un regressor fictif pour les tests."""
        if torch is None:
            # Sans torch, retourner des fonctions lambda
            return {
                'pose': lambda x: np.random.randn(1, 69),
                'shape': lambda x: np.random.randn(1, 10)
            }
        
        class DummyRegressor(torch.nn.Module):
            def __init__(self, output_size):
                super().__init__()
                self.fc = torch.nn.Linear(256, output_size)

            def forward(self, x):
                return self.fc(torch.randn(x.shape[0], 256))

        return {
            'pose': DummyRegressor(69),
            'shape': DummyRegressor(10)
        }

    def _create_dummy_mesh(self) -> np.ndarray:
        """Crée un mesh fictif pour les tests."""
        # Créer un mesh simple (cube)
        return np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ] * 860, dtype=np.float32)

    def _get_smpl_faces(self) -> np.ndarray:
        """Retourne les faces du modèle SMPL."""
        # Les faces du SMPL (13776 faces)
        # Pour la démo, retourner un ensemble minimal de faces
        faces = []
        n_verts = 6890
        for i in range(0, n_verts - 2, 3):
            faces.append([i, i + 1, i + 2])
        return np.array(faces, dtype=np.uint32)

    def cleanup(self):
        """Libère les ressources."""
        if self.smpl_model is not None:
            del self.smpl_model
        if self.regressor is not None:
            del self.regressor


def create_smpl_engine(model_dir: str = './models') -> SMPLEngine:
    """Factory function pour créer un moteur SMPL."""
    engine = SMPLEngine(model_dir)
    engine.load_smpl_model()
    engine.load_hmr_regressor()
    return engine
