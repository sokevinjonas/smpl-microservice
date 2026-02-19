import numpy as np
import torch
from smplx import SMPL

# Monkeypatch for chumpy/numpy compatibility
try:
    np.bool = np.bool_
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str
    np.str = str
except:
    pass

# Config
MODEL_PATH = 'models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl'

def main():
    print(f"Chargement du modèle: {MODEL_PATH}")
    smpl = SMPL(model_path=MODEL_PATH, gender='neutral')
    
    # Paramètres neutres
    betas = torch.zeros(1, 10)
    body_pose = torch.zeros(1, 69)
    global_orient = torch.zeros(1, 3)
    transl = torch.zeros(1, 3)
    
    output = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
    verts = output.vertices.detach().numpy()[0]
    
    print(f"Modèle généré: {len(verts)} vertices")
    
    # Indices à checker
    indices_to_check = {
        'Cou (5588)': 5588,
        'Acromion Droit (4847)': 4847,
        'Acromion Gauche (4153)': 4153, 
        'Bassin/Nombril (3500)': 3500,
        'Genou (2000)': 2000,
        'Cheville (6775)': 6775,
        'Poignet (5361)': 5361,
        'Coude (5035)': 5035
    }
    
    # Calculer hauteur totale (max Y - min Y)
    min_y = np.min(verts[:, 1])
    max_y = np.max(verts[:, 1])
    height = max_y - min_y
    print(f"Hauteur totale du mesh neutre: {height:.2f}m (Min Y: {min_y:.2f}, Max Y: {max_y:.2f})")
    
    print("\n--- Coordonnées des Vertices ---")
    for name, idx in indices_to_check.items():
        if idx < len(verts):
            v = verts[idx]
            print(f"{name}: [X={v[0]:.2f}, Y={v[1]:.2f}, Z={v[2]:.2f}]")
        else:
            print(f"{name}: Index hors limites !")

    # Check distance Epaule (Cou -> Acromion Droit)
    p1 = verts[5588]
    p2 = verts[4847]
    dist = np.linalg.norm(p1 - p2)
    print(f"\nDistance Cou -> Acromion (Epaule): {dist:.2f}m")

    # Check distance Poignet (Largeur ?)
    # Poignet indices: 5361, 5362, 5363
    p_wrist = verts[[5361, 5362, 5363]]
    center_wrist = np.mean(p_wrist, axis=0)
    print(f"Centre Poignet (Moyenne 5361-5363): {center_wrist}")
    
    # Check orientation bras (Coude -> Poignet)
    p_elbow = verts[5035]
    arm_vec = center_wrist - p_elbow
    print(f"Vecteur Avant-Bras (Coude->Poignet): {arm_vec}")

if __name__ == "__main__":
    main()
