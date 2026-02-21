import numpy as np
import torch
from smplx import SMPL
import json

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

def find_indices():
    print(f"Analyse du modèle: {MODEL_PATH}")
    smpl = SMPL(model_path=MODEL_PATH, gender='neutral')
    
    # Pose neutre
    output = smpl(
        betas=torch.zeros(1, 10),
        body_pose=torch.zeros(1, 69),
        global_orient=torch.zeros(1, 3),
        transl=torch.zeros(1, 3)
    )
    verts = output.vertices.detach().numpy()[0]
    
    y_max = np.max(verts[:, 1])
    y_min = np.min(verts[:, 1])
    total_h = y_max - y_min
    
    def get_closest_to_pos(x, y, z):
        dist = np.linalg.norm(verts - np.array([x, y, z]), axis=1)
        return np.argmin(dist)

    # 1. Torso Centers
    print("\n--- Torse (Axe Central X=0) ---")
    torso_targets = {
        'cou': y_min + 0.84 * total_h,
        'poitrine': y_min + 0.74 * total_h,
        'taille': y_min + 0.62 * total_h,
        'hanches': y_min + 0.52 * total_h
    }
    for name, y in torso_targets.items():
        idx = get_closest_to_pos(0, y, 0)
        v = verts[idx]
        print(f"{name.capitalize()}: Index {idx} at [X={v[0]:.3f}, Y={v[1]:.3f}, Z={v[2]:.3f}]")

    # 2. Members (Using X symmetry)
    print("\n--- Membres (Gauche X > 0, Droite X < 0) ---")
    
    # Épaule (Shoulder) - Wide part of torso
    shoulder_y = y_min + 0.82 * total_h
    shoulder_x = 0.18 # approx
    l_sh = get_closest_to_pos(shoulder_x, shoulder_y, 0)
    r_sh = get_closest_to_pos(-shoulder_x, shoulder_y, 0)
    print(f"Shoulder L: {l_sh} at {verts[l_sh]}")
    print(f"Shoulder R: {r_sh} at {verts[r_sh]}")

    # Coude (Elbow) - Mid arm
    elbow_y = shoulder_y - 0.15 # approx for A-pose/T-pose
    elbow_x = 0.45
    l_el = get_closest_to_pos(elbow_x, elbow_y, 0)
    r_el = get_closest_to_pos(-elbow_x, elbow_y, 0)
    print(f"Elbow L: {l_el} at {verts[l_el]}")
    print(f"Elbow R: {r_el} at {verts[r_el]}")

    # Poignet (Wrist)
    wrist_y = elbow_y - 0.15
    wrist_x = 0.65
    l_wr = get_closest_to_pos(wrist_x, wrist_y, 0)
    r_wr = get_closest_to_pos(-wrist_x, wrist_y, 0)
    print(f"Wrist L: {l_wr} at {verts[l_wr]}")
    print(f"Wrist R: {r_wr} at {verts[r_wr]}")

    # Genou (Knee)
    knee_y = y_min + 0.28 * total_h
    knee_x = 0.10
    l_kn = get_closest_to_pos(knee_x, knee_y, 0)
    r_kn = get_closest_to_pos(-knee_x, knee_y, 0)
    print(f"Knee L: {l_kn} at {verts[l_kn]}")
    print(f"Knee R: {r_kn} at {verts[r_kn]}")

    # Cheville (Ankle)
    ankle_y = y_min + 0.05 * total_h
    l_an = get_closest_to_pos(knee_x, ankle_y, 0)
    print(f"Ankle L: {l_an} at {verts[l_an]}")

if __name__ == "__main__":
    find_indices()
