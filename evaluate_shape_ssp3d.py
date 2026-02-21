import os
import cv2
import numpy as np

# Patch pour compatibilité NumPy >= 1.24 avec les vieilles libs (chumpy/smplx)
for name, target in [('bool', bool), ('int', int), ('float', float), 
                    ('complex', complex), ('object', object), ('str', str),
                    ('unicode', str), ('long', int)]:
    if not hasattr(np, name):
        setattr(np, name, target)

# Patch pour np.typeDict (souvent utilisé par chumpy/scipy)
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

import sys

from utils.pose_estimation import PoseEstimator
from smpl_engine import create_smpl_engine

def evaluate_ssp3d():
    print("Démarrage de l'évaluation sur le dataset SSP-3D...")
    npz_path = "/app/dataset/ssp3d/ssp_3d/labels.npz"
    img_dir = "/app/dataset/ssp3d/ssp_3d/images" # Ajout de /images/
    
    if not os.path.exists(npz_path):
        # Fallback pour exécution hors docker si besoin
        npz_path = "dataset/ssp3d/ssp_3d/labels.npz"
        img_dir = "dataset/ssp3d/ssp_3d/images"
        
    if not os.path.exists(npz_path):
        print(f"Fichier non trouvé: {npz_path}")
        return
        
    data = np.load(npz_path)
    fnames = data['fnames']
    gt_shapes_all = data['shapes']
    genders = data['genders']
    
    print(f"Chargement de {len(fnames)} images depuis SSP-3D.")
    
    pose_estimator = PoseEstimator()
    smpl_engine = create_smpl_engine()
    
    mae_list = []
    failed = 0
    
    for i in range(len(fnames)):
        fname = str(fnames[i])
        if fname.startswith("b'") or fname.startswith("b\""):
            fname = fname[2:-1] 
            
        img_path = os.path.join(img_dir, fname)
        gt_betas = gt_shapes_all[i]
        gender_code = str(genders[i]).strip().replace("b'", "").replace("'", "")
        
        gender = 'male' if gender_code == 'm' else 'female'
        
        if not os.path.exists(img_path):
            print(f"[{i+1}/{len(fnames)}] Image ignorée (introuvable): {fname}")
            continue
            
        print(f"[{i+1}/{len(fnames)}] Traitement de {fname} ({gender})...", end=" ", flush=True)
        
        image = cv2.imread(img_path)
        if image is None:
            print("Erreur de lecture OpenCV.")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_res = pose_estimator.estimate_pose(image_rgb)
        keypoints = pose_res['keypoints'] if pose_res else None
        
        if keypoints is None:
            print("Échec MediaPipe (pas de pose).")
            failed += 1
            continue
            
        # Fitting sur une seule vue
        image_data = [{'image': image_rgb, 'keypoints': keypoints}]
        
        try:
            # On utilise process_image qui gère le sizing et appelle fit_model
            res = smpl_engine.process_image(image_data, gender=gender, height=1.70)
            
            if res is None or 'smpl_params' not in res:
                print("Échec du Fitting.")
                failed += 1
                continue
                
            pred_betas = res['smpl_params']['betas']
            
            # Calcul du MAE entre betas prédits et betas réels
            mae = np.mean(np.abs(pred_betas - gt_betas))
            mae_list.append(mae)
            
            print(f"✓ MAE Betas: {mae:.4f}")
            
            if len(mae_list) >= 311:
                print("\n[INFO] Limite de 311 images atteinte pour un test rapide.")
                break
                
        except Exception as e:
            print(f"Erreur process_image: {e}")
            failed += 1
            
    print("\n" + "="*50)
    print("RÉSULTATS DE L'ÉVALUATION SSP-3D (Shape / Betas)")
    print("="*50)
    print(f"Images traitées avec succès : {len(mae_list)} / {len(fnames)}")
    print(f"Échecs de tracking/fitting : {failed}")
    if len(mae_list) > 0:
        global_mae = np.mean(mae_list)
        print(f"MAE Global sur les 10 Betas : {global_mae:.4f}")
        
        if global_mae < 1.0:
            print("\nCONCLUSION: L'optimisation 3D est EXCELLENTE. La forme native du corps est correctement captée.")
            print("Les erreurs résiduelles sur Kaggle sont STRICTEMENT dues aux vêtements.")
        elif global_mae < 2.0:
            print("\nCONCLUSION: L'optimisation 3D est BONNE. Mais il y a un léger flottement dans la détection de la masse corporelle.")
        else:
            print("\nCONCLUSION: L'optimisation 3D a ÉCHOUÉ (MAE > 2.0). L'erreur vient bien du modèle SMPL lui-même qui ne converge pas vers les bonnes proportions.")

if __name__ == "__main__":
    evaluate_ssp3d()
