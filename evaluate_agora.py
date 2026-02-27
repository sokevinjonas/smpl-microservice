import os
import cv2
import numpy as np
import pickle
import pandas as pd
import torch
import traceback
import io
from pathlib import Path
import sys

# Patch for NumPy >= 1.24 compatibility with old libs (chumpy/smplx)
for name, target in [('bool', bool), ('int', int), ('float', float), 
                    ('complex', complex), ('object', object), ('str', str),
                    ('unicode', str), ('long', int)]:
    if not hasattr(np, name):
        setattr(np, name, target)

if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

from utils.pose_estimation import PoseEstimator
from smpl_engine import create_smpl_engine

def robust_torch_load(path):
    """
    Les fichiers AGORA sont des pickles standard contenant des tenseurs Torch.
    Si on utilise pickle.load, il échoue sur les tenseurs CUDA.
    Si on utilise torch.load, il échoue sur le "magic number" car ce n'est pas un format Torch natif.
    Cette classe intercepte les chargements de stockage Torch pour les forcer sur CPU.
    """
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        return CPU_Unpickler(f).load()

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def evaluate_agora(max_images: int = 100):
    print(f"🎯 Démarrage de l'évaluation sur le dataset AGORA (max {max_images} images)...")
    
    base_dir = Path("dataset/agora")
    img_dir = base_dir / "images" 
    ann_dir = base_dir / "annotations"
    
    # Path inside the container for ground truth models
    # Note: According to annotations, they are in 'smpl_gt/...'
    gt_root = base_dir / "smpl_gt" 
    
    if not img_dir.exists() or not ann_dir.exists():
        print(f"⚠️ [ERREUR] Dossiers manquants. Images: {img_dir.exists()}, Annotations: {ann_dir.exists()}")
        return

    # Load IA models
    pose_estimator = PoseEstimator()
    smpl_engine = create_smpl_engine()
    
    mae_list = []
    mae_adults = []
    mae_kids = []
    failed = 0
    processed_images = 0
    
    # 1. Iterate through annotation files
    for ann_file in sorted(ann_dir.glob("*.pkl")):
        if processed_images >= max_images:
            break
            
        print(f"Loading annotations: {ann_file.name}")
        try:
            df = pd.read_pickle(ann_file)
            
            # Si on demande beaucoup d'images, on échantillonne pour aller plus vite
            # et voir plus de diversité
            if max_images > 200:
                step = len(df) // (max_images // 10) # ~10% de chaque fichier
                if step > 1:
                    df = df.iloc[::step]
        except Exception as e:
            print(f"Error reading {ann_file.name}: {e}")
            continue

        for _, row in df.iterrows():
            if processed_images >= max_images:
                break
                
            img_name = row['imgPath']
            img_path = img_dir / img_name
            
            if not img_path.exists():
                # Try common variations
                alt_path_1 = img_dir / img_name.replace('.png', '.jpg')
                alt_path_2 = img_dir / img_name.replace('.png', '_1280x720.png')
                alt_path_3 = img_dir / img_name.replace('.jpg', '_1280x720.jpg')
                
                if alt_path_1.exists():
                    img_path = alt_path_1
                elif alt_path_2.exists():
                    img_path = alt_path_2
                elif alt_path_3.exists():
                    img_path = alt_path_3
                else:
                    continue

            print(f"[{processed_images+1}] ⏳ Image: {img_name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print("   ❌ Error loading image.")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # For each image, we might have multiple people. 
            # AGORA provides gt_path_smpl as a list of OBJ paths.
            gt_paths = row.get('gt_path_smpl', [])
            genders = row.get('gender', [])
            
            # For now, let's take the first valid person in the image for evaluation
            # (In a full benchmark we would loop through all, but we need to crop them)
            if not gt_paths:
                print("   ⚠️ No ground truth persons for this image.")
                continue
            
            # For simplicity, we process only one person per image for now (the first one)
            # To handle multiple, we would need to crop or use a bounding box
            person_found = False
            for i, gt_obj_path in enumerate(gt_paths):
                if person_found: break
                
                # Conversion OBJ path to PKL path (params)
                # ex: smpl_gt/folder/person.obj -> folder/person.pkl
                pkl_rel_path = gt_obj_path.replace('.obj', '.pkl')
                
                # Le dataframe contient déjà "smpl_gt/" dans le nom (ex: 'smpl_gt/trainset_renderpeople.../file.obj')
                # base_dir est 'dataset/agora'
                pkl_abs_path = base_dir / pkl_rel_path
                
                # Si le dataframe ne contient pas smpl_gt, on l'ajoute
                if not str(pkl_rel_path).startswith('smpl_gt'):
                    pkl_abs_path = gt_root / pkl_rel_path
                
                if not pkl_abs_path.exists():
                     print(f"   ⚠️ GT introuvable pour {i}: cherché dans {pkl_abs_path}")
                     continue

                # Load Ground Truth Betas
                try:
                    # Utilisation du loader robuste pour contrer les erreurs CUDA imbriquées
                    gt_data = robust_torch_load(pkl_abs_path)
                    
                    # Extraction des betas (peuvent être des tenseurs avec grad)
                    raw_betas = gt_data.get('betas', gt_data.get('shape', []))
                    gt_betas = to_numpy(raw_betas).flatten()[:10]
                    
                    if len(gt_betas) == 0:
                        print(f"   ⚠️ Betas vides pour person {i} dans {pkl_abs_path.name}. Keys: {list(gt_data.keys())}")
                        continue
                    
                    gender = genders[i] if i < len(genders) else 'neutral'
                    
                    # Correction: 'kid' est une liste par image
                    kids_list = row.get('kid', [])
                    is_kid = kids_list[i] if i < len(kids_list) else False
                    
                    # NEW: Get GT height and weight to use as "perfect" constraints for validation
                    gt_metrics = smpl_engine.get_metrics_from_betas(gt_betas, gender=gender)
                    gt_height = gt_metrics['height']
                    gt_weight = gt_metrics['weight']
                    
                except Exception as e:
                    print(f"   ❌ Erreur lecture GT person {i} ({pkl_abs_path.name}): {e}")
                    # traceback.print_exc()
                    continue
                
                # Step 1: Detect pose (MediaPipe)
                pose_res = pose_estimator.estimate_pose(image_rgb)
                keypoints = pose_res['keypoints'] if pose_res else None
                
                if keypoints is None:
                    # On ne compte pas comme échec de MAE mais comme échec de détection
                    print(f"   ❌ No person detected by MediaPipe for person {i}.")
                    continue
                
                # Step 2: Perspective Calibration
                # AGORA a souvent '50mm' dans le nom de fichier. 
                # Focale 50mm sur capteur 36mm -> focal_px = (50/36) * width
                focal_px = 1777.8 # Par défaut pour 50mm / 1280px
                if '28mm' in img_name:
                    focal_px = (28/36) * 1280 # ~995.5
                
                # Step 3: Fitting with GT constraints and Perspective
                # Simuler 2 vues (Face + "Mock" Profil) car l'API exige 2 vues
                image_data = [
                    {'image': image_rgb, 'keypoints': keypoints, 'segmentation_mask': pose_res.get('segmentation_mask')},
                    {'image': image_rgb, 'keypoints': keypoints, 'segmentation_mask': pose_res.get('segmentation_mask')}
                ]
                try:
                    # On utilise la taille et le poids RÉELS du sujet AGORA + Projection Perspective
                    res = smpl_engine.process_image(
                        image_data, 
                        gender=gender, 
                        height=gt_height, 
                        target_weight=gt_weight,
                        focal_length=focal_px
                    )
                    
                    if res and 'smpl_params' in res:
                        pred_betas = res['smpl_params']['betas']
                        mae = np.mean(np.abs(pred_betas - gt_betas))
                        mae_list.append(mae)
                        
                        if is_kid:
                            mae_kids.append(mae)
                        else:
                            mae_adults.append(mae)
                            
                        type_str = "KID" if is_kid else "ADULT"
                        print(f"   ✅ Person {i} ({gender}, {type_str}): MAE = {mae:.4f} (Height={gt_height:.2f}m, Weight={gt_weight:.1f}kg)")
                        person_found = True
                    else:
                        print(f"   ❌ Fitting failed for person {i}.")
                except Exception as e:
                    print(f"   ❌ Error fitting person {i}: {e}")
            
            if not person_found:
                print("   ❌ No Ground Truth file found for the persons in this image.")
                failed += 1
            
            processed_images += 1
            
    # Bilan Final
    print("\n" + "="*50)
    print("RÉSULTATS DE L'ÉVALUATION AGORA (Shape / Betas)")
    print("="*50)
    print(f"Images traitées : {processed_images}")
    print(f"Évaluations réussies : {len(mae_list)}")
    print(f"Échecs : {failed}")
    
    if len(mae_list) > 0:
        global_mae = np.mean(mae_list)
        print(f"MAE GLOBAL sur la forme (Betas) : {global_mae:.4f}")
        if mae_adults:
            print(f"MAE ADULTES : {np.mean(mae_adults):.4f} ({len(mae_adults)} sujets)")
        if mae_kids:
            print(f"MAE ENFANTS : {np.mean(mae_kids):.4f} ({len(mae_kids)} sujets)")
    print("==================================================\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=100)
    args = parser.parse_args()
    evaluate_agora(max_images=args.max)
