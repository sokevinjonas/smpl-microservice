import os
import cv2
import numpy as np
import pickle
import pandas as pd
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

def evaluate_agora(max_images: int = 10):
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
    failed = 0
    processed_images = 0
    
    # 1. Iterate through annotation files
    for ann_file in sorted(ann_dir.glob("*.pkl")):
        if processed_images >= max_images:
            break
            
        print(f"Loading annotations: {ann_file.name}")
        try:
            df = pd.read_pickle(ann_file)
        except Exception as e:
            print(f"Error reading {ann_file.name}: {e}")
            continue

        for _, row in df.iterrows():
            if processed_images >= max_images:
                break
                
            img_name = row['imgPath']
            img_path = img_dir / img_name
            
            if not img_path.exists():
                # Try common variations if extension differs
                img_path = img_dir / img_name.replace('.png', '.jpg')
                if not img_path.exists():
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
                # ex: smpl_gt/folder/person.obj -> dataset/agora/smpl_gt/folder/person.pkl
                pkl_rel_path = gt_obj_path.replace('.obj', '.pkl')
                pkl_abs_path = base_dir / pkl_rel_path
                
                if not pkl_abs_path.exists():
                    # Check if smpl_gt is at root or elsewhere
                    # Sometimes the path already includes 'dataset/agora'
                    if 'dataset/agora' in pkl_rel_path:
                        pkl_abs_path = Path(pkl_rel_path)
                    else:
                        pkl_abs_path = base_dir.parent / pkl_rel_path
                
                if not pkl_abs_path.exists():
                    # Final attempt: search by filename if structure changed
                    filename = Path(pkl_rel_path).name
                    # We skip the heavy search for now to keep it fast
                    continue

                # Load Ground Truth Betas
                try:
                    with open(pkl_abs_path, 'rb') as f:
                        gt_data = pickle.load(f, encoding='latin1')
                    
                    gt_betas = np.array(gt_data.get('betas', gt_data.get('shape', []))).flatten()[:10]
                    if len(gt_betas) == 0: continue
                    
                    gender = genders[i] if i < len(genders) else 'neutral'
                except Exception as e:
                    continue
                
                # Step 1: Detect pose (MediaPipe)
                # Note: AGORA scenes have multiple people, MediaPipe might pick the wrong one.
                # In a real evaluation, we should use the ground truth 2D joints to crop the image.
                pose_res = pose_estimator.estimate_pose(image_rgb)
                keypoints = pose_res['keypoints'] if pose_res else None
                
                if keypoints is None:
                    print(f"   ❌ No person detected in scene.")
                    failed += 1
                    continue
                
                # Step 2: Fitting
                image_data = [{'image': image_rgb, 'keypoints': keypoints}]
                try:
                    res = smpl_engine.process_image(image_data, gender=gender, height=1.70)
                    if res and 'smpl_params' in res:
                        pred_betas = res['smpl_params']['betas']
                        mae = np.mean(np.abs(pred_betas - gt_betas))
                        mae_list.append(mae)
                        print(f"   ✅ Person {i} ({gender}): MAE = {mae:.4f}")
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

if __name__ == "__main__":
    evaluate_agora(max_images=10)
