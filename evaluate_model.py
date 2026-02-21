import os
import requests
import json
import csv

# Configuration
DATASET_DIR = "dataset/kaggle"
CSV_FILE = os.path.join(DATASET_DIR, "Body Measurements Image Dataset.csv")
API_URL = "http://localhost:5000/estimate"

def run_evaluation():
    print("Démarrage de l'évaluation du modèle SMPL...")
    
    if not os.path.exists(CSV_FILE):
        print(f"Erreur: Le fichier {CSV_FILE} n'existe pas.")
        return
        
    results = []
    
    with open(CSV_FILE, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        print(f"Dataset chargé avec {len(rows)} individus.")
        
        for row in rows:
            person_id = str(row['set_id']).strip()
            gender = row.get('gender', 'neutral').strip()
            height_str = row.get('height', '')
            
            try:
                height_cm = float(height_str)
                height_m = height_cm / 100.0
            except ValueError:
                height_m = None
            
            # Mapping des vraies mesures (en mm pour correspondre à notre API)
            # Gestion des valeurs avec suffixes '_tbr'
            def clean_val(v):
                if not v or v.strip() == '': return None
                v = v.replace('_tbr', '').strip()
                try: return float(v) * 10.0 # cm to mm
                except: return None
            
            true_measures = {
                'tour_poitrine': clean_val(row['chest_circumference_cm']),
                'taille': clean_val(row['waist_circumference_cm']),
                'hanche': clean_val(row['hips_circumference_cm']),
                'cuisse': clean_val(row['thigh_circumference_cm']),
                'bras': clean_val(row['arm_circumference_cm'])
            }
            
            # Filtrer ce qui n'est pas None
            true_measures = {k: v for k, v in true_measures.items() if v is not None}
            
            # Localiser les photos
            person_dir = os.path.join(DATASET_DIR, person_id)
            if not os.path.exists(person_dir):
                print(f"  ⚠️ Photos introuvables pour ID {person_id}. Ignoré.")
                continue
                
            # Chercher les images de face et profil
            images = os.listdir(person_dir)
            front_img = None
            side_img = None
            
            for img in images:
                if 'front' in img.lower():
                    front_img = os.path.join(person_dir, img)
                elif 'side' in img.lower():
                    side_img = os.path.join(person_dir, img)
                    
            if not front_img:
                # S'il n'y en a qu'une ou deux avec des noms différents, on prend la 1ère
                if len(images) > 0: front_img = os.path.join(person_dir, images[0])
            if not side_img:
                # On prend la 2ème si elle existe
                if len(images) > 1: side_img = os.path.join(person_dir, images[1])
                
            if not front_img:
                 print(f"  ⚠️ Aucune image pour ID {person_id}. Ignoré.")
                 continue
                 
            photos_to_send = [front_img]
            if side_img: photos_to_send.append(side_img)
            
            print(f"\nTraitement ID {person_id} ({gender}, {height_m}m)...")
            
            # Préparer la requête multi-part
            files = [('photos', (os.path.basename(p), open(p, 'rb'), 'image/jpeg')) for p in photos_to_send]
            data = {
                'measures_table': json.dumps(list(true_measures.keys())),
                'gender': gender,
                'height': str(height_m) if height_m else ''
            }
            
            # Appel API
            try:
                response = requests.post(API_URL, files=files, data=data)
                response.raise_for_status()
                res_json = response.json()
                
                pred_measures = res_json.get('measurements', {})
                
                person_result = {'id': person_id, 'errors': {}}
                
                print("  Comparaison (Vrai vs Prédit) :")
                for measure_name, true_val in true_measures.items():
                    pred_val = pred_measures.get(measure_name)
                    if pred_val is not None:
                        error = abs(true_val - pred_val)
                        person_result['errors'][measure_name] = error
                        print(f"    - {measure_name}: {true_val/10:.1f}cm vs {pred_val/10:.1f}cm (Diff: {error/10:.1f}cm)")
                    else:
                        print(f"    - {measure_name}: {true_val/10:.1f}cm vs NON PREDIT")
                        
                results.append(person_result)
                
            except Exception as e:
                print(f"  ❌ Erreur API : {e}")
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"  Détails : {response.text}")
                    
            # Fermer les fichiers
            for _, f_tuple in files:
                f_tuple[1].close()

    # Synthèse Finale
    print("\n\n" + "="*40)
    print("RÉSULTATS DE L'ÉVALUATION (MAE - Erreur Moyenne)")
    print("="*40)
    
    if not results:
        print("Aucun résultat à analyser.")
        return
        
    # Calculer MAE globale par mesure
    errors_by_measure = {}
    for r in results:
        for m, err in r['errors'].items():
            if m not in errors_by_measure:
                errors_by_measure[m] = []
            errors_by_measure[m].append(err)
            
    for measure_name, errors in errors_by_measure.items():
        if len(errors) > 0:
            mae = sum(errors) / len(errors)
            print(f"MAE sur {measure_name.upper()} : {mae/10:.2f} cm (sur {len(errors)} individus)")

if __name__ == "__main__":
    run_evaluation()
