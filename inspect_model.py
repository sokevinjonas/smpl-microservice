import numpy as np
import pickle
import sys
import os

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

import chumpy as ch

def inspect_model(model_path):
    print(f"--- Inspection du modèle : {model_path} ---")
    
    if not os.path.exists(model_path):
        print(f"❌ Fichier non trouvé : {model_path}")
        return

    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print(f"Type de données : {type(data)}")
        
        if isinstance(data, dict):
            print(f"Clés disponibles : {list(data.keys())}")
            
            # Vérifier les dimensions critiques
            for key in ['J_regressor', 'weights', 'v_template', 'shapedirs', 'posedirs']:
                if key in data:
                    val = data[key]
                    if hasattr(val, 'shape'):
                        print(f"Shape of {key}: {val.shape}")
                    elif hasattr(val, 'r'): # chumpy object
                         print(f"Shape of {key} (chumpy): {val.r.shape}")
                    else:
                        print(f"Type of {key}: {type(val)}")
                        
        else:
            print("Le modèle n'est pas un dictionnaire.")

    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")

if __name__ == "__main__":
    # Chemin par défaut dans le conteneur
    default_path = "/app/models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    
    if len(sys.argv) > 1:
        inspect_model(sys.argv[1])
    else:
        inspect_model(default_path)
