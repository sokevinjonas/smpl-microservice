import pickle
import numpy as np
import sys

# MONKEYPATCH for chumpy compatibility
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

def inspect_pkl(file_path):
    print(f"Inspecting {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", data.keys())
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    print(f"{k}: {v.shape}")
                else:
                    print(f"{k}: {type(v)}")
        else:
            print(data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_pkl(sys.argv[1])
    else:
        print("Usage: python inspect_pkl.py <path_to_pkl>")
