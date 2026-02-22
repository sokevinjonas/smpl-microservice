import pickle
import torch
import io
import numpy as np

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

path = '/app/dataset/agora/smpl_gt/trainset_renderpeople_adults_body/rp_alison_posed_002_0_0.pkl'
with open(path, 'rb') as f:
    data = CPU_Unpickler(f).load()

print('--- Keys ---')
print(data.keys())

for key in data.keys():
    val = data[key]
    if torch.is_tensor(val):
        print(f'{key}: Tensor shape={val.shape}, grad={val.requires_grad}')
    elif isinstance(val, (np.ndarray, list)):
        print(f'{key}: array/list len={len(val)}')
    else:
        print(f'{key}: type={type(val)}, value={val}')

if 'betas' in data:
    if torch.is_tensor(data['betas']):
        betas = data['betas'].detach().cpu().numpy()
    else:
        betas = np.array(data['betas'])
    print(f'Betas (10): {betas.flatten()[:10]}')
