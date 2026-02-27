import torch
import numpy as np

def calculate_mesh_volume_tensor(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Calcule le volume d'un maillage 3D fermé en utilisant PyTorch.
    Cette fonction est différentiable et peut être intégrée dans une boucle d'optimisation (loss).
    
    Args:
        vertices: Tensor de forme (B, V, 3) ou (V, 3) où B est le batch size et V le nombre de sommets.
        faces: Tensor de forme (F, 3) contenant les indices des sommets pour chaque triangle.
        
    Returns:
        Tensor contenant le volume calculé (en m³ si les vertices sont en mètres).
    """
    # Gérer le cas sans batch dimension
    has_batch = vertices.dim() == 3
    if not has_batch:
        vertices = vertices.unsqueeze(0)
        
    # Extraire les coordonnées des 3 sommets pour chaque face
    # p1, p2, p3 ont pour forme (B, F, 3)
    p1 = vertices[:, faces[:, 0], :]
    p2 = vertices[:, faces[:, 1], :]
    p3 = vertices[:, faces[:, 2], :]
    
    # Produit vectoriel (Cross product): p2 x p3
    cross_prod = torch.cross(p2, p3, dim=-1)
    
    # Produit scalaire (Dot product): p1 . (p2 x p3)
    # Le volume signé d'un tétraèdre (origine, p1, p2, p3) est 1/6 * p1 . (p2 x p3)
    # Le volume total est la somme des volumes signés
    dot_prod = torch.sum(p1 * cross_prod, dim=-1)
    
    # Somme sur toutes les faces et division par 6
    volume = torch.sum(dot_prod, dim=-1) / 6.0
    
    # Retourner la valeur absolue
    volume = torch.abs(volume)
    
    if not has_batch:
        volume = volume.squeeze(0)
        
    return volume


def estimate_weight_from_volume(volume_m3: torch.Tensor, density_kg_per_m3: float = 1010.0) -> torch.Tensor:
    """
    Estime le poids d'un corps humain en fonction de son volume.
    
    Args:
        volume_m3: Tensor contenant le volume en mètres cubes.
        density_kg_per_m3: Densité moyenne du corps humain (environ 1.01 kg/L ou 1010 kg/m³).
        
    Returns:
        Tensor contenant le poids estimé en kilogrammes.
    """
    return volume_m3 * density_kg_per_m3

def np_calculate_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Calcule le volume d'un maillage 3D fermé en utilisant NumPy.
    Utile pour l'évaluation post-optimisation.
    
    Args:
        vertices: Array NumPy de forme (V, 3)
        faces: Array NumPy de forme (F, 3)
        
    Returns:
        Volume en m³.
    """
    p1 = vertices[faces[:, 0]]
    p2 = vertices[faces[:, 1]]
    p3 = vertices[faces[:, 2]]
    
    # Produit vectoriel et scalaire
    cross_prod = np.cross(p2, p3, axis=1)
    signed_volumes = np.sum(p1 * cross_prod, axis=1) / 6.0
    
    return float(np.abs(np.sum(signed_volumes)))
