"""
Service fallback pour les mensurations
Génère des mensurations réalistes quand les services réels ne fonctionnent pas
"""

import numpy as np
from typing import Dict, List


class FallbackMeasurementService:
    """
    Génère des mensurations réalistes sans dépendre de services externes
    Utile pour développement/test quand SMPL ou MediaPipe ne marchent pas
    """
    
    # Mensurations moyennes (adultes, en mm)
    AVERAGE_MEASUREMENTS = {
        'tour_poitrine': 950,      # 95 cm
        'taille': 800,              # 80 cm
        'hanche': 950,              # 95 cm
        'longueur_bras': 650,       # 65 cm
        'longueur_jambe': 900,      # 90 cm
        'largeur_epaules': 420      # 42 cm
    }
    
    # Variation acceptable (±variance)
    VARIANCE = {
        'tour_poitrine': 150,
        'taille': 120,
        'hanche': 150,
        'longueur_bras': 80,
        'longueur_jambe': 100,
        'largeur_epaules': 60
    }
    
    @staticmethod
    def generate_measurements(requested_measures: List[str]) -> Dict[str, float]:
        """
        Génère des mensurations réalistes et cohérentes
        
        Args:
            requested_measures: Liste des mesures demandées
        
        Returns:
            Dict avec les mensurations générées
        """
        measurements = {}
        
        # Générer une "taille de base" pour cohérence
        size_factor = np.random.normal(1.0, 0.15)  # 85% à 115% de la moyenne
        
        for measure in requested_measures:
            measure_key = measure.lower().strip()
            
            # Mapper aux clés connues
            mapped_key = FallbackMeasurementService._map_measure_name(measure_key)
            
            if mapped_key in FallbackMeasurementService.AVERAGE_MEASUREMENTS:
                avg = FallbackMeasurementService.AVERAGE_MEASUREMENTS[mapped_key]
                var = FallbackMeasurementService.VARIANCE[mapped_key]
                
                # Générer avec variation
                value = avg * size_factor + np.random.normal(0, var * 0.1)
                measurements[measure] = round(max(value, 100), 1)  # Minimum 100mm
            else:
                measurements[measure] = 0.0
        
        return measurements
    
    @staticmethod
    def _map_measure_name(name: str) -> str:
        """Mappe tous les noms possibles vers la clé standard"""
        mapping = {
            'tour_poitrine': 'tour_poitrine',
            'chest_circumference': 'tour_poitrine',
            'poitrine': 'tour_poitrine',
            'chest': 'tour_poitrine',
            
            'taille': 'taille',
            'waist': 'taille',
            
            'hanche': 'hanche',
            'hip': 'hanche',
            'hanches': 'hanche',
            'hip_circumference': 'hanche',
            
            'longueur_bras': 'longueur_bras',
            'arm_length': 'longueur_bras',
            
            'longueur_jambe': 'longueur_jambe',
            'leg_length': 'longueur_jambe',
            
            'largeur_epaules': 'largeur_epaules',
            'shoulder_width': 'largeur_epaules',
        }
        return mapping.get(name, name)
    
    @staticmethod
    def get_fallback_metadata() -> Dict:
        """Retourne les métadonnées de fallback"""
        return {
            'image_shape': [1080, 720],
            'num_keypoints': 33,
            'mesh_vertices': 6890,
            'validation_errors': [],
            'note': 'Mensurations générées (mode fallback - services réels indisponibles)',
            'is_simulated': True
        }
