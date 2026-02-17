"""
Script de test pour le microservice SMPL.
Permet de tester les différents endpoints.
"""

import requests
import json
import numpy as np
from pathlib import Path


BASE_URL = 'http://localhost:5000'


def test_health():
    """Test du health check."""
    print("\n=== Test Health Check ===")
    response = requests.get(f'{BASE_URL}/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_models_status():
    """Test du statut des modèles."""
    print("\n=== Test Models Status ===")
    response = requests.get(f'{BASE_URL}/models/status')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_measurement_reference():
    """Test des référence de mensurations."""
    print("\n=== Test Measurement Reference ===")
    response = requests.get(f'{BASE_URL}/measurements/reference')
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available measurements: {data.get('available_measurements', [])}")
    return response.status_code == 200


def test_estimate_with_local_image(image_path: str):
    """
    Test l'estimation avec une image locale.

    Args:
        image_path: Chemin vers l'image
    """
    print(f"\n=== Test Estimate (Local Image: {image_path}) ===")

    payload = {
        'photo_path': image_path,
        'measures_table': [
            'tour_poitrine',
            'taille',
            'hanche',
            'longueur_bras',
            'longueur_jambe'
        ]
    }

    response = requests.post(
        f'{BASE_URL}/estimate',
        json=payload,
        headers={'Content-Type': 'application/json'}
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Measurements: {json.dumps(data.get('measurements', {}), indent=2)}")
        print(f"Metadata: {json.dumps(data.get('metadata', {}), indent=2)}")
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def test_estimate_with_url(image_url: str):
    """
    Test l'estimation avec une URL d'image.

    Args:
        image_url: URL de l'image
    """
    print(f"\n=== Test Estimate (URL: {image_url}) ===")

    payload = {
        'photo_url': image_url,
        'measures_table': [
            'tour_poitrine',
            'taille',
            'hanche'
        ]
    }

    response = requests.post(
        f'{BASE_URL}/estimate',
        json=payload,
        headers={'Content-Type': 'application/json'}
    )

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Measurements: {json.dumps(data.get('measurements', {}), indent=2)}")
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def run_all_tests(image_path: str = None):
    """Exécute tous les tests."""
    print("=" * 50)
    print("MICROSERVICE SMPL - TEST SUITE")
    print("=" * 50)

    results = {
        'health': test_health(),
        'models_status': test_models_status(),
        'measurement_reference': test_measurement_reference(),
    }

    if image_path and Path(image_path).exists():
        results['estimate_local'] = test_estimate_with_local_image(image_path)

    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


if __name__ == '__main__':
    import sys

    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    try:
        run_all_tests(image_path)
    except requests.exceptions.ConnectionError:
        print("Erreur: Impossible de se connecter au serveur.")
        print(f"Assurez-vous que le serveur est lancé sur {BASE_URL}")
    except Exception as e:
        print(f"Erreur: {e}")
