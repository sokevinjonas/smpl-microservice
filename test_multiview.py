import requests
import json
import base64
import time

# Use any two local images for the test
IMG1_PATH = "dataset/agora/images/ag_trainset_renderpeople_bfh_archviz_5_10_cam02_00778_1280x720.png"
IMG2_PATH = "dataset/agora/images/ag_trainset_renderpeople_body_hdri_50mm_5_10_00066_1280x720.png"

url = "http://localhost:5000/estimate"

print("--- Testing Single View Exception (Should Fail) ---")
json_data_1 = {
    "photos": [
        IMG1_PATH
    ],
    "measures_table": ["taille", "poitrine", "hanche"],
    "gender": "male",
    "height": 1.75,
    "weight": 70.0
}

try:
    response1 = requests.post(url, json=json_data_1)
    print(f"Status 1 View: {response1.status_code}")
    print(response1.text)
except Exception as e:
    print("Error:", e)

time.sleep(1)

print("\n--- Testing Two Views (Should Pass/Optimize) ---")
json_data_2 = {
    "photos": [
        IMG1_PATH,
        IMG2_PATH
    ],
    "measures_table": ["taille", "poitrine", "hanche"],
    "gender": "male",
    "height": 1.75,
    "weight": "70-75"
}

try:
    response2 = requests.post(url, json=json_data_2)
    print(f"Status 2 Views: {response2.status_code}")
    # Print only beginning to avoid huge 3D payload dumps
    print(response2.text[:300] + "...")
except Exception as e:
    print("Error:", e)
