import requests
import time
import os

print("--- Testing Exact Weight ---")
json_data_exact = {
    "photo_path": "dataset/agora/images/ag_trainset_renderpeople_bfh_archviz_5_10_cam02_00778_1280x720.png",
    "measures_table": ["taille", "poitrine", "hanche"],
    "gender": "male",
    "height": 1.80,
    "weight": 75.0,
    "include_mesh": True
}
res = requests.post("http://localhost:5000/estimate", json=json_data_exact)
print(f"Status Exact: {res.status_code}")
if res.status_code == 200:
    print(res.json()['metadata'])
else:
    print(res.text)

time.sleep(1)

print("\n--- Testing Weight Interval ---")
json_data_interval = {
    "photo_path": "dataset/agora/images/ag_trainset_renderpeople_bfh_archviz_5_10_cam02_00778_1280x720.png",
    "measures_table": ["taille", "poitrine", "hanche"],
    "gender": "male",
    "height": 1.80,
    "weight": "75-80",
    "include_mesh": True
}
res2 = requests.post("http://localhost:5000/estimate", json=json_data_interval)
print(f"Status Interval: {res2.status_code}")
if res2.status_code == 200:
    print(res2.json()['metadata'])
else:
    print(res2.text)
