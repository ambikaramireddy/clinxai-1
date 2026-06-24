import requests
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import time

def test_api():
    print("Testing API Endpoints...")
    url = "http://127.0.0.1:8000"
    
    # 1. PING / Wait for server
    # Note: Automation script assumes server is running.
    print(f"Target URL: {url}")

    # 2. Test Data
    data_dir = "c:/Users/ijgam/OneDrive/Desktop/soha/clinxai/data/raw"
    img_name = [f for f in os.listdir(data_dir) if f.endswith('.png')][0]
    img_path = os.path.join(data_dir, img_name)
    
    # Payload
    patient_data = {
        "age": 65, "gender": "M", "fever": 1, "cough": 1, "oxygen": 90
    }
    
    # --- Test /predict/fused ---
    print("\n[1/3] Testing POST /predict/fused...")
    with open(img_path, "rb") as f:
        files = {"file": f}
        # requests parameters for query params
        start = time.time()
        response = requests.post(f"{url}/predict/fused", params=patient_data, files=files)
        elapsed = time.time() - start
        
    if response.status_code == 200:
        print(f"SUCCESS ({elapsed:.3f}s): {response.json()}")
    else:
        print(f"FAILED: {response.text}")

    # --- Test /explain/image ---
    print("\n[2/3] Testing POST /explain/image...")
    with open(img_path, "rb") as f:
        files = {"file": f}
        start = time.time()
        response = requests.post(f"{url}/explain/image", files=files)
        elapsed = time.time() - start
        
    if response.status_code == 200:
        data = response.json()
        print(f"SUCCESS ({elapsed:.3f}s): Heatmap Base64 length = {len(data['heatmap_base64'])}")
    else:
        print(f"FAILED: {response.text}")

    # --- Test /explain/patient ---
    print("\n[3/3] Testing POST /explain/patient...")
    # JSON body for this one
    body = {
        "Age": 65, "Gender": "M", "Fever": 1, "Cough": 1, "Oxygen": 90
    }
    start = time.time()
    response = requests.post(f"{url}/explain/patient", json=body)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        print(f"SUCCESS ({elapsed:.3f}s): {response.json()}")
    else:
        print(f"FAILED: {response.text}")

if __name__ == "__main__":
    test_api()
