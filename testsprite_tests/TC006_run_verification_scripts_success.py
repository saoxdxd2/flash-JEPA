import requests

BASE_URL = "http://localhost:5173"
TIMEOUT = 30
HEADERS = {
    "Content-Type": "application/json"
}

def test_run_verification_scripts_success():
    url = f"{BASE_URL}/tests/run"
    try:
        response = requests.post(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
        # Additional checks can be done here if response body contains details
    except requests.exceptions.RequestException as e:
        assert False, f"Request failed: {e}"

test_run_verification_scripts_success()