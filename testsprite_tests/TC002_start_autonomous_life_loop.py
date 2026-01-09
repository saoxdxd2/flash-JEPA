import requests

BASE_URL = "http://localhost:5173"
TIMEOUT = 30
HEADERS = {
    "Content-Type": "application/json",
    # Add Authorization header here if needed, e.g. "Authorization": "Bearer <token>"
}

def test_start_autonomous_life_loop():
    url = f"{BASE_URL}/life/start"
    try:
        response = requests.post(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        # Additional checks could be added here if response body is specified by API schema
    except requests.exceptions.RequestException as e:
        assert False, f"Request to start autonomous life loop failed: {e}"

test_start_autonomous_life_loop()