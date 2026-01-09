import requests

def test_run_lifecycle_cycle_success():
    base_url = "http://localhost:5173"
    endpoint = f"{base_url}/lifecycle/run_cycle"
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(endpoint, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        assert False, f"Request to run lifecycle cycle failed: {e}"
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    # Assuming the response confirmation is in text or JSON, we check for confirmation keywords in response text
    assert "confirmation" in response.text.lower() or "cycle completed" in response.text.lower() or "success" in response.text.lower(), \
        "Response does not contain confirmation of lifecycle cycle completion"

test_run_lifecycle_cycle_success()