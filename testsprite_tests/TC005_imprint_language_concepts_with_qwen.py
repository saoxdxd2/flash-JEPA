import requests

def test_imprint_language_concepts_with_qwen():
    base_url = "http://localhost:5173"
    endpoint = "/transfer/imprint_language"
    url = base_url + endpoint
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "teacher": "qwen",
        "batch_size": 16,
        "steps": 5
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        # Depending on API response, we can check more details if available:
        # e.g. assert "imprint_status" in response.json(), "Response JSON missing 'imprint_status'"
    except requests.exceptions.RequestException as e:
        assert False, f"Request failed: {e}"

test_imprint_language_concepts_with_qwen()