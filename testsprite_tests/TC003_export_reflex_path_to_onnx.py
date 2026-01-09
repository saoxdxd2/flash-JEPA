import requests

BASE_URL = "http://localhost:5173"
TIMEOUT = 30

def test_export_reflex_path_to_onnx():
    url = f"{BASE_URL}/inference/export_reflex"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
        json_resp = response.json()
        # As per PRD, it should return export confirmation; confirm if a key or message exists
        # We check if response body contains something indicating success (not detailed in PRD, so minimally check JSON)
        assert isinstance(json_resp, dict), "Response is not a JSON object"
        # We can look for typical keys or messages, but PRD only states "ONNX model exported"
        # So any message or confirmation would be acceptable.
        assert any("export" in key.lower() or "success" in key.lower() for key in json_resp.keys()) or json_resp == {}, \
            "Response JSON does not indicate export confirmation"
    except requests.exceptions.RequestException as e:
        assert False, f"Request to export_reflex failed: {e}"

test_export_reflex_path_to_onnx()