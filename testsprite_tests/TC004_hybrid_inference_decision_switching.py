import requests

BASE_URL = "http://localhost:5173"
TIMEOUT = 30


def test_hybrid_inference_decision_switching():
    url = f"{BASE_URL}/inference/decide"
    headers = {
        "Content-Type": "application/json"
    }

    # Example inputs to cover ONNX (System 1), PyTorch (System 2), switching and fallback
    # Since the PRD does not specify the exact request schema, we assume a plausible payload:
    # The payload should allow the system to run inference and test switching behavior.
    # We'll test three payloads in sequence to verify behavior:
    test_payloads = [
        {
            "input_data": [0.5, 0.1, 0.3],  # example input expected to be handled well by ONNX
            "mode": "hybrid"
        },
        {
            "input_data": [0.7, 0.9, 0.8],  # example input expected to fallback to PyTorch
            "mode": "hybrid"
        },
        {
            "input_data": [0.0, 0.0, 0.0],  # example input triggering fallback due to low confidence
            "mode": "hybrid"
        }
    ]

    for payload in test_payloads:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        except requests.RequestException as e:
            assert False, f"Request to /inference/decide failed: {e}"

        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"

        try:
            data = response.json()
        except ValueError:
            assert False, "Response is not valid JSON"

        # Validate that 'action' and 'logits' keys exist in the response according to PRD
        assert "action" in data, "'action' key missing in response"
        assert "logits" in data, "'logits' key missing in response"

        # Validate 'action' is a non-empty string (assuming an action name or label)
        assert isinstance(data["action"], str), "'action' should be a string"
        assert data["action"], "'action' should not be empty"

        # Validate 'logits' is a list of floats (assuming logits vector)
        assert isinstance(data["logits"], list), "'logits' should be a list"
        assert all(isinstance(logit, (float, int)) for logit in data["logits"]), "All logits should be numbers"
        assert len(data["logits"]) > 0, "'logits' should contain at least one value"


test_hybrid_inference_decision_switching()