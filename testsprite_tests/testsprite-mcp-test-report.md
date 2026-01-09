# TestSprite AI Testing Report(MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** model (Flash-JEPA)
- **Date:** 2026-01-07
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

### Requirement: Lifecycle Pipeline
- **Description:** Ability to run a full lifecycle cycle (load → imprint → stabilize → grounding → schooling → consolidation/dreaming → evolution → save).

#### Test TC001
- **Test Name:** run_lifecycle_cycle_success
- **Test Code:** [TC001_run_lifecycle_cycle_success.py](./TC001_run_lifecycle_cycle_success.py)
- **Test Error:**
  ```
  requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/lifecycle/run_cycle
  AssertionError: Request to run lifecycle cycle failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/lifecycle/run_cycle
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/099765e8-0ca6-44e2-9d1c-129640d74d4a
- **Status:** ❌ Failed
- **Severity:** HIGH
- **Analysis / Findings:** The service running on `localhost:5173` does not implement a POST handler for `/lifecycle/run_cycle` (received HTTP 501 Unsupported method). In this repository, lifecycle execution appears to be a **Python CLI/script workflow** (`run_lifecycle.py`, `brain/lifecycle.py`), not an HTTP API. As a result, this test cannot validate lifecycle functionality via HTTP.

---

### Requirement: Autonomous Life Loop (Population Mode)
- **Description:** Ability to start and run the population-based autonomous life loop.

#### Test TC002
- **Test Name:** start_autonomous_life_loop
- **Test Code:** [TC002_start_autonomous_life_loop.py](./TC002_start_autonomous_life_loop.py)
- **Test Error:**
  ```
  requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/life/start
  AssertionError: Request to start autonomous life loop failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/life/start
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/8fe3adcd-96c8-4aeb-85e8-f2cbafd8e455
- **Status:** ❌ Failed
- **Severity:** HIGH
- **Analysis / Findings:** The service on `localhost:5173` does not support POST requests to `/life/start` (HTTP 501). The implementation for this feature appears to be a **long-running script** (`autonomous_life.py`) rather than an HTTP endpoint. Test plan should be adapted to invoke the script (or a dedicated API server should be introduced).

---

### Requirement: ONNX Export + Hybrid Inference
- **Description:** Ability to export the reflex path to ONNX and run inference/decision logic with hybrid switching.

#### Test TC003
- **Test Name:** export_reflex_path_to_onnx
- **Test Code:** [TC003_export_reflex_path_to_onnx.py](./TC003_export_reflex_path_to_onnx.py)
- **Test Error:**
  ```
  requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/inference/export_reflex
  AssertionError: Request to export_reflex failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/inference/export_reflex
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/2ea86b19-9fa3-471d-a112-a3f171356fc2
- **Status:** ❌ Failed
- **Severity:** HIGH
- **Analysis / Findings:** The server on `localhost:5173` is not an inference API and does not support POST. ONNX export/hybrid inference seems implemented as Python methods and verification scripts (e.g., `tests/deep_integrity_test.py`, `tests/final_verification.py`) rather than HTTP endpoints.

---

#### Test TC004
- **Test Name:** hybrid_inference_decision_switching
- **Test Code:** [TC004_hybrid_inference_decision_switching.py](./TC004_hybrid_inference_decision_switching.py)
- **Test Error:**
  ```
  AssertionError: Expected status 200, got 501
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/f32616e6-15de-446c-ba59-5c594aa1c737
- **Status:** ❌ Failed
- **Severity:** HIGH
- **Analysis / Findings:** Same root cause as TC003: the target on `localhost:5173` does not implement the expected HTTP endpoint `/inference/decide` and returns 501 for POST requests. This blocks validation of hybrid switching behavior through the generated HTTP tests.

---

### Requirement: Knowledge Transfer / Language Imprinting (Qwen Teacher)
- **Description:** Ability to run language imprinting/transfer steps (conceptual Qwen teacher integration).

#### Test TC005
- **Test Name:** imprint_language_concepts_with_qwen
- **Test Code:** [TC005_imprint_language_concepts_with_qwen.py](./TC005_imprint_language_concepts_with_qwen.py)
- **Test Error:**
  ```
  requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/transfer/imprint_language
  AssertionError: Request failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/transfer/imprint_language
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/d15891bd-ee64-499b-a0ef-b9b9c56894a8
- **Status:** ❌ Failed
- **Severity:** HIGH
- **Analysis / Findings:** The generated test assumes an HTTP endpoint exists for imprinting (`/transfer/imprint_language`), but the codebase appears to do imprinting internally via Python (`brain/lifecycle.py`, `scripts/start_n2n2_qwen3_agentic.py`). Additionally, this feature may require heavy model weights/configuration; even after adding an API layer, the environment must be prepared accordingly.

---

### Requirement: Verification / Integrity Test Scripts
- **Description:** Ability to run verification scripts validating ONNX export, fallback logic, and dynamic growth stability.

#### Test TC006
- **Test Name:** run_verification_scripts_success
- **Test Code:** [TC006_run_verification_scripts_success.py](./TC006_run_verification_scripts_success.py)
- **Test Error:**
  ```
  requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/tests/run
  AssertionError: Request failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/tests/run
  ```
- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/cf7ab713-cd9b-4daa-afbb-8ff943392165
- **Status:** ❌ Failed
- **Severity:** HIGH
- **Analysis / Findings:** The codebase includes test/verification scripts under `tests/` that appear intended to be executed as scripts (e.g., `python tests/deep_integrity_test.py`). The generated test expects an HTTP endpoint `/tests/run`, which is not implemented; the server returns 501 for POST.

---

## 3️⃣ Coverage & Matching Metrics

- **0 / 6** tests passed (**0.00%**)

| Requirement                                  | Total Tests | ✅ Passed | ❌ Failed |
|----------------------------------------------|-------------|-----------|----------|
| Lifecycle Pipeline                            | 1           | 0         | 1        |
| Autonomous Life Loop (Population Mode)        | 1           | 0         | 1        |
| ONNX Export + Hybrid Inference                | 2           | 0         | 2        |
| Knowledge Transfer / Language Imprinting (Qwen)| 1           | 0         | 1        |
| Verification / Integrity Test Scripts          | 1           | 0         | 1        |

---

## 4️⃣ Key Gaps / Risks

- **No HTTP backend implemented**
  The generated tests are HTTP-based and send POST requests to endpoints like `/lifecycle/run_cycle`, but the service running on port `5173` behaves like a static server (501 Unsupported method for POST). This prevents validation of the project’s actual Python functionality.

- **Conceptual API docs vs. actual runtime**
  The OpenAPI-like docs used for test generation are conceptual mappings of Python operations, not evidence that an HTTP server exists.

- **Heavy dependencies / model availability risk**
  Features like Qwen teacher imprinting likely require large weights and configuration that may not be available in the test execution environment, even if an API layer existed.

### Recommended next steps
- **Option A (preferred for TestSprite HTTP tests):** Add a lightweight backend (e.g., FastAPI) that exposes the required endpoints and maps them to existing Python functions.
- **Option B (preferred for current codebase intent):** Adjust the test plan to run CLI scripts directly (e.g., run `python run_lifecycle.py --cycles 1 --mode meditation` and `python tests/deep_integrity_test.py`) and assert exit codes/log markers, instead of using HTTP.
