
# TestSprite AI Testing Report(MCP)

---

## 1️⃣ Document Metadata
- **Project Name:** model
- **Date:** 2026-01-07
- **Prepared by:** TestSprite AI Team

---

## 2️⃣ Requirement Validation Summary

#### Test TC001
- **Test Name:** run_lifecycle_cycle_success
- **Test Code:** [TC001_run_lifecycle_cycle_success.py](./TC001_run_lifecycle_cycle_success.py)
- **Test Error:** Traceback (most recent call last):
  File "<string>", line 11, in test_run_lifecycle_cycle_success
  File "/var/task/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/lifecycle/run_cycle

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 19, in <module>
  File "<string>", line 13, in test_run_lifecycle_cycle_success
AssertionError: Request to run lifecycle cycle failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/lifecycle/run_cycle

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/099765e8-0ca6-44e2-9d1c-129640d74d4a
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC002
- **Test Name:** start_autonomous_life_loop
- **Test Code:** [TC002_start_autonomous_life_loop.py](./TC002_start_autonomous_life_loop.py)
- **Test Error:** Traceback (most recent call last):
  File "<string>", line 14, in test_start_autonomous_life_loop
  File "/var/task/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/life/start

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 20, in <module>
  File "<string>", line 18, in test_start_autonomous_life_loop
AssertionError: Request to start autonomous life loop failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/life/start

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/8fe3adcd-96c8-4aeb-85e8-f2cbafd8e455
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC003
- **Test Name:** export_reflex_path_to_onnx
- **Test Code:** [TC003_export_reflex_path_to_onnx.py](./TC003_export_reflex_path_to_onnx.py)
- **Test Error:** Traceback (most recent call last):
  File "<string>", line 14, in test_export_reflex_path_to_onnx
  File "/var/task/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/inference/export_reflex

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 27, in <module>
  File "<string>", line 25, in test_export_reflex_path_to_onnx
AssertionError: Request to export_reflex failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/inference/export_reflex

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/2ea86b19-9fa3-471d-a112-a3f171356fc2
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC004
- **Test Name:** hybrid_inference_decision_switching
- **Test Code:** [TC004_hybrid_inference_decision_switching.py](./TC004_hybrid_inference_decision_switching.py)
- **Test Error:** Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 59, in <module>
  File "<string>", line 38, in test_hybrid_inference_decision_switching
AssertionError: Expected status 200, got 501

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/f32616e6-15de-446c-ba59-5c594aa1c737
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC005
- **Test Name:** imprint_language_concepts_with_qwen
- **Test Code:** [TC005_imprint_language_concepts_with_qwen.py](./TC005_imprint_language_concepts_with_qwen.py)
- **Test Error:** Traceback (most recent call last):
  File "<string>", line 17, in test_imprint_language_concepts_with_qwen
  File "/var/task/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/transfer/imprint_language

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 24, in <module>
  File "<string>", line 22, in test_imprint_language_concepts_with_qwen
AssertionError: Request failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/transfer/imprint_language

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/d15891bd-ee64-499b-a0ef-b9b9c56894a8
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---

#### Test TC006
- **Test Name:** run_verification_scripts_success
- **Test Code:** [TC006_run_verification_scripts_success.py](./TC006_run_verification_scripts_success.py)
- **Test Error:** Traceback (most recent call last):
  File "<string>", line 13, in test_run_verification_scripts_success
  File "/var/task/requests/models.py", line 1024, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/tests/run

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/var/task/handler.py", line 258, in run_with_retry
    exec(code, exec_env)
  File "<string>", line 19, in <module>
  File "<string>", line 17, in test_run_verification_scripts_success
AssertionError: Request failed: 501 Server Error: Unsupported method ('POST') for url: http://localhost:5173/tests/run

- **Test Visualization and Result:** https://www.testsprite.com/dashboard/mcp/tests/2e36158d-fe5b-4ed5-9031-c8391c34939c/cf7ab713-cd9b-4daa-afbb-8ff943392165
- **Status:** ❌ Failed
- **Analysis / Findings:** {{TODO:AI_ANALYSIS}}.
---


## 3️⃣ Coverage & Matching Metrics

- **0.00** of tests passed

| Requirement        | Total Tests | ✅ Passed | ❌ Failed  |
|--------------------|-------------|-----------|------------|
| ...                | ...         | ...       | ...        |
---


## 4️⃣ Key Gaps / Risks
{AI_GNERATED_KET_GAPS_AND_RISKS}
---