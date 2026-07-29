# ADR-0004: Pin Python to 3.12

- **Status:** Accepted
- **Date:** 2026-07-28

## Context
Development began on Python 3.14. PyTorch supports 3.14 on Windows, but this
project's dependency tail includes packages with compiled extensions
(scikit-survival, SimpleITK, vLLM) whose wheel availability lags new Python
releases. MONAI does not explicitly document 3.14 support.

## Decision
Pin to Python 3.12 (`requires-python = ">=3.11,<3.13"`, plus `.python-version`).

## Consequences
- Avoids build-from-source failures mid-project on a Windows machine.
- Costs nothing: the project uses no 3.13/3.14-specific language features.
- Revisit once MONAI and scikit-survival publish 3.14 wheels.