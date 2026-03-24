# CCM — Agent & Developer Instructions

- **How to use CCM** (CLI, YAML, SDK, benchmarks, resilience): [`docs/usage.md`](docs/usage.md)
- **Project status** (sprint tracker, what's done, architecture): [`AGENTS.md`](AGENTS.md)
- **Build & test**: `pip install -e ".[dev]"` then `pytest tests/ --ignore=tests/test_e2e_full_lifecycle.py --ignore=tests/test_integration_vast.py`
- **Key convention**: All wrapper scripts come from `core/wrapper.py` (single source of truth). `get_session()` auto-commits. Job isolation is via the `project` field.
