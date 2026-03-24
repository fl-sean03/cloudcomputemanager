---
name: ccm
description: CloudComputeManager — GPU cloud management platform for running any workload on Vast.ai with checkpointing, preemption recovery, multi-stage pipelines, environment management, benchmarking, and cost optimization. Use when working with CCM code, submitting cloud jobs, managing instances, managing environments, or building on the CCM platform.
argument-hint: "[topic or task]"
---

# CloudComputeManager (CCM) Agent Skill

You are working with **CloudComputeManager** at `/home/sf2/Workspace/main/46-CCM`.

**User request**: $ARGUMENTS

## How to answer questions about CCM

The **single canonical reference** is [`docs/usage.md`](../../../docs/usage.md). Read it for:
- Architecture and file map
- Job YAML schema (all fields)
- CLI command reference (50+ commands)
- Environment management (conda-pack, conda env, Docker)
- Python SDK for agents
- Benchmark framework
- Resilience and recovery
- Multi-stage pipelines

For **development context** (sprint status, what is done, architecture decisions): read `AGENTS.md`.

For **environment design** (strategies, implementation plan): read `docs/ENVIRONMENT_DESIGN.md`.

For **build and test**: `pip install -e ".[dev]"` then `pytest tests/ --ignore=tests/test_e2e_full_lifecycle.py --ignore=tests/test_integration_vast.py`

**Key convention**: All wrapper scripts come from `core/wrapper.py` (single source of truth). Job isolation is via the `project` field. Environment setup runs via SSH after file upload, not in onstart.

If a topic is given, read the relevant source files and docs, then answer. If no topic, give a brief overview and ask what they need.
