# CloudComputeManager (CCM)

A GPU cloud management platform for running any workload on Vast.ai spot instances with automatic checkpointing, preemption recovery, multi-stage pipelines, and cost-performance benchmarking.

## Features

- **Setup Commands** — Install software before jobs run (`apt`, `pip`, `conda`, anything)
- **Multi-Stage Pipelines** — Chain sequential stages in one job (equilibration → production → analysis)
- **Variable Substitution** — `${VAR}` syntax with `--set KEY=VALUE` for parameterized configs
- **Batch Matrix Expansion** — Cartesian product sweeps from a single YAML
- **Automatic Checkpointing** — Detect and sync application checkpoints (LAMMPS, PyTorch, generic)
- **SIGTERM-Aware Execution** — Traps preemption signals, triggers app checkpoints, auto-recovers
- **Progress Monitoring** — Pluggable parsers (regex, file growth, custom command)
- **Budget Enforcement** — Per-job cost/time limits with auto-termination
- **Benchmark Framework** — Automated GPU cost-performance comparison across tiers
- **Agent-Native** — Python SDK, REST API, and CLI for AI agent integration
- **Resilient** — Survives laptop sleep, daemon restarts, instance preemption

## Quick Start

```bash
# Install
cd ~/Workspace/main/46-CCM
pip install -e ".[dev]"
ccm config init
echo "your-api-key" > ~/.vast_api_key

# Submit a job
ccm jobs submit job.yaml

# Or one-liner
ccm run "python train.py" --upload ./project --gpu RTX_3060 --wait

# Check on jobs after being away
ccm reconnect
```

## Documentation

| Document | Purpose |
|----------|---------|
| **[docs/usage.md](docs/usage.md)** | Full reference — architecture, YAML schema, CLI, SDK, benchmarks, resilience, multi-agent |
| **[AGENTS.md](AGENTS.md)** | Current development status, sprint tracker |
| **[docs/SPRINT_2026-03-23.md](docs/SPRINT_2026-03-23.md)** | Sprint plan and design decisions |

## Example Job Config

```yaml
name: my-simulation
project: my-project
image: ubuntu:22.04
setup: |
  apt-get update && apt-get install -y lammps
command: mpirun -np 16 lmp -in simulation.inp

resources:
  gpu_type: RTX_3060
  cpu_cores: 16
  disk_gb: 50

budget:
  max_cost_usd: 5.00
  max_hourly_rate: 0.10

stages:
  - name: equilibration
    command: mpirun -np 16 lmp -in equil.inp
  - name: production
    command: mpirun -np 16 lmp -in prod.inp

progress:
  type: regex_parse
  file: /workspace/output.log
  regex: 'Step\s+(\d+)'
  total: 5000000
```

```bash
ccm jobs submit job.yaml --set STRUCTURE=F100
```

## Templates

```bash
ccm templates list
ccm templates show lammps-gpu
ccm jobs submit job.yaml --template pytorch-train
```

| Template | GPU | Checkpoint | Use Case |
|----------|-----|------------|----------|
| `quick-gpu` | RTX_3060 | No | Quick experiments |
| `lammps-gpu` | RTX_3060 | Yes | MD simulations |
| `namd-production` | RTX_3060 | Yes | NAMD production |
| `pytorch-train` | RTX_4090 | Yes | ML training |
| `jupyter-dev` | RTX_3060 | No | Interactive dev |
| `llm-inference` | RTX_4090 | No | LLM serving |

## Tests

```bash
# Unit tests (314 passing)
pytest tests/ --ignore=tests/test_e2e_full_lifecycle.py --ignore=tests/test_integration_vast.py

# Integration tests (requires API key, costs money)
pytest tests/test_integration_vast.py --run-integration
```

## License

MIT License
