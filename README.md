# CloudComputeManager (CCM)

A robust GPU cloud management platform with automatic checkpointing, spot instance recovery, and batch job orchestration.

## Features

- **Job Templates**: Pre-configured templates for NAMD, LAMMPS, PyTorch, and more
- **Automatic Checkpointing**: Detect and sync application checkpoints
- **Spot Instance Recovery**: Seamlessly recover from preemptions with checkpoint restore
- **Background Daemon**: Automatic job monitoring, sync, and completion handling
- **Batch Operations**: Submit and manage multiple jobs efficiently
- **Continuous Sync**: Keep data synchronized between instances and local storage
- **Agent-Native API**: Full REST API for AI agent integration
- **CLI Interface**: Rich terminal UI for interactive use

## Installation

```bash
# Clone the repository
cd /home/sf2/Workspace/main/46-VastManager

# Install in development mode
pip install -e ".[dev]"

# Initialize configuration
ccm config init
```

## Quick Start

### 1. Configure Vast.ai API Key

```bash
# Option A: Create key file
echo "your-api-key" > ~/.vast_api_key

# Option B: Set environment variable
export CCM_VAST_API_KEY="your-api-key"
```

### 2. One-Liner Job Submission

```bash
# Quick GPU job
ccm run "python train.py" --upload ./project --gpu RTX_3060

# With specific template
ccm run "namd3 production.namd" --template namd-production --wait
```

### 3. Submit Job with Template

```bash
# List available templates
ccm templates list

# Generate minimal config from template
ccm templates generate namd-production -o job.yaml

# Edit job.yaml to add your command and upload source
# Then submit:
ccm submit job.yaml --template namd-production
```

### 4. Batch Job Submission

```bash
# Submit multiple jobs
ccm batch submit jobs/*.yaml --parallel 5

# Monitor batch status
ccm batch status --project my-project

# Wait for all jobs to complete
ccm batch wait --project my-project
```

### 5. Start Background Daemon

```bash
# Start daemon for automatic monitoring
ccm daemon start

# Check daemon status
ccm daemon status

# View daemon logs
ccm daemon logs --follow
```

## Job Configuration

Create a job configuration file (`job.yaml`):

```yaml
# Use a template as base
template: namd-production

name: my-simulation
project: protein-folding

# Override template values
command: namd3 +p8 +devices 0 production.namd

upload:
  source: ./simulation_files/

# Optional overrides
resources:
  gpu_type: RTX_4090

budget:
  max_hourly_rate: 0.15
```

Full configuration example without template:

```yaml
name: lammps-simulation
project: materials-science

image: nvcr.io/nvidia/lammps:stable

command: |
  cd /workspace
  mpirun -np 4 lmp -in simulation.in -sf gpu -pk gpu 1

resources:
  gpu_type: RTX_3060
  gpu_count: 1
  gpu_memory_min: 12
  disk_gb: 50

upload:
  source: ./input_files/
  destination: /workspace
  exclude:
    - "*.dump"
    - "__pycache__"

sync:
  enabled: true
  interval: 300
  source: /workspace
  destination: ./results
  include_patterns:
    - "*.dump"
    - "*.log"

checkpoint:
  enabled: true
  patterns:
    - "restart.*"
    - "*.restart"
  interval: 600

budget:
  max_hourly_rate: 0.10

terminate_on_complete: true
```

## Available Templates

| Template | Description | GPU | Checkpoint |
|----------|-------------|-----|------------|
| `namd-production` | NAMD molecular dynamics | RTX_3060 | Yes |
| `lammps-gpu` | LAMMPS GPU simulations | RTX_3060 | Yes |
| `pytorch-train` | PyTorch deep learning | RTX_4090 | Yes |
| `quick-gpu` | Quick jobs, no checkpoint | RTX_3060 | No |
| `jupyter-dev` | Interactive Jupyter | RTX_3060 | No |
| `llm-inference` | LLM inference (vLLM) | RTX_4090 | No |

```bash
# Show template details
ccm templates show namd-production
```

## CLI Commands

### Job Management

```bash
ccm submit job.yaml                    # Submit job
ccm submit job.yaml --template pytorch-train  # With template
ccm status                             # List all jobs
ccm jobs status JOB_ID                 # Job details
ccm jobs logs JOB_ID --follow          # View logs
ccm jobs sync JOB_ID                   # Trigger sync
ccm jobs wait JOB_ID                   # Wait for completion
ccm jobs complete JOB_ID               # Mark as complete
ccm jobs cancel JOB_ID                 # Cancel job
ccm jobs recover JOB_ID                # Recover failed job
```

### Batch Operations

```bash
ccm batch submit jobs/*.yaml           # Submit multiple jobs
ccm batch submit *.yaml --parallel 5   # Limit concurrent jobs
ccm batch status                       # Show batch summary
ccm batch wait                         # Wait for all jobs
ccm batch cancel --project NAME        # Cancel all project jobs
```

### Instance Management

```bash
ccm search RTX_4090                    # Search offers
ccm instances list                     # List instances
ccm instances ssh 12345678             # SSH into instance
ccm instances terminate 12345678       # Terminate instance
```

### Background Daemon

```bash
ccm daemon start                       # Start daemon
ccm daemon start --foreground          # Run in foreground
ccm daemon stop                        # Stop daemon
ccm daemon status                      # Check status
ccm daemon logs                        # View logs
ccm daemon logs --follow               # Follow logs
```

### Maintenance

```bash
ccm cleanup                            # Preview stale jobs
ccm cleanup --execute                  # Clean up stale jobs
ccm cleanup --orphans --execute        # Also terminate orphans
```

### Templates

```bash
ccm templates list                     # List templates
ccm templates show NAME                # Show template details
ccm templates generate NAME            # Generate minimal config
ccm templates generate NAME -o job.yaml  # Save to file
```

## Daemon Features

The CCM daemon runs in the background and automatically:

- **Monitors running jobs** - Polls for completion every 30 seconds
- **Detects job completion** - Looks for `.ccm_exit_code` file
- **Syncs results** - Downloads results on job completion
- **Terminates instances** - Cleans up after completion (configurable)
- **Handles preemption** - Detects spot instance termination
- **Recovers jobs** - Automatically restarts preempted jobs

Start the daemon:

```bash
ccm daemon start

# View what the daemon is doing
ccm daemon logs --follow
```

## Preemption Recovery

CCM automatically recovers from spot instance preemption:

1. Daemon detects instance termination
2. Job marked as `RECOVERING`
3. Latest checkpoint located
4. New instance created with same spec
5. Checkpoint restored
6. Job resumed

Recovery settings in job config:

```yaml
# Enable preemption recovery (default: true)
preemption_recovery: true
max_recovery_attempts: 3
```

Manual recovery:

```bash
ccm jobs recover JOB_ID
```

## REST API

Start the API server:

```bash
ccm api start
```

Key endpoints:

- `POST /v1/jobs` - Submit job
- `GET /v1/jobs` - List jobs
- `GET /v1/jobs/{job_id}` - Get job details
- `DELETE /v1/jobs/{job_id}` - Cancel job
- `POST /v1/jobs/{job_id}/sync` - Trigger sync
- `GET /v1/offers` - Search GPU offers

API documentation: `http://localhost:8765/docs`

## Python SDK

```python
import asyncio
from cloudcomputemanager.providers import VastProvider
from cloudcomputemanager.core.templates import create_job_config_from_template

async def main():
    provider = VastProvider()

    # Search for offers
    offers = await provider.search_offers(
        gpu_type="RTX_4090",
        max_hourly_rate=0.50,
    )

    # Create instance from best offer
    instance = await provider.create_instance(
        offer_id=offers[0].offer_id,
        image="pytorch/pytorch:latest",
        disk_gb=100,
    )

    # Wait for ready
    await provider.wait_for_ready(instance.instance_id)

    # Execute command
    exit_code, stdout, stderr = await provider.execute_command(
        instance.instance_id,
        "python train.py",
    )

asyncio.run(main())
```

## Configuration

Environment variables (prefix with `CCM_`):

| Variable | Description | Default |
|----------|-------------|---------|
| `CCM_VAST_API_KEY` | Vast.ai API key | (from ~/.vast_api_key) |
| `CCM_DATA_DIR` | Data directory | `~/.cloudcomputemanager` |
| `CCM_DATABASE_URL` | Database URL | `sqlite:///DATA_DIR/ccm.db` |
| `CCM_CHECKPOINT_STORAGE` | Checkpoint storage | `local` |
| `CCM_API_HOST` | API host | `127.0.0.1` |
| `CCM_API_PORT` | API port | `8765` |

View configuration:

```bash
ccm config show
```

## Architecture

```
CloudComputeManager
├── CLI (Typer + Rich)
│   ├── jobs      - Job management
│   ├── batch     - Batch operations
│   ├── instances - Instance control
│   ├── templates - Template management
│   └── daemon    - Background service
├── Daemon
│   ├── monitor   - Job monitoring loop
│   └── service   - Daemon lifecycle
├── Core
│   ├── models    - Job, Instance, Checkpoint
│   ├── database  - Async SQLModel/SQLite
│   ├── recovery  - Preemption recovery
│   ├── cleanup   - Stale job cleanup
│   └── templates - Job templates
├── Providers
│   └── vast      - Vast.ai API integration
└── REST API (FastAPI)
    └── /v1/*     - Job and instance endpoints
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run specific test file
pytest tests/test_templates.py -v

# Type checking
mypy src/cloudcomputemanager

# Linting
ruff check src/

# Format code
ruff format src/
```

## License

MIT License
