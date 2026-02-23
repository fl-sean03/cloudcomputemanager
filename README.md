# CloudComputeManager

A robust GPU cloud management platform with automatic checkpointing and spot instance recovery.

## Features

- **Automatic Checkpointing**: Detect and sync application checkpoints (LAMMPS, PyTorch, etc.)
- **Spot Instance Recovery**: Seamlessly recover from preemptions with checkpoint restore
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
cloudcomputemanager config init
```

## Quick Start

### 1. Configure Vast.ai API Key

```bash
# Option A: Create key file
echo "your-api-key" > ~/.vast_api_key

# Option B: Set environment variable
export CCM_VAST_API_KEY="your-api-key"
```

### 2. Search for GPU Offers

```bash
# Search for RTX 4090 instances
ccm search RTX_4090

# More detailed search
ccm instances search --gpu RTX_4090 --max-price 0.50 --limit 20
```

### 3. Submit a Job

Create a job configuration file (`job.yaml`):

```yaml
name: lammps-simulation
project: mxenes

image: lammps/lammps:stable_29Aug2024_update1

command: |
  cd /workspace
  mpirun -np 4 lmp -in simulation.in

resources:
  gpu_type: RTX_4090
  gpu_count: 1
  gpu_memory_min: 16
  disk_gb: 100

checkpoint:
  strategy: application
  path: /workspace
  interval_minutes: 30
  patterns:
    - "restart.*.bin"
    - "*.restart"

sync:
  enabled: true
  source: /workspace/results
  destination: ~/cloudcomputemanager_sync/
  interval_minutes: 15
  include_patterns:
    - "*.dump"
    - "*.log"
    - "thermo.dat"

budget:
  max_cost_usd: 25.00
  max_hours: 24
```

Submit the job:

```bash
ccm submit job.yaml

# Or with explicit command
ccm jobs submit job.yaml --name my-simulation
```

### 4. Monitor Job Status

```bash
# List all jobs
ccm status

# Get specific job status
ccm jobs status job_abc123

# View logs
ccm jobs logs job_abc123 --tail 100

# Trigger manual sync
ccm jobs sync job_abc123

# Trigger manual checkpoint
ccm jobs checkpoint job_abc123
```

### 5. Manage Instances

```bash
# List running instances
ccm instances list

# SSH into instance
ccm instances ssh 12345678

# Terminate instance
ccm instances terminate 12345678
```

## REST API

Start the API server:

```bash
# Using the CLI
ccm api start

# Or directly with uvicorn
uvicorn cloudcomputemanager.api.app:app --host 0.0.0.0 --port 8765
```

API endpoints:

- `GET /health` - Health check
- `POST /v1/jobs` - Submit job
- `GET /v1/jobs` - List jobs
- `GET /v1/jobs/{job_id}` - Get job details
- `DELETE /v1/jobs/{job_id}` - Cancel job
- `POST /v1/jobs/{job_id}/checkpoint` - Trigger checkpoint
- `POST /v1/jobs/{job_id}/sync` - Trigger sync
- `GET /v1/instances` - List instances
- `GET /v1/offers` - Search offers
- `GET /v1/checkpoints` - List checkpoints

API documentation available at `http://localhost:8765/docs`

## Python SDK

```python
import asyncio
from cloudcomputemanager.providers import VastProvider
from cloudcomputemanager.checkpoint import CheckpointOrchestrator
from cloudcomputemanager.sync import SyncEngine

async def main():
    # Initialize provider
    provider = VastProvider()

    # Search for offers
    offers = await provider.search_offers(
        gpu_type="RTX_4090",
        gpu_count=1,
        max_hourly_rate=0.50,
    )

    # Create instance
    instance = await provider.create_instance(
        offer_id=offers[0].offer_id,
        image="lammps/lammps:latest",
        disk_gb=100,
    )

    # Wait for ready
    await provider.wait_for_ready(instance.instance_id)

    # Execute command
    exit_code, stdout, stderr = await provider.execute_command(
        instance.instance_id,
        "nvidia-smi",
    )
    print(stdout)

    # Setup checkpoint monitoring
    orchestrator = CheckpointOrchestrator(provider)
    checkpoint = await orchestrator.get_latest_checkpoint(
        instance.instance_id,
        checkpoint_path="/workspace",
    )

asyncio.run(main())
```

## Configuration

All settings can be configured via environment variables with `CCM_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `CCM_VAST_API_KEY` | Vast.ai API key | (from ~/.vast_api_key) |
| `CCM_DATA_DIR` | Data directory | `~/.cloudcomputemanager` |
| `CCM_DATABASE_URL` | Database URL | `sqlite:///~/.cloudcomputemanager/cloudcomputemanager.db` |
| `CCM_CHECKPOINT_STORAGE` | Checkpoint storage | `local` |
| `CCM_SYNC_DEFAULT_INTERVAL` | Sync interval (seconds) | `300` |
| `CCM_API_HOST` | API host | `127.0.0.1` |
| `CCM_API_PORT` | API port | `8765` |
| `CCM_DEBUG` | Debug mode | `false` |

View current configuration:

```bash
ccm config show
```

## LAMMPS Integration

CloudComputeManager has specialized support for LAMMPS molecular dynamics simulations:

### Checkpoint Detection

CloudComputeManager automatically detects LAMMPS restart files:

- `restart.*.bin` - Numbered restart files
- `restart.bin` - Single restart file
- `*.restart` - Alternative restart format

### Auto-Restart

Configure your LAMMPS input file to support restart:

```lammps
# In your LAMMPS input file
variable restart_file index none

if "${restart_file} != none" then &
    "read_restart ${restart_file}" &
else &
    "read_data structure.data"

# Write restart files every 100k steps
restart 100000 restart.*.bin
```

### Progress Tracking

CloudComputeManager parses `log.lammps` to track simulation progress:

```bash
# View current progress
ccm jobs status job_abc123
```

## Architecture

```
CloudComputeManager
├── CLI (Typer + Rich)
│   └── Commands: submit, status, sync, checkpoint
├── REST API (FastAPI)
│   └── Routes: /v1/jobs, /v1/instances, /v1/offers
├── Core
│   ├── Job Manager - Job lifecycle
│   ├── Checkpoint Orchestrator - Save/restore
│   └── Sync Engine - Continuous sync
└── Providers
    └── Vast.ai (via vastai CLI)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/cloudcomputemanager

# Linting
ruff check src/

# Format code
ruff format src/
```

## License

MIT License
