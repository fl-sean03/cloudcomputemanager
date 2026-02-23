# CloudComputeManager - Agent Context

## Project Overview

**CloudComputeManager** (CCM) is a cloud compute management platform that provides:
- Automatic checkpointing for long-running jobs
- Continuous data synchronization with local storage
- Spot instance preemption recovery
- Agent-native APIs for AI integration
- PackStore for pre-built scientific computing packages

## Current Status

**Version**: 0.1.0 (Alpha)
**Tests**: 58/58 passing
**Provider**: Vast.ai (via CLI wrapper)

### Implemented Components

| Component | Status | Description |
|-----------|--------|-------------|
| Core Models | Complete | Job, Instance, Checkpoint, Sync models with SQLModel |
| VastProvider | Complete | Vast.ai API integration via CLI |
| Checkpoint Detection | Complete | LAMMPS, PyTorch, generic pattern detection |
| Sync Engine | Complete | Rsync-based continuous sync |
| Agent SDK | Complete | CloudComputeAgent async interface |
| PackStore | Complete | Package registry and deployment |
| CLI | Complete | Typer-based CLI with Rich output |
| REST API | Complete | FastAPI endpoints for all operations |

### Key Files

```
src/cloudcomputemanager/
├── __init__.py           # Main exports
├── core/
│   ├── config.py         # Settings (env prefix: CCM_)
│   ├── models.py         # SQLModel data models
│   └── database.py       # Async SQLite setup
├── providers/
│   ├── base.py           # Provider interface
│   └── vast.py           # Vast.ai implementation
├── checkpoint/
│   ├── detectors.py      # LAMMPS, PyTorch, generic
│   └── orchestrator.py   # Checkpoint orchestration
├── sync/
│   ├── engine.py         # Rsync wrapper
│   └── monitor.py        # Continuous sync
├── agents/
│   └── sdk.py            # CloudComputeAgent class
├── packstore/
│   ├── registry.py       # Package definitions
│   ├── detector.py       # GPU environment detection
│   └── deployer.py       # Package deployment
├── api/
│   ├── app.py            # FastAPI app factory
│   └── routes.py         # API routes
└── cli/
    └── main.py           # CLI commands
```

## Configuration

Environment variables use `CCM_` prefix:
- `CCM_VAST_API_KEY` - Vast.ai API key
- `CCM_DATA_DIR` - Data directory (~/.cloudcomputemanager)
- `CCM_DATABASE_URL` - SQLite database path
- `CCM_DEBUG` - Debug mode

## CLI Commands

```bash
ccm search RTX_4090           # Search GPU offers
ccm instances list            # List running instances
ccm packages list             # List available packages
ccm submit job.yaml           # Submit a job
ccm config show               # Show configuration
```

## Agent SDK Usage

```python
from cloudcomputemanager.agents import CloudComputeAgent, JobSpec

async with CloudComputeAgent() as agent:
    # Search for GPUs
    offers = await agent.search_gpus(gpu_type="RTX_4090", max_price=0.50)

    # Submit job with event streaming
    async for event in agent.submit_job(JobSpec(
        name="my-job",
        image="pytorch/pytorch:latest",
        command="python train.py",
    )):
        print(f"{event.type}: {event.message}")
```

## REST API

Start server: `uvicorn cloudcomputemanager.api.app:app --port 8765`

Key endpoints:
- `GET /health` - Health check
- `GET /v1/packages` - List packages
- `GET /v1/offers` - Search offers
- `POST /v1/jobs` - Submit job
- `GET /v1/instances` - List instances

## Testing

```bash
pytest tests/ -v           # Run all tests
pytest tests/ -k packstore # Run PackStore tests only
```

## Next Steps

1. Add more cloud providers (Lambda Labs, RunPod)
2. Implement job manager with database persistence
3. Add preemption detection and auto-recovery
4. S3/GCS checkpoint storage backends
5. Web dashboard UI
