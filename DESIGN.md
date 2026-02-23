# CloudComputeManager: A Robust GPU Cloud Management Platform

## Design Document v1.0

**Author**: Claude Code
**Date**: 2026-02-23
**Status**: Draft

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Existing Solutions Analysis](#existing-solutions-analysis)
4. [Design Goals & Principles](#design-goals--principles)
5. [Proposed Architecture](#proposed-architecture)
6. [Key Components](#key-components)
7. [API Design](#api-design)
8. [Data Models](#data-models)
9. [Failure Recovery & Fault Tolerance](#failure-recovery--fault-tolerance)
10. [Implementation Roadmap](#implementation-roadmap)
11. [Tech Stack Recommendations](#tech-stack-recommendations)
12. [Security Considerations](#security-considerations)
13. [Future Extensions](#future-extensions)

---

## Executive Summary

CloudComputeManager is a robust, extensible platform for managing GPU workloads on Vast.ai (and potentially other GPU cloud providers). It addresses the critical challenges of running long-running computational jobs on interruptible/spot instances, including automatic checkpointing, result syncing, failure recovery, and cost optimization.

The platform is designed to be **agent-native**, meaning AI agents can programmatically control all aspects of job submission, monitoring, and recovery. This aligns with modern AI-driven infrastructure automation trends while remaining fully usable by human operators.

### Key Value Propositions

- **Zero-loss preemption handling**: Automatic checkpointing ensures no work is lost when spot instances are interrupted
- **Set-and-forget job submission**: Submit jobs and walk away; the system handles all failure scenarios
- **Cost optimization**: Intelligent bidding and instance selection to minimize costs
- **Universal workload support**: Extensible architecture supports any containerized workload, not just specific use cases
- **Agent-first API**: Every feature accessible programmatically for AI agent integration

---

## Problem Statement

### Current Pain Points (From Experience)

We have been running LAMMPS molecular dynamics simulations on Vast.ai spot instances. The major challenges encountered include:

1. **Spot Instance Preemption**: Vast.ai interruptible instances can be stopped at any time when a higher bidder arrives or on-demand demand increases. When this happens, all in-progress work is lost.

2. **Manual Monitoring Burden**: Currently relying on manual SSH checks to verify job status, which is time-consuming and error-prone.

3. **Fragile Data Synchronization**: Using manual rsync scripts to pull results, which requires human intervention and can miss critical data before preemption.

4. **No Automatic Recovery**: When instances fail or are preempted, jobs must be manually restarted, often from scratch if checkpoints weren't saved.

5. **Lack of Job Queuing**: No system to queue multiple jobs and automatically provision instances for them.

6. **Cost Tracking Difficulty**: Hard to track actual costs per job/project without manual bookkeeping.

### The Broader Challenge

These problems are not unique to LAMMPS or molecular dynamics. Any researcher or engineer running long-running GPU workloads on spot instances faces similar challenges:

- Machine learning training jobs (hours to days)
- Scientific simulations (HPC workloads)
- Batch inference pipelines
- Rendering workloads
- Data processing jobs

The lack of a robust management layer between the user and the raw cloud API creates significant operational overhead and risk of lost work.

---

## Existing Solutions Analysis

### Commercial GPU Cloud Platforms

#### Modal.com
**Approach**: Python-first serverless GPU platform with decorators

**Strengths**:
- Excellent developer experience with Python SDK
- 1-2 second cold starts with custom infrastructure
- Cloud-agnostic (runs across multiple providers)
- Built-in persistent volumes and secrets management
- Automatic scaling to zero

**Weaknesses**:
- Opinionated framework requires code adaptation
- Higher abstraction may not suit all workloads
- Limited control over underlying infrastructure
- No support for raw container workloads

**Relevance**: Modal's Python SDK design is an excellent model for developer experience, but their approach of requiring code modification (decorators) is not suitable for legacy scientific software.

#### RunPod
**Approach**: GPU cloud with both serverless and pod-based options

**Strengths**:
- 48% cold starts under 200ms (FlashBoot technology)
- Multi-GPU support (up to 10x 24GB GPUs)
- Comprehensive monitoring dashboard
- SOC2 Type 1 certified
- No data transfer fees

**Weaknesses**:
- Limited spot/interruptible options compared to Vast.ai
- Higher baseline costs than Vast.ai
- Less flexible for custom workloads

**Relevance**: RunPod's monitoring dashboard and API design provide good examples, but their pricing structure is less favorable for long-running batch jobs.

#### Lambda Labs
**Approach**: Traditional cloud VM model with GPU focus

**Strengths**:
- Pre-installed Lambda Stack (CUDA, PyTorch, etc.)
- 1-Click Clusters for large-scale training
- Enterprise-grade infrastructure

**Weaknesses**:
- No spot/interruptible pricing (deprecated inference API)
- Higher costs than marketplace models
- Limited automation capabilities

**Relevance**: Lambda's approach to pre-built environments is useful for quick starts, but lacks the cost optimization needed for budget-conscious research.

### Open-Source Orchestration Tools

#### SkyPilot
**Approach**: Multi-cloud GPU orchestration framework from UC Berkeley

**Strengths**:
- Supports 20+ clouds including Vast.ai
- YAML-based job definitions
- Built-in spot instance management with retry
- Cost-aware scheduling across providers
- Active development and community (v0.11 released Dec 2025)
- Managed job pools feature

**Weaknesses**:
- Designed for multi-cloud, may be overkill for single-provider use
- Learning curve for YAML configuration
- Less integrated with Vast.ai-specific features (autogroups, etc.)

**Relevance**: SkyPilot is the closest existing solution to our needs. Key considerations:
- We could build on top of SkyPilot as a foundation
- Or build a more focused, Vast.ai-native solution
- SkyPilot's fault tolerance patterns are worth adopting

#### NVIDIA KAI Scheduler
**Approach**: Kubernetes-native GPU scheduler

**Strengths**:
- Enterprise-scale GPU cluster management
- Advanced scheduling policies
- Open source

**Weaknesses**:
- Requires Kubernetes infrastructure
- Overkill for cloud GPU rental scenarios

**Relevance**: Not directly applicable, but scheduling algorithms could inform our design.

### GPU Checkpointing Technologies

#### CRIU + cuda-checkpoint
**Status**: Production-ready as of CRIU 4.0+ (2025)

**Capabilities**:
- Transparent checkpoint/restore of CUDA applications
- Supports NVIDIA H100, A100 and AMD MI210
- Integrates with container runtimes (Podman)
- Evaluated on LLaMA 3.1 (8B), GPT-2 (1.5B) training

**Limitations**:
- Requires NVIDIA driver 550+
- No GPU migration support yet
- Must wait for in-flight CUDA work to complete

**Relevance**: This is a key enabling technology. For workloads that support it, transparent checkpointing eliminates the need for application-level checkpoint code.

### Vast.ai Native Capabilities

**Current API Features**:
- Full instance lifecycle management (create, start, stop, destroy)
- Autoscaling groups for automatic worker management
- Cloud copy for S3/GCS transfers
- Interruptible (spot) and on-demand pricing
- SSH key management
- Volume management for persistent storage

**Gaps**:
- No built-in job queue or orchestration
- No automatic checkpointing integration
- No webhook/callback system for events
- No native monitoring beyond basic stats
- Manual intervention required for recovery

---

## Design Goals & Principles

### Primary Goals

1. **Resilience**: Jobs should complete successfully despite any number of instance failures or preemptions

2. **Simplicity**: Users should be able to submit jobs with minimal configuration and trust the system to handle complexity

3. **Visibility**: Clear, real-time visibility into job status, costs, and instance health

4. **Extensibility**: Support any containerized workload without code modification

5. **Agent-Native**: Every feature accessible via clean APIs for AI agent integration

### Design Principles

1. **Convention over Configuration**: Sensible defaults that work for 90% of cases, with full customization available

2. **Fail-Safe by Default**: Always err on the side of preserving data and work

3. **Idempotent Operations**: All operations should be safe to retry

4. **Event-Driven Architecture**: React to instance events rather than polling

5. **Separation of Concerns**: Clean boundaries between orchestration, monitoring, and workload execution

6. **Infrastructure as Code**: All configuration should be version-controllable

### Provider Strategy: Vast.ai First, Extensible by Design

**Approach**: Build for Vast.ai first, but with a clean provider abstraction layer that enables future multi-provider support without core rewrites.

**Rationale**:
- **YAGNI (You Ain't Gonna Need It)**: We're using Vast.ai now; building multiple adapters before shipping delays value delivery
- **Learn the right abstraction**: Building one adapter teaches what the interface should actually look like
- **Avoid tight coupling**: Provider-specific code stays in adapters, core logic remains portable
- **Future-proof**: Adding RunPod/Lambda later = implement new adapter, no core changes

**Implementation**:
```
Core Logic (jobs, sync, checkpoints)  ← Provider-agnostic
              │
              ▼
     Provider Interface               ← Clean 10-method contract
              │
              ▼
┌─────────────┬─────────────┬─────────────┐
│  Vast.ai    │   RunPod    │   Lambda    │
│  (Phase 1)  │  (Phase 5)  │  (Phase 5)  │
└─────────────┴─────────────┴─────────────┘
```

This adds ~5-10% upfront design overhead but prevents massive refactoring when extending to other providers.

---

## Proposed Architecture

### High-Level Architecture

```
+------------------------------------------------------------------+
|                        CloudComputeManager Platform                        |
+------------------------------------------------------------------+
|                                                                    |
|  +--------------------+    +--------------------+                  |
|  |    API Gateway     |    |   Web Dashboard    |                  |
|  |   (REST + gRPC)    |    |    (Optional)      |                  |
|  +--------------------+    +--------------------+                  |
|           |                         |                              |
|           v                         v                              |
|  +--------------------------------------------------+              |
|  |              Core Orchestration Engine           |              |
|  |  +------------+  +------------+  +------------+  |              |
|  |  |    Job     |  |  Instance  |  |   Cost     |  |              |
|  |  |  Scheduler |  |  Manager   |  |  Tracker   |  |              |
|  |  +------------+  +------------+  +------------+  |              |
|  +--------------------------------------------------+              |
|           |                                                        |
|           v                                                        |
|  +--------------------------------------------------+              |
|  |              Plugin/Adapter Layer                |              |
|  |  +------------+  +------------+  +------------+  |              |
|  |  | Checkpoint |  |    Sync    |  |  Monitor   |  |              |
|  |  |   Engine   |  |   Engine   |  |   Agent    |  |              |
|  |  +------------+  +------------+  +------------+  |              |
|  +--------------------------------------------------+              |
|           |                                                        |
|           v                                                        |
|  +--------------------------------------------------+              |
|  |              Cloud Provider Adapters              |              |
|  |  +------------+  +------------+  +------------+  |              |
|  |  |  Vast.ai   |  |   RunPod   |  |   Lambda   |  |              |
|  |  |  Adapter   |  |  Adapter   |  |   Adapter  |  |              |
|  |  +------------+  +------------+  +------------+  |              |
|  +--------------------------------------------------+              |
|                                                                    |
+------------------------------------------------------------------+
           |                    |                    |
           v                    v                    v
    +------------+       +------------+       +------------+
    |  Vast.ai   |       |   RunPod   |       |   Lambda   |
    |   Cloud    |       |   Cloud    |       |   Cloud    |
    +------------+       +------------+       +------------+
```

### Component Interaction Flow

```
User/Agent                CloudComputeManager                 Vast.ai
    |                          |                         |
    |-- Submit Job ----------->|                         |
    |                          |-- Search Offers ------->|
    |                          |<-- Offer List ----------|
    |                          |-- Select Best Offer --->|
    |                          |-- Create Instance ----->|
    |                          |<-- Instance ID ---------|
    |                          |-- Deploy Workload ----->|
    |                          |-- Start Monitoring ---->|
    |<-- Job Submitted --------|                         |
    |                          |                         |
    |                          |<-- Instance Event ------|
    |                          |   (preemption warning)  |
    |                          |                         |
    |                          |-- Trigger Checkpoint -->|
    |                          |-- Sync Data ----------->|
    |                          |-- Search New Offers --->|
    |                          |<-- New Offer List ------|
    |                          |-- Create New Instance ->|
    |                          |-- Restore Checkpoint -->|
    |                          |-- Resume Workload ----->|
    |                          |                         |
    |<-- Job Status Update ----|                         |
    |                          |                         |
    |                          |<-- Job Completed -------|
    |                          |-- Final Sync ---------->|
    |                          |-- Terminate Instance -->|
    |<-- Job Completed --------|                         |
```

### Deployment Modes

#### Mode 1: Self-Hosted (Primary)
- CloudComputeManager runs on user's local machine or a control server
- Connects to Vast.ai via API
- Stores state in local SQLite or PostgreSQL

#### Mode 2: Agent-Sidecar
- Lightweight agent runs inside GPU instances
- Reports back to control plane
- Handles local checkpoint/sync operations

#### Mode 3: Serverless Control Plane (Future)
- CloudComputeManager core runs as serverless functions
- Event-driven architecture
- No always-on infrastructure required

---

## Key Components

### 1. Job Scheduler

**Purpose**: Manage the lifecycle of computational jobs from submission to completion.

**Responsibilities**:
- Accept job definitions (container image, command, resources)
- Queue jobs when no instances available
- Match jobs to available offers based on requirements
- Handle job dependencies and workflows
- Track job state transitions

**Key Features**:
- Priority queues for different job types
- Resource requirement matching (GPU type, VRAM, disk)
- Fair scheduling across multiple users/projects
- Job timeout and resource limit enforcement

**State Machine**:
```
PENDING -> PROVISIONING -> RUNNING -> CHECKPOINTING -> COMPLETED
              |              |            |
              v              v            v
           FAILED <------ FAILED <----- FAILED
              |              |            |
              v              v            v
          RETRYING ----> RETRYING ----> RETRYING
```

### 2. Instance Manager

**Purpose**: Manage GPU instance lifecycle and health.

**Responsibilities**:
- Search and evaluate available offers
- Create/destroy instances
- Monitor instance health and status
- Handle preemption events
- Manage bid prices for spot instances

**Key Features**:
- Smart offer selection (cost vs. reliability vs. performance)
- Automatic bid adjustment based on market conditions
- Instance pooling for reduced cold start times
- Multi-instance support for distributed jobs

**Offer Scoring Algorithm**:
```python
score = (
    w_cost * (1 / hourly_cost) +
    w_reliability * reliability_score +
    w_performance * (gpu_performance / required_performance) +
    w_location * location_preference +
    w_availability * availability_probability
)
```

### 3. Checkpoint Engine

**Purpose**: Ensure no work is lost during instance interruptions.

**Checkpoint Strategies** (Plugin-Based):

#### Strategy A: Application-Level Checkpointing
- Workload implements its own checkpoint/restore
- CloudComputeManager triggers checkpoint on signal
- Most reliable but requires app support

#### Strategy B: CRIU/cuda-checkpoint Integration
- Transparent checkpointing of entire container
- No application modification required
- Requires compatible drivers and frameworks

#### Strategy C: Filesystem-Level Snapshots
- Periodic sync of working directory
- Simple but may capture inconsistent state
- Best for fault-tolerant workloads

#### Strategy D: Hybrid Approach
- Combine app-level checkpoints with filesystem sync
- App checkpoints for clean state
- Filesystem sync for intermediate results

**Checkpoint Storage Options**:
- Local volume (Vast.ai persistent storage)
- S3-compatible object storage
- rsync.net / B2 / Wasabi
- Google Cloud Storage

### 4. Sync Engine

**Purpose**: Keep data synchronized between instances and persistent storage.

**Responsibilities**:
- Continuous/periodic sync of results
- Checkpoint upload/download
- Input data staging
- Output data collection

**Sync Methods**:

| Method | Use Case | Implementation |
|--------|----------|----------------|
| rclone | S3/GCS/B2 | Incremental, resumable |
| rsync | SSH targets | Delta sync |
| Vast cloud copy | Provider-native | Fast but limited |
| Custom | Specialized needs | Plugin interface |

**Sync Policies**:
- **Real-time**: inotify-based, sync on write
- **Periodic**: Cron-like scheduled sync
- **Checkpoint-triggered**: Sync after each checkpoint
- **Manual**: On-demand sync

### 5. Monitor Agent

**Purpose**: Collect and report instance/job health metrics.

**Metrics Collected**:
- GPU utilization and memory
- CPU and system memory
- Disk I/O and space
- Network throughput
- Application-specific metrics (if exposed)
- Estimated time to completion

**Alert Triggers**:
- Disk space low (configurable threshold)
- GPU utilization anomaly (stuck job detection)
- Preemption warning (if available)
- Instance health degradation
- Cost threshold exceeded

### 6. Cost Tracker

**Purpose**: Track and optimize spending.

**Features**:
- Per-job cost attribution
- Per-project cost aggregation
- Real-time cost projection
- Budget alerts and limits
- Historical cost analysis
- Cost optimization recommendations

**Cost Model**:
```python
job_cost = (
    instance_hours * hourly_rate +
    storage_gb * storage_rate +
    network_egress_gb * egress_rate +
    overhead_percentage
)
```

### 7. Cloud Provider Adapter (Vast.ai)

**Purpose**: Abstract Vast.ai-specific API interactions.

**API Mapping**:

| CloudComputeManager Operation | Vast.ai API |
|----------------------|-------------|
| search_offers() | search offers |
| create_instance() | create instance |
| destroy_instance() | destroy instance |
| get_instance_status() | show instances |
| execute_command() | execute |
| sync_data() | cloud copy |
| get_logs() | logs |
| change_bid() | change bid |

**Error Handling**:
- Retry with exponential backoff
- Automatic failover to alternative offers
- Graceful degradation on API limits

---

## API Design

### Design Philosophy

The API follows REST principles with clear resource hierarchies. Every operation is also available via a Python SDK for agent integration.

### Base URL Structure

```
https://api.cloudcomputemanager.local/v1/
```

### Authentication

```http
Authorization: Bearer <api_key>
```

API keys support scoping:
- `jobs:read` - Read job status
- `jobs:write` - Submit/modify jobs
- `instances:read` - View instances
- `instances:write` - Manage instances
- `admin` - Full access

### Core Resources

#### Jobs

```http
# List all jobs
GET /v1/jobs
GET /v1/jobs?status=running&project=lammps

# Get job details
GET /v1/jobs/{job_id}

# Submit new job
POST /v1/jobs
{
  "name": "lammps-simulation-001",
  "project": "mxenes-study",
  "image": "lammps/lammps:latest",
  "command": "mpirun -np 4 lmp -in simulation.in",
  "resources": {
    "gpu_type": "RTX_4090",
    "gpu_count": 1,
    "gpu_memory_min": 16,
    "disk_gb": 50,
    "cpu_cores": 8,
    "memory_gb": 32
  },
  "checkpoint": {
    "strategy": "application",
    "signal": "SIGUSR1",
    "interval_minutes": 30,
    "path": "/workspace/checkpoints"
  },
  "sync": {
    "source": "/workspace/results",
    "destination": "s3://my-bucket/results/",
    "interval_minutes": 15
  },
  "input_data": {
    "source": "s3://my-bucket/inputs/simulation.in",
    "destination": "/workspace/"
  },
  "budget": {
    "max_cost_usd": 50.00,
    "max_hours": 24
  },
  "retry": {
    "max_attempts": 5,
    "backoff_minutes": 5
  },
  "priority": "normal",
  "tags": ["simulation", "mxenes", "phase-1"]
}

# Response
{
  "job_id": "job_abc123",
  "status": "pending",
  "created_at": "2026-02-23T10:00:00Z",
  "estimated_cost": 12.50,
  "estimated_duration_hours": 8
}

# Cancel job
DELETE /v1/jobs/{job_id}

# Retry failed job
POST /v1/jobs/{job_id}/retry

# Get job logs
GET /v1/jobs/{job_id}/logs?tail=1000

# Get job metrics
GET /v1/jobs/{job_id}/metrics

# Trigger manual checkpoint
POST /v1/jobs/{job_id}/checkpoint

# Trigger manual sync
POST /v1/jobs/{job_id}/sync
```

#### Instances

```http
# List managed instances
GET /v1/instances

# Get instance details
GET /v1/instances/{instance_id}

# Create standalone instance (not tied to job)
POST /v1/instances
{
  "offer_id": "12345",
  "image": "pytorch/pytorch:latest",
  "disk_gb": 100,
  "ssh_key": "ssh-rsa AAAA..."
}

# Execute command on instance
POST /v1/instances/{instance_id}/exec
{
  "command": "nvidia-smi"
}

# Terminate instance
DELETE /v1/instances/{instance_id}
```

#### Offers

```http
# Search available offers
GET /v1/offers
GET /v1/offers?gpu_type=RTX_4090&min_gpu_memory=16&max_price=0.50

# Get offer details
GET /v1/offers/{offer_id}

# Get price history
GET /v1/offers/{offer_id}/price_history
```

#### Projects

```http
# List projects
GET /v1/projects

# Create project
POST /v1/projects
{
  "name": "mxenes-study",
  "budget_usd": 500.00,
  "default_resources": {...}
}

# Get project statistics
GET /v1/projects/{project_id}/stats
```

#### Webhooks (Event Notifications)

```http
# Register webhook
POST /v1/webhooks
{
  "url": "https://my-server.com/webhooks/cloudcomputemanager",
  "events": ["job.completed", "job.failed", "instance.preempted"],
  "secret": "webhook_secret_123"
}

# Webhook payload example
{
  "event": "job.completed",
  "timestamp": "2026-02-23T18:00:00Z",
  "data": {
    "job_id": "job_abc123",
    "status": "completed",
    "duration_hours": 8.5,
    "total_cost_usd": 14.25,
    "output_location": "s3://my-bucket/results/job_abc123/"
  }
}
```

### Python SDK Usage

```python
from cloudcomputemanager import CloudComputeManager, Job, Resources, Checkpoint, Sync

# Initialize client
vm = CloudComputeManager(api_key="your_api_key")

# Define job
job = Job(
    name="lammps-simulation-001",
    project="mxenes-study",
    image="lammps/lammps:latest",
    command="mpirun -np 4 lmp -in simulation.in",
    resources=Resources(
        gpu_type="RTX_4090",
        gpu_count=1,
        gpu_memory_min=16,
        disk_gb=50
    ),
    checkpoint=Checkpoint(
        strategy="application",
        signal="SIGUSR1",
        interval_minutes=30,
        path="/workspace/checkpoints"
    ),
    sync=Sync(
        source="/workspace/results",
        destination="s3://my-bucket/results/",
        interval_minutes=15
    )
)

# Submit job
submitted_job = vm.jobs.submit(job)
print(f"Job submitted: {submitted_job.job_id}")

# Monitor job
for event in vm.jobs.watch(submitted_job.job_id):
    print(f"Status: {event.status}, Progress: {event.progress}%")
    if event.status == "completed":
        break

# Get results
results = vm.jobs.get_results(submitted_job.job_id)
print(f"Output: {results.output_location}")
print(f"Cost: ${results.total_cost_usd}")
```

### Agent Integration Example

```python
# Example: AI agent submitting simulation jobs

async def run_simulation_campaign(parameters: list[dict]) -> list[dict]:
    """AI agent function to run multiple simulations."""

    vm = CloudComputeManager(api_key=os.environ["CCM_API_KEY"])

    jobs = []
    for params in parameters:
        job = Job(
            name=f"sim-{params['id']}",
            image="lammps/lammps:latest",
            command=f"lmp -in input.in -var temp {params['temperature']}",
            resources=Resources(gpu_type="RTX_4090"),
            checkpoint=Checkpoint(strategy="application", interval_minutes=30)
        )
        submitted = await vm.jobs.submit_async(job)
        jobs.append(submitted)

    # Wait for all jobs
    results = await vm.jobs.wait_all([j.job_id for j in jobs])

    return [
        {
            "id": params["id"],
            "status": result.status,
            "output": result.output_location,
            "cost": result.total_cost_usd
        }
        for params, result in zip(parameters, results)
    ]
```

---

## Data Models

### Job Model

```python
@dataclass
class Job:
    job_id: str
    name: str
    project: Optional[str]
    status: JobStatus  # pending, provisioning, running, checkpointing, completed, failed

    # Container configuration
    image: str
    command: str
    environment: dict[str, str]
    volumes: list[VolumeMount]

    # Resource requirements
    resources: Resources

    # Checkpoint configuration
    checkpoint: Optional[Checkpoint]

    # Sync configuration
    sync: Optional[Sync]

    # Input data
    input_data: Optional[DataSource]

    # Budget constraints
    budget: Optional[Budget]

    # Retry policy
    retry: RetryPolicy

    # Metadata
    priority: Priority  # low, normal, high, critical
    tags: list[str]
    labels: dict[str, str]

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    # Execution details
    instance_id: Optional[str]
    attempt_number: int
    checkpoints: list[CheckpointRecord]

    # Results
    output_location: Optional[str]
    exit_code: Optional[int]
    total_cost_usd: float
    total_runtime_seconds: int
```

### Instance Model

```python
@dataclass
class Instance:
    instance_id: str
    offer_id: str
    status: InstanceStatus  # creating, running, stopping, stopped, error

    # Hardware specs
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: int
    memory_gb: int
    disk_gb: int

    # Networking
    ssh_host: str
    ssh_port: int
    jupyter_url: Optional[str]

    # Pricing
    rental_type: RentalType  # interruptible, on_demand
    hourly_rate: float
    current_bid: Optional[float]

    # Association
    job_id: Optional[str]

    # Health
    last_health_check: datetime
    health_status: HealthStatus

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
```

### Checkpoint Record

```python
@dataclass
class CheckpointRecord:
    checkpoint_id: str
    job_id: str
    instance_id: str

    # Checkpoint details
    strategy: CheckpointStrategy
    trigger: CheckpointTrigger  # scheduled, preemption, manual

    # Storage
    location: str  # s3://bucket/path or local path
    size_bytes: int

    # Timing
    created_at: datetime
    duration_seconds: int

    # State
    application_state: Optional[dict]  # App-reported checkpoint metadata

    # Validity
    verified: bool
    verification_error: Optional[str]
```

---

## Failure Recovery & Fault Tolerance

### Failure Scenarios and Handling

#### Scenario 1: Spot Instance Preemption

**Detection**:
- Vast.ai API status change to "stopped"
- SSH connection failure
- Monitor agent heartbeat timeout

**Recovery Process**:
1. Detect preemption event
2. Mark job as "recovering"
3. Verify last checkpoint integrity
4. Search for new suitable offers
5. Create new instance
6. Restore from checkpoint
7. Resume job execution
8. Update job metrics (attempt count, etc.)

**Time Budget**:
- Detection: < 30 seconds
- Checkpoint verification: < 60 seconds
- New instance provisioning: 1-5 minutes
- Checkpoint restoration: 1-10 minutes (depends on size)

#### Scenario 2: Instance Hardware Failure

**Detection**:
- GPU errors in logs
- Monitor agent reports hardware issues
- Job progress stalls

**Recovery Process**:
1. Attempt graceful checkpoint
2. If checkpoint fails, use last good checkpoint
3. Destroy failed instance
4. Provision new instance
5. Restore and resume

#### Scenario 3: Network Partition

**Detection**:
- Monitor agent heartbeat timeout
- SSH connection failures
- API timeouts

**Recovery Process**:
1. Exponential backoff reconnection attempts
2. If instance reachable, continue monitoring
3. If unreachable for threshold period, assume lost
4. Trigger preemption recovery flow

#### Scenario 4: Application Crash

**Detection**:
- Process exit with non-zero code
- Monitor agent detects process death

**Recovery Process**:
1. Collect crash logs
2. Determine if retry is appropriate
3. If retriable, restore from checkpoint and retry
4. If not retriable, mark job as failed

#### Scenario 5: Disk Full

**Detection**:
- Monitor agent disk space alerts
- Write failures in application logs

**Recovery Process**:
1. Trigger emergency sync of critical data
2. Clear temporary files
3. If insufficient, checkpoint and migrate to larger instance

### Checkpoint Verification

Every checkpoint must be verified before relying on it:

```python
async def verify_checkpoint(checkpoint: CheckpointRecord) -> bool:
    """Verify checkpoint integrity and restorability."""

    # 1. Verify file exists and is complete
    if not await storage.exists(checkpoint.location):
        return False

    actual_size = await storage.get_size(checkpoint.location)
    if actual_size != checkpoint.size_bytes:
        return False

    # 2. Verify checksum
    actual_hash = await storage.compute_hash(checkpoint.location)
    if actual_hash != checkpoint.hash:
        return False

    # 3. Application-specific verification (if available)
    if checkpoint.application_state:
        try:
            await verify_app_checkpoint(checkpoint)
        except VerificationError:
            return False

    return True
```

### Idempotency Guarantees

All operations are designed to be safely retriable:

- Job submission with same name returns existing job
- Checkpoint uploads use content-addressable storage
- Instance creation is tracked to prevent duplicates
- Sync operations verify before overwriting

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goal**: Basic job submission and monitoring for Vast.ai

**Deliverables**:
- [ ] Core Python package structure
- [ ] Vast.ai adapter (instance lifecycle)
- [ ] Basic job scheduler (single job at a time)
- [ ] SQLite state persistence
- [ ] CLI for job submission and monitoring
- [ ] Simple checkpoint support (filesystem-based)
- [ ] Manual rsync-based sync

**Milestone**: Submit a LAMMPS job, have it run to completion, results synced

### Phase 2: Resilience (Weeks 5-8)

**Goal**: Automatic failure recovery

**Deliverables**:
- [ ] Preemption detection and recovery
- [ ] Automatic retry with exponential backoff
- [ ] S3-compatible checkpoint storage
- [ ] Checkpoint verification
- [ ] Instance health monitoring
- [ ] Alert system (email/webhook)

**Milestone**: Job survives 3 preemptions and completes successfully

### Phase 3: Optimization (Weeks 9-12)

**Goal**: Cost optimization and scaling

**Deliverables**:
- [ ] Smart offer selection algorithm
- [ ] Bid management for spot instances
- [ ] Job queuing (multiple concurrent jobs)
- [ ] Cost tracking and reporting
- [ ] Budget limits and alerts
- [ ] Project-based organization

**Milestone**: Run 10 jobs in parallel, stay within budget

### Phase 4: API & Integration (Weeks 13-16)

**Goal**: Full API and agent integration

**Deliverables**:
- [ ] REST API server
- [ ] Python SDK
- [ ] Webhook system
- [ ] API authentication and authorization
- [ ] Rate limiting
- [ ] OpenAPI documentation

**Milestone**: AI agent successfully runs a simulation campaign

### Phase 5: Advanced Features (Weeks 17-20)

**Goal**: Production-ready platform

**Deliverables**:
- [ ] Web dashboard (optional)
- [ ] Multi-provider support (RunPod, Lambda)
- [ ] CRIU/cuda-checkpoint integration
- [ ] Distributed job support
- [ ] Advanced scheduling (priorities, dependencies)
- [ ] Audit logging

**Milestone**: Platform handles 100+ jobs/day reliably

### Phase 6: Polish & Documentation (Weeks 21-24)

**Goal**: Production release

**Deliverables**:
- [ ] Comprehensive documentation
- [ ] Tutorial and examples
- [ ] Performance optimization
- [ ] Security audit
- [ ] Container images for deployment
- [ ] CI/CD pipeline

**Milestone**: v1.0 release

---

## Tech Stack Recommendations

### Core Platform

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Ecosystem, SDK compatibility, agent integration |
| Async Framework | asyncio + httpx | Non-blocking I/O for concurrent operations |
| CLI Framework | Typer + Rich | Modern CLI with great UX |
| Database | SQLite (dev) / PostgreSQL (prod) | Simple start, scale when needed |
| ORM | SQLModel (SQLAlchemy + Pydantic) | Type safety, easy serialization |
| Task Queue | None initially / Celery (later) | Start simple, add if needed |

### API Layer

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Web Framework | FastAPI | Async, auto-docs, Pydantic integration |
| Serialization | Pydantic v2 | Performance, validation |
| API Docs | OpenAPI 3.1 | Standard, tooling support |
| Authentication | API keys + JWT | Simple, secure |

### Data & Storage

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| State DB | SQLite/PostgreSQL | Reliable, transactional |
| Checkpoint Storage | S3-compatible | Universal, cheap |
| Sync Tool | rclone | Multi-provider, battle-tested |
| Cache | Redis (optional) | If needed for scale |

### Monitoring & Observability

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Logging | structlog | Structured, JSON-ready |
| Metrics | Prometheus client | Standard, extensive tooling |
| Tracing | OpenTelemetry (optional) | Distributed tracing if needed |

### Deployment

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Packaging | Docker | Portable, reproducible |
| Config | Pydantic Settings | Type-safe env vars |
| Process Manager | systemd / supervisord | Simple, reliable |

### Development

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Testing | pytest + pytest-asyncio | Standard, async support |
| Mocking | respx (for httpx) | HTTP mocking |
| Linting | ruff | Fast, comprehensive |
| Type Checking | mypy / pyright | Catch bugs early |
| Docs | mkdocs-material | Beautiful, searchable |

### Dependencies Summary

```toml
[project]
dependencies = [
    # Core
    "httpx>=0.27.0",
    "pydantic>=2.5.0",
    "sqlmodel>=0.0.16",

    # CLI
    "typer>=0.9.0",
    "rich>=13.0.0",

    # API
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",

    # Storage/Sync
    "boto3>=1.34.0",  # S3 operations

    # Utilities
    "structlog>=24.1.0",
    "python-dotenv>=1.0.0",
    "tenacity>=8.2.0",  # Retry logic
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "respx>=0.20.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
]
```

---

## Security Considerations

### API Key Management

- API keys stored encrypted at rest
- Support for key rotation
- Scoped keys with minimal necessary permissions
- Key expiration support

### Secrets Handling

- Environment variables for credentials
- Integration with secret managers (optional)
- Never log sensitive values
- Secrets masked in CLI output

### Instance Security

- SSH key-based authentication only
- Optional VPN/WireGuard for instance communication
- Firewall rules for instance access
- Audit logging of all instance operations

### Data Security

- Encryption in transit (TLS 1.3)
- Optional encryption at rest for checkpoints
- Data retention policies
- Secure deletion of temporary data

### Access Control

- Role-based access (admin, user, viewer)
- Project-level isolation
- Audit trail for all operations

---

## Future Extensions

### Multi-Provider Support

Extend beyond Vast.ai to:
- RunPod (serverless GPU)
- Lambda Labs (on-demand)
- AWS/GCP/Azure spot instances
- On-premise GPU clusters

### Advanced Scheduling

- Job dependencies (DAG-based workflows)
- Time-based scheduling (cron)
- Resource-aware bin packing
- Preemptible job demotion

### Distributed Training Support

- Multi-node job coordination
- Collective checkpoint/restore
- Elastic training (node add/remove)
- NCCL/MPI integration

### Smart Optimization

- ML-based cost prediction
- Automatic checkpoint interval tuning
- Predictive preemption avoidance
- Workload-aware instance selection

### Ecosystem Integrations

- MLflow/W&B experiment tracking
- Jupyter notebook support
- Git-based workflow triggers
- Slack/Discord notifications

### Managed Service Option

- Hosted control plane
- Team collaboration features
- Advanced analytics
- SLA guarantees

---

## Appendix A: LAMMPS-Specific Configuration

Example configuration for molecular dynamics simulations:

```yaml
# lammps-job.yaml
name: mxene-md-simulation
project: mxenes-study

image: lammps/lammps:stable_29Aug2024_update1

command: |
  cd /workspace
  mpirun -np 4 lmp -in simulation.in -var restart_file restart.latest

resources:
  gpu_type: RTX_4090
  gpu_count: 1
  gpu_memory_min: 16
  disk_gb: 100
  cpu_cores: 8
  memory_gb: 64

checkpoint:
  strategy: application
  # LAMMPS restart file support
  path: /workspace/restarts
  interval_minutes: 30
  # Custom checkpoint command
  pre_checkpoint_command: |
    # Signal LAMMPS to write restart file
    touch /workspace/.write_restart
  # Verify restart file was written
  verify_command: |
    ls -la /workspace/restarts/restart.latest

sync:
  source: /workspace/results
  destination: s3://simulations/mxenes/${JOB_ID}/
  interval_minutes: 15
  # Sync patterns
  include:
    - "*.dump"
    - "*.log"
    - "thermo.dat"
  exclude:
    - "*.tmp"

input_data:
  - source: s3://simulations/inputs/simulation.in
    destination: /workspace/
  - source: s3://simulations/inputs/data.mxene
    destination: /workspace/

environment:
  OMP_NUM_THREADS: "2"
  LAMMPS_POTENTIALS: /opt/lammps/potentials

budget:
  max_cost_usd: 25.00
  max_hours: 12

retry:
  max_attempts: 10
  # LAMMPS can resume from restart files
  resume_from_checkpoint: true
```

---

## Appendix B: Agent Integration Patterns

### Pattern 1: Fire-and-Forget Batch Jobs

```python
async def submit_batch_simulations(params: list[SimParams]) -> str:
    """Submit batch and return batch ID for later checking."""
    vm = CloudComputeManager()
    batch = await vm.batches.create(
        name="param-sweep",
        jobs=[create_job(p) for p in params]
    )
    return batch.batch_id
```

### Pattern 2: Synchronous Execution

```python
async def run_simulation_sync(params: SimParams) -> SimResult:
    """Run simulation and wait for completion."""
    vm = CloudComputeManager()
    job = await vm.jobs.submit(create_job(params))
    result = await vm.jobs.wait(job.job_id, timeout=3600)
    return parse_results(result.output_location)
```

### Pattern 3: Streaming Progress

```python
async def run_with_progress(params: SimParams):
    """Stream progress updates."""
    vm = CloudComputeManager()
    job = await vm.jobs.submit(create_job(params))

    async for event in vm.jobs.stream(job.job_id):
        yield {
            "status": event.status,
            "progress": event.progress,
            "metrics": event.metrics
        }
```

### Pattern 4: Adaptive Campaigns

```python
async def adaptive_optimization(objective: Callable):
    """AI agent running adaptive optimization."""
    vm = CloudComputeManager()

    while not converged:
        # Get next parameters from optimizer
        params = optimizer.suggest()

        # Run simulation
        job = await vm.jobs.submit(create_job(params))
        result = await vm.jobs.wait(job.job_id)

        # Parse and report results
        value = parse_objective(result)
        optimizer.report(params, value)

        # Check convergence
        converged = optimizer.is_converged()

    return optimizer.best_params()
```

---

## Appendix C: Comparison Matrix

| Feature | CloudComputeManager | SkyPilot | Modal | Raw Vast.ai |
|---------|-------------|----------|-------|-------------|
| Vast.ai Native | Yes | Partial | No | Yes |
| Automatic Checkpointing | Yes | Partial | No | No |
| Spot Preemption Recovery | Yes | Yes | N/A | No |
| Cost Tracking | Yes | Yes | Yes | Basic |
| Job Queuing | Yes | Yes | Yes | No |
| Agent API | Yes | CLI | Yes | Yes |
| Multi-Provider | Planned | Yes | Yes | No |
| Setup Complexity | Low | Medium | Low | N/A |
| Learning Curve | Low | Medium | Low | Low |
| Code Modification Needed | No | Minimal | Yes | No |

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-23 | Claude Code | Initial design document |

---

*This design document serves as the foundation for CloudComputeManager development. It will be updated as implementation progresses and requirements evolve.*
