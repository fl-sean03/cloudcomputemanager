"""API route handlers for CloudComputeManager."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from cloudcomputemanager import __version__
from cloudcomputemanager.core.database import get_session
from cloudcomputemanager.core.models import (
    Job,
    JobStatus,
    Checkpoint,
    Resources,
    CheckpointConfig,
    SyncConfig,
    Budget,
    RetryPolicy,
)
from cloudcomputemanager.providers.vast import VastProvider
from cloudcomputemanager.checkpoint import CheckpointOrchestrator

# ============================================================================
# Health Routes
# ============================================================================

health_router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime


@health_router.get("/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
    )


# ============================================================================
# Jobs Routes
# ============================================================================

jobs_router = APIRouter()


class JobSubmitRequest(BaseModel):
    """Request to submit a new job."""

    name: str = Field(..., description="Job name")
    project: Optional[str] = Field(None, description="Project name")
    image: str = Field(..., description="Docker image")
    command: str = Field(..., description="Command to run")
    resources: Resources = Field(default_factory=Resources)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    sync: Optional[SyncConfig] = None
    budget: Budget = Field(default_factory=Budget)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    tags: list[str] = Field(default_factory=list)
    priority: str = Field(default="normal")


class JobResponse(BaseModel):
    """Job response model."""

    job_id: str
    name: str
    project: Optional[str]
    status: str
    instance_id: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_cost_usd: float
    total_runtime_seconds: int


class JobListResponse(BaseModel):
    """Response containing list of jobs."""

    jobs: list[JobResponse]
    total: int


@jobs_router.post("", response_model=JobResponse)
async def submit_job(request: JobSubmitRequest) -> JobResponse:
    """Submit a new job."""
    import json

    # Create job record
    job = Job(
        name=request.name,
        project=request.project,
        status=JobStatus.PENDING,
        image=request.image,
        command=request.command,
        resources_json=request.resources.model_dump_json(),
        checkpoint_json=request.checkpoint.model_dump_json(),
        sync_json=request.sync.model_dump_json() if request.sync else "{}",
        budget_json=request.budget.model_dump_json(),
        retry_json=request.retry.model_dump_json(),
        tags=request.tags,
    )

    async with get_session() as session:
        session.add(job)
        await session.flush()
        await session.refresh(job)

    return JobResponse(
        job_id=job.job_id,
        name=job.name,
        project=job.project,
        status=job.status.value,
        instance_id=job.instance_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        total_cost_usd=job.total_cost_usd,
        total_runtime_seconds=job.total_runtime_seconds,
    )


@jobs_router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    project: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(20, ge=1, le=100, description="Maximum jobs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> JobListResponse:
    """List all jobs."""
    from sqlmodel import select, func

    async with get_session() as session:
        # Build query
        stmt = select(Job).order_by(Job.created_at.desc())

        if status:
            stmt = stmt.where(Job.status == JobStatus(status))
        if project:
            stmt = stmt.where(Job.project == project)

        # Get total count
        count_stmt = select(func.count()).select_from(Job)
        if status:
            count_stmt = count_stmt.where(Job.status == JobStatus(status))
        if project:
            count_stmt = count_stmt.where(Job.project == project)

        total = await session.scalar(count_stmt)

        # Apply pagination
        stmt = stmt.offset(offset).limit(limit)
        result = await session.execute(stmt)
        jobs = result.scalars().all()

    return JobListResponse(
        jobs=[
            JobResponse(
                job_id=j.job_id,
                name=j.name,
                project=j.project,
                status=j.status.value,
                instance_id=j.instance_id,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
                total_cost_usd=j.total_cost_usd,
                total_runtime_seconds=j.total_runtime_seconds,
            )
            for j in jobs
        ],
        total=total or 0,
    )


@jobs_router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Get job details."""
    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobResponse(
        job_id=job.job_id,
        name=job.name,
        project=job.project,
        status=job.status.value,
        instance_id=job.instance_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        total_cost_usd=job.total_cost_usd,
        total_runtime_seconds=job.total_runtime_seconds,
    )


@jobs_router.delete("/{job_id}")
async def cancel_job(job_id: str, force: bool = Query(False)) -> dict:
    """Cancel a job."""
    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        if job.status not in [JobStatus.RUNNING, JobStatus.PROVISIONING, JobStatus.RECOVERING]:
            raise HTTPException(
                status_code=400,
                detail=f"Job cannot be cancelled: {job.status.value}",
            )

        # Terminate instance
        if job.instance_id:
            provider = VastProvider()
            await provider.terminate_instance(job.instance_id)

        job.status = JobStatus.CANCELLED
        session.add(job)

    return {"status": "cancelled", "job_id": job_id}


@jobs_router.post("/{job_id}/checkpoint")
async def trigger_checkpoint(job_id: str) -> dict:
    """Trigger a checkpoint for a running job."""
    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if not job.instance_id:
        raise HTTPException(status_code=400, detail="Job has no instance")

    provider = VastProvider()
    orchestrator = CheckpointOrchestrator(provider)

    checkpoint = await orchestrator.save_checkpoint(
        job.job_id,
        job.instance_id,
        job.checkpoint_config,
    )

    if not checkpoint:
        raise HTTPException(status_code=500, detail="Failed to create checkpoint")

    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "location": checkpoint.location,
        "size_bytes": checkpoint.size_bytes,
        "verified": checkpoint.verified,
    }


@jobs_router.post("/{job_id}/sync")
async def trigger_sync(job_id: str) -> dict:
    """Trigger a sync for a running job."""
    from sqlmodel import select

    from cloudcomputemanager.sync import SyncEngine

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if not job.instance_id:
        raise HTTPException(status_code=400, detail="Job has no instance")

    sync_config = job.sync_config
    if not sync_config:
        raise HTTPException(status_code=400, detail="Job has no sync configuration")

    provider = VastProvider()
    sync_engine = SyncEngine(provider)

    record = await sync_engine.sync_once(job.job_id, job.instance_id, sync_config)

    return {
        "sync_id": record.sync_id,
        "status": record.status.value,
        "files_synced": record.files_synced,
        "bytes_synced": record.bytes_synced,
        "duration_seconds": record.duration_seconds,
    }


# ============================================================================
# Instances Routes
# ============================================================================

instances_router = APIRouter()


class InstanceResponse(BaseModel):
    """Instance response model."""

    instance_id: str
    provider: str
    status: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    hourly_rate: float
    ssh_host: str
    ssh_port: int


class InstanceListResponse(BaseModel):
    """Response containing list of instances."""

    instances: list[InstanceResponse]


@instances_router.get("", response_model=InstanceListResponse)
async def list_instances() -> InstanceListResponse:
    """List all instances."""
    provider = VastProvider()
    instances = await provider.list_instances()

    return InstanceListResponse(
        instances=[
            InstanceResponse(
                instance_id=i.instance_id,
                provider=i.provider,
                status=i.status.value,
                gpu_type=i.gpu_type,
                gpu_count=i.gpu_count,
                gpu_memory_gb=i.gpu_memory_gb,
                hourly_rate=i.hourly_rate,
                ssh_host=i.ssh_host,
                ssh_port=i.ssh_port,
            )
            for i in instances
        ]
    )


@instances_router.get("/{instance_id}", response_model=InstanceResponse)
async def get_instance(instance_id: str) -> InstanceResponse:
    """Get instance details."""
    provider = VastProvider()
    instance = await provider.get_instance(instance_id)

    if not instance:
        raise HTTPException(status_code=404, detail=f"Instance not found: {instance_id}")

    return InstanceResponse(
        instance_id=instance.instance_id,
        provider=instance.provider,
        status=instance.status.value,
        gpu_type=instance.gpu_type,
        gpu_count=instance.gpu_count,
        gpu_memory_gb=instance.gpu_memory_gb,
        hourly_rate=instance.hourly_rate,
        ssh_host=instance.ssh_host,
        ssh_port=instance.ssh_port,
    )


@instances_router.delete("/{instance_id}")
async def terminate_instance(instance_id: str) -> dict:
    """Terminate an instance."""
    provider = VastProvider()

    success = await provider.terminate_instance(instance_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to terminate instance")

    return {"status": "terminated", "instance_id": instance_id}


# ============================================================================
# Offers Routes
# ============================================================================

offers_router = APIRouter()


class OfferResponse(BaseModel):
    """Offer response model."""

    offer_id: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: int
    cpu_cores: float  # Vast.ai returns fractional cores
    memory_gb: float  # Also fractional
    disk_gb: int
    hourly_rate: float
    location: str
    reliability_score: float


class OfferListResponse(BaseModel):
    """Response containing list of offers."""

    offers: list[OfferResponse]


@offers_router.get("", response_model=OfferListResponse)
async def search_offers(
    gpu_type: Optional[str] = Query(None, description="GPU type"),
    gpu_count: int = Query(1, ge=1, description="Number of GPUs"),
    gpu_memory_min: int = Query(16, ge=4, description="Minimum GPU memory"),
    max_price: Optional[float] = Query(None, description="Maximum hourly price"),
    limit: int = Query(20, ge=1, le=100, description="Maximum offers"),
) -> OfferListResponse:
    """Search available GPU offers."""
    provider = VastProvider()

    offers = await provider.search_offers(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        gpu_memory_min=gpu_memory_min,
        max_hourly_rate=max_price,
    )

    return OfferListResponse(
        offers=[
            OfferResponse(
                offer_id=o.offer_id,
                gpu_type=o.gpu_type,
                gpu_count=o.gpu_count,
                gpu_memory_gb=o.gpu_memory_gb,
                cpu_cores=o.cpu_cores,
                memory_gb=o.memory_gb,
                disk_gb=o.disk_gb,
                hourly_rate=o.hourly_rate,
                location=o.location,
                reliability_score=o.reliability_score,
            )
            for o in offers[:limit]
        ]
    )


# ============================================================================
# Checkpoints Routes
# ============================================================================

checkpoints_router = APIRouter()


class CheckpointResponse(BaseModel):
    """Checkpoint response model."""

    checkpoint_id: str
    job_id: str
    strategy: str
    trigger: str
    location: str
    size_bytes: int
    created_at: datetime
    verified: bool


class CheckpointListResponse(BaseModel):
    """Response containing list of checkpoints."""

    checkpoints: list[CheckpointResponse]


@checkpoints_router.get("", response_model=CheckpointListResponse)
async def list_checkpoints(
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum checkpoints"),
) -> CheckpointListResponse:
    """List checkpoints."""
    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Checkpoint).order_by(Checkpoint.created_at.desc())

        if job_id:
            stmt = stmt.where(Checkpoint.job_id == job_id)

        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        checkpoints = result.scalars().all()

    return CheckpointListResponse(
        checkpoints=[
            CheckpointResponse(
                checkpoint_id=c.checkpoint_id,
                job_id=c.job_id,
                strategy=c.strategy.value,
                trigger=c.trigger.value,
                location=c.location,
                size_bytes=c.size_bytes,
                created_at=c.created_at,
                verified=c.verified,
            )
            for c in checkpoints
        ]
    )


@checkpoints_router.get("/{checkpoint_id}", response_model=CheckpointResponse)
async def get_checkpoint(checkpoint_id: str) -> CheckpointResponse:
    """Get checkpoint details."""
    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Checkpoint).where(Checkpoint.checkpoint_id == checkpoint_id)
        result = await session.execute(stmt)
        checkpoint = result.scalar_one_or_none()

    if not checkpoint:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_id}")

    return CheckpointResponse(
        checkpoint_id=checkpoint.checkpoint_id,
        job_id=checkpoint.job_id,
        strategy=checkpoint.strategy.value,
        trigger=checkpoint.trigger.value,
        location=checkpoint.location,
        size_bytes=checkpoint.size_bytes,
        created_at=checkpoint.created_at,
        verified=checkpoint.verified,
    )


# ============================================================================
# PackStore Routes
# ============================================================================

packages_router = APIRouter()


class PackageVariantResponse(BaseModel):
    """Package variant response model."""

    id: str
    version: str
    description: str
    cuda_versions: list[str]
    gpu_architectures: list[str]


class PackageResponse(BaseModel):
    """Package response model."""

    name: str
    display_name: str
    category: str
    description: str
    homepage: Optional[str]
    license: Optional[str]
    variants: list[PackageVariantResponse]


class PackageListResponse(BaseModel):
    """Response containing list of packages."""

    packages: list[PackageResponse]
    total: int


class DeploymentRequest(BaseModel):
    """Request to deploy packages to an instance."""

    packages: list[str] = Field(..., description="Package names to deploy")
    strategy: str = Field(default="auto", description="Deployment strategy")
    verify: bool = Field(default=True, description="Verify after deployment")


class PackageDeploymentResponse(BaseModel):
    """Package deployment result."""

    package_name: str
    variant_id: str
    status: str
    strategy_used: str
    verified: bool
    duration_seconds: int
    error_message: Optional[str] = None


class DeploymentResponse(BaseModel):
    """Deployment response model."""

    instance_id: str
    success: bool
    total_duration_seconds: int
    environment: dict
    deployments: list[PackageDeploymentResponse]


@packages_router.get("", response_model=PackageListResponse)
async def list_packages(
    category: Optional[str] = Query(None, description="Filter by category"),
) -> PackageListResponse:
    """List available packages."""
    from cloudcomputemanager.packstore import PackageRegistry
    from cloudcomputemanager.packstore.registry import PackageCategory

    registry = PackageRegistry()

    cat_filter = None
    if category:
        try:
            cat_filter = PackageCategory(category)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown category: {category}. Valid: {[c.value for c in PackageCategory]}",
            )

    packages = registry.list_packages(category=cat_filter)

    return PackageListResponse(
        packages=[
            PackageResponse(
                name=p.name,
                display_name=p.display_name,
                category=p.category.value,
                description=p.description,
                homepage=p.homepage,
                license=p.license,
                variants=[
                    PackageVariantResponse(
                        id=v.id,
                        version=v.version,
                        description=v.description,
                        cuda_versions=v.compatibility.cuda_versions,
                        gpu_architectures=v.compatibility.gpu_architectures,
                    )
                    for v in p.variants
                ],
            )
            for p in packages
        ],
        total=len(packages),
    )


@packages_router.get("/search")
async def search_packages(
    query: str = Query(..., description="Search query"),
    cuda_version: Optional[str] = Query(None, description="CUDA version filter"),
    gpu_arch: Optional[str] = Query(None, description="GPU architecture filter"),
) -> PackageListResponse:
    """Search for packages."""
    from cloudcomputemanager.packstore import PackageRegistry

    registry = PackageRegistry()
    results = registry.search(query, cuda_version=cuda_version, gpu_arch=gpu_arch)

    return PackageListResponse(
        packages=[
            PackageResponse(
                name=p.name,
                display_name=p.display_name,
                category=p.category.value,
                description=p.description,
                homepage=p.homepage,
                license=p.license,
                variants=[
                    PackageVariantResponse(
                        id=v.id,
                        version=v.version,
                        description=v.description,
                        cuda_versions=v.compatibility.cuda_versions,
                        gpu_architectures=v.compatibility.gpu_architectures,
                    )
                    for v in p.variants
                ],
            )
            for p in results
        ],
        total=len(results),
    )


@packages_router.get("/{package_name}", response_model=PackageResponse)
async def get_package(package_name: str) -> PackageResponse:
    """Get package details."""
    from cloudcomputemanager.packstore import PackageRegistry

    registry = PackageRegistry()
    package = registry.get(package_name)

    if not package:
        raise HTTPException(status_code=404, detail=f"Package not found: {package_name}")

    return PackageResponse(
        name=package.name,
        display_name=package.display_name,
        category=package.category.value,
        description=package.description,
        homepage=package.homepage,
        license=package.license,
        variants=[
            PackageVariantResponse(
                id=v.id,
                version=v.version,
                description=v.description,
                cuda_versions=v.compatibility.cuda_versions,
                gpu_architectures=v.compatibility.gpu_architectures,
            )
            for v in package.variants
        ],
    )


@packages_router.post("/deploy/{instance_id}", response_model=DeploymentResponse)
async def deploy_packages(
    instance_id: str,
    request: DeploymentRequest,
) -> DeploymentResponse:
    """Deploy packages to an instance."""
    from cloudcomputemanager.packstore import PackageDeployer, DeploymentStrategy

    provider = VastProvider()
    deployer = PackageDeployer(provider)

    try:
        strategy = DeploymentStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}. Valid: {[s.value for s in DeploymentStrategy]}",
        )

    result = await deployer.deploy(
        instance_id=instance_id,
        packages=request.packages,
        strategy=strategy,
        verify=request.verify,
    )

    return DeploymentResponse(
        instance_id=instance_id,
        success=result.success,
        total_duration_seconds=result.total_duration_seconds,
        environment={
            "gpu_name": result.environment.gpu_name,
            "gpu_count": result.environment.gpu_count,
            "gpu_arch": result.environment.gpu_arch,
            "cuda_version": result.environment.cuda_version,
            "driver_version": result.environment.driver_version,
        },
        deployments=[
            PackageDeploymentResponse(
                package_name=d.package_name,
                variant_id=d.variant_id,
                status=d.status.value,
                strategy_used=d.strategy_used.value,
                verified=d.verified,
                duration_seconds=d.duration_seconds,
                error_message=d.error_message,
            )
            for d in result.deployments
        ],
    )


@packages_router.post("/verify/{instance_id}/{package_name}")
async def verify_package(
    instance_id: str,
    package_name: str,
) -> dict:
    """Verify a package installation on an instance."""
    from cloudcomputemanager.packstore import PackageRegistry

    registry = PackageRegistry()
    provider = VastProvider()

    package = registry.get(package_name)
    if not package:
        raise HTTPException(status_code=404, detail=f"Package not found: {package_name}")

    results = []
    for variant in package.variants:
        if not variant.test_command:
            continue

        # Set up environment
        env_setup = ""
        for key, value in variant.environment.items():
            env_setup += f"export {key}={value}; "

        cmd = f"{env_setup}{variant.test_command}"
        exit_code, stdout, stderr = await provider.execute_command(instance_id, cmd)

        passed = exit_code == 0
        if passed and variant.expected_output_contains:
            passed = variant.expected_output_contains in stdout

        results.append({
            "variant_id": variant.id,
            "passed": passed,
            "exit_code": exit_code,
            "stdout": stdout[:500] if stdout else None,
            "stderr": stderr[:500] if stderr else None,
        })

    return {
        "package": package_name,
        "instance_id": instance_id,
        "results": results,
        "all_passed": all(r["passed"] for r in results) if results else False,
    }
