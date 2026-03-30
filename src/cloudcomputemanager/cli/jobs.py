"""Job management CLI commands."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.live import Live

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.database import init_db, get_session
from cloudcomputemanager.core.models import Job, JobStatus
from cloudcomputemanager.providers.vast import VastProvider

console = Console()


async def check_job_completion(provider: VastProvider, instance_id: str) -> Tuple[bool, Optional[int]]:
    """
    Check if a job has completed by looking for the .ccm_exit_code file.

    Returns:
        (completed, exit_code) - completed is True if job finished, exit_code is the job's exit code
    """
    try:
        exit_code, stdout, stderr = await provider.execute_command(
            instance_id,
            "cat /workspace/.ccm_exit_code 2>/dev/null || echo 'running'"
        )

        if exit_code == 0 and stdout.strip() != "running":
            try:
                job_exit_code = int(stdout.strip())
                return True, job_exit_code
            except ValueError:
                return False, None
        return False, None
    except Exception:
        return False, None


async def wait_for_job_completion(
    provider: VastProvider,
    job: Job,
    instance_id: str,
    timeout: int = 86400,
    poll_interval: int = 30,
) -> Tuple[bool, Optional[int]]:
    """
    Wait for a job to complete, polling periodically.

    Args:
        provider: VastProvider instance
        job: Job record
        instance_id: Instance ID running the job
        timeout: Maximum time to wait in seconds (default: 24 hours)
        poll_interval: Time between status checks in seconds (default: 30)

    Returns:
        (completed, exit_code) - completed is True if job finished within timeout
    """
    start_time = asyncio.get_event_loop().time()
    last_log_lines = ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Waiting for job...", total=None)

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout:
                progress.update(task, description="[red]Timeout reached")
                return False, None

            # Check completion
            completed, exit_code = await check_job_completion(provider, instance_id)
            if completed:
                return True, exit_code

            # Get last few lines of log for progress display
            try:
                _, stdout, _ = await provider.execute_command(
                    instance_id,
                    "tail -1 /workspace/job.log 2>/dev/null || echo 'Starting...'"
                )
                log_line = stdout.strip()[:60]
                if log_line != last_log_lines:
                    last_log_lines = log_line
                    progress.update(task, description=f"Running: {log_line}")
            except Exception:
                pass

            # Update progress
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            progress.update(task, description=f"Running ({hours}h {minutes}m)... {last_log_lines[:40]}")

            await asyncio.sleep(poll_interval)


async def submit_job(
    config_file: Path,
    name_override: Optional[str],
    wait: bool,
    template_name: Optional[str] = None,
    quiet: bool = False,
    variables: Optional[dict[str, str]] = None,
) -> None:
    """Submit a job from a YAML configuration file.

    Args:
        config_file: Path to job configuration YAML
        name_override: Optional name to override config
        wait: Whether to wait for job completion
        template_name: Optional template to use as base config
        quiet: If True, suppress Rich displays (for batch mode)
        variables: Optional dict of variable substitutions for ${VAR} in YAML
    """
    from cloudcomputemanager.core.templates import load_config_with_template, validate_job_config
    import structlog
    logger = structlog.get_logger(__name__)

    # Load configuration (with optional template merging and variable substitution)
    # Note: load_config_with_template now normalizes resource keys
    config = load_config_with_template(config_file, template_name, variables=variables)

    # Validate configuration
    errors = validate_job_config(config)
    if errors and not config.get("command"):
        # Allow missing command if we just want to inspect config
        pass

    job_name = name_override or config.get("name", config_file.stem)

    if not quiet:
        console.print(f"\n[bold]Submitting job:[/bold] {job_name}")
    else:
        logger.info("Submitting job", job_name=job_name)

    # Parse configuration
    image = config.get("image", "vastai/pytorch:latest")
    command = config.get("command", "")
    setup_commands = config.get("setup")  # Setup script before job
    resources = config.get("resources", {})
    checkpoint_config = config.get("checkpoint", {})
    sync_config = config.get("sync", {})
    budget = config.get("budget", {})
    upload_config = config.get("upload", {})
    retry_config = config.get("retry", {})
    stages = config.get("stages", [])
    progress_config = config.get("progress", {})
    notifications = config.get("notifications", {})
    provisioning_timeout = config.get("provisioning", {}).get("timeout", 300)

    # Parse environment configuration (if specified)
    from cloudcomputemanager.core.environment import (
        parse_environment, validate_environment, get_setup_commands,
        get_command_prefix, get_upload_files, get_recommended_timeout,
    )

    env_config = parse_environment(config)
    env_upload_files = []
    env_post_upload_setup = ""  # Setup commands that run AFTER file upload (via SSH)

    if env_config:
        # Validate environment
        env_errors = validate_environment(env_config)
        if env_errors:
            for err in env_errors:
                if not quiet:
                    console.print(f"[red]Environment error: {err}[/red]")
                else:
                    logger.error("Environment error", error=err)
            return

        # Override image if docker_image strategy
        if env_config.docker_image:
            image = env_config.docker_image

        # Environment setup commands run AFTER file upload (not in onstart).
        # This is because strategies like conda_pack need files uploaded first.
        env_post_upload_setup = get_setup_commands(env_config)

        # User setup commands still go in onstart (they don't depend on uploaded files)
        # Do NOT merge env setup into setup_commands

        # Prepend activation to job command
        prefix = get_command_prefix(env_config)
        if prefix and command:
            command = prefix + command

        # Collect environment files to upload
        env_upload_files = get_upload_files(env_config)

        # Use recommended timeout if user did not specify one
        user_timeout = config.get("provisioning", {}).get("timeout")
        if not user_timeout:
            recommended = get_recommended_timeout(env_config)
            if recommended > provisioning_timeout:
                provisioning_timeout = recommended

        if not quiet:
            console.print(f"  Environment: {env_config.strategy.value}")
        else:
            logger.info("Environment configured", strategy=env_config.strategy.value)

    # Initialize database
    await init_db()

    # Search for offers
    provider = VastProvider()

    # Use user-specified values, falling back to provider defaults only if not specified
    # Note: resources keys are already normalized by load_config_with_template
    search_params = {
        "gpu_type": resources.get("gpu_type"),
        "gpu_count": resources.get("gpu_count", 1),
        "gpu_memory_min": resources.get("gpu_memory_min"),
        "disk_gb_min": resources.get("disk_gb", 50),
        "max_hourly_rate": budget.get("max_hourly_rate"),
        "cpu_cores_min": resources.get("cpu_cores"),
        "cuda_version_min": resources.get("cuda_version_min"),
        "reliability_min": resources.get("reliability_min"),
        "min_duration_hours": resources.get("min_duration_hours"),
    }

    if not quiet:
        with console.status("Searching for GPU offers..."):
            offers = await provider.search_offers(**search_params)
    else:
        logger.info("Searching for GPU offers", **{k: v for k, v in search_params.items() if v is not None})
        offers = await provider.search_offers(**search_params)

    if not offers:
        error_msg = f"No suitable offers found for: gpu_type={search_params['gpu_type']}, gpu_memory_min={search_params['gpu_memory_min']}, max_rate={search_params['max_hourly_rate']}"
        if not quiet:
            console.print(f"[red]{error_msg}[/red]")
        else:
            logger.error(error_msg)
        return

    # Show best offer
    best = offers[0]
    if not quiet:
        console.print(f"\n[green]Found {len(offers)} offers. Best:[/green]")
        console.print(f"  GPU: {best.gpu_type} x{best.gpu_count}")
        console.print(f"  Price: ${best.hourly_rate:.3f}/hr")
        console.print(f"  Location: {best.location}")
    else:
        logger.info("Found offers", count=len(offers), best_gpu=best.gpu_type, best_price=best.hourly_rate)

    # Generate job_id early so we can label the instance with it
    from cloudcomputemanager.core.models import generate_id
    from cloudcomputemanager.core.instances import build_instance_label
    early_job_id = f"job_{generate_id()}"
    instance_label = build_instance_label(early_job_id, config.get("project", ""), job_name)

    # Create instance (pass setup commands as startup_script if provided)
    create_kwargs = {
        "offer_id": best.offer_id,
        "image": image,
        "disk_gb": resources.get("disk_gb", 50),
        "label": instance_label,
    }
    if setup_commands:
        create_kwargs["startup_script"] = setup_commands

    if not quiet:
        with console.status("Creating instance..."):
            instance = await provider.create_instance(**create_kwargs)
    else:
        logger.info("Creating instance", offer_id=best.offer_id, image=image, has_setup=bool(setup_commands))
        instance = await provider.create_instance(**create_kwargs)

    # Save Instance record to database
    from cloudcomputemanager.core.instances import upsert_instance
    await upsert_instance(instance, job_id=early_job_id)

    # Create Job record IMMEDIATELY so it's tracked even if later steps fail.
    # Status starts as PROVISIONING — will be updated to RUNNING after setup completes.
    import json
    effective_command = command
    if stages:
        effective_command = stages[0].get("command", command)

    job = Job(
        job_id=early_job_id,
        name=job_name,
        project=config.get("project"),
        status=JobStatus.PROVISIONING,
        image=image,
        command=effective_command,
        setup_commands=setup_commands,
        provisioning_timeout=provisioning_timeout,
        stages_json=json.dumps(stages),
        current_stage=0,
        progress_json=json.dumps(progress_config),
        notifications_json=json.dumps(notifications),
        resources_json=json.dumps(resources),
        checkpoint_json=json.dumps(checkpoint_config),
        sync_json=json.dumps(sync_config),
        budget_json=json.dumps(budget),
        retry_json=json.dumps(retry_config),
        upload_json=json.dumps(upload_config),
        instance_id=instance.instance_id,
        started_at=datetime.utcnow(),
    )
    async with get_session() as session:
        session.add(job)

    if not quiet:
        console.print(f"\n[green]Instance created:[/green] {instance.instance_id}")
        console.print(f"  Job ID: {early_job_id}")
        console.print(f"  SSH: ssh -p {instance.ssh_port} {instance.ssh_user}@{instance.ssh_host}")
    else:
        logger.info("Instance created", instance_id=instance.instance_id, job_id=early_job_id)

    # Wait for instance to be ready (using configurable provisioning timeout)
    if not quiet:
        with console.status(f"Waiting for instance to be ready (timeout: {provisioning_timeout}s)..."):
            ready = await provider.wait_for_ready(instance.instance_id, timeout=provisioning_timeout)
    else:
        logger.info("Waiting for instance", instance_id=instance.instance_id, timeout=provisioning_timeout)
        ready = await provider.wait_for_ready(instance.instance_id, timeout=provisioning_timeout)

    if not ready:
        error_msg = f"Instance {instance.instance_id} failed to start within {provisioning_timeout}s"
        if not quiet:
            console.print(f"[red]{error_msg}![/red]")
            console.print("[yellow]Destroying instance to prevent charges...[/yellow]")
        else:
            logger.error(error_msg)
        # Mark job as FAILED
        async with get_session() as session:
            from sqlmodel import select as sel
            stmt = sel(Job).where(Job.job_id == early_job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                db_job.status = JobStatus.FAILED
                db_job.error_message = error_msg
                db_job.completed_at = datetime.utcnow()
                session.add(db_job)
        try:
            await provider.terminate_instance(instance.instance_id)
            if not quiet:
                console.print("[green]Instance destroyed successfully.[/green]")
            else:
                logger.info("Instance destroyed", instance_id=instance.instance_id)
        except Exception as e:
            if not quiet:
                console.print(f"[red]Failed to destroy instance: {e}[/red]")
                console.print(f"[yellow]Manually destroy with: vastai destroy instance {instance.instance_id}[/yellow]")
            else:
                logger.error("Failed to destroy instance", instance_id=instance.instance_id, error=str(e))
        return

    if not quiet:
        console.print("[green]Instance is ready![/green]")
    else:
        logger.info("Instance ready", instance_id=instance.instance_id)

    # Upload files if configured
    if upload_config.get("source"):
        source_path = Path(upload_config["source"]).expanduser()
        dest_path = upload_config.get("destination", "/workspace")

        if not source_path.exists():
            error_msg = f"Upload source not found: {source_path}"
            if not quiet:
                console.print(f"[red]{error_msg}[/red]")
                console.print("[yellow]Destroying instance...[/yellow]")
            else:
                logger.error(error_msg)
            await provider.terminate_instance(instance.instance_id)
            return

        if not quiet:
            console.print(f"\n[bold]Uploading files:[/bold] {source_path} -> {dest_path}")
            with console.status("Uploading files..."):
                upload_success = await provider.rsync_upload(
                    instance.instance_id,
                    str(source_path) + "/",
                    dest_path + "/",
                    exclude=upload_config.get("exclude", []),
                )
        else:
            logger.info("Uploading files", source=str(source_path), dest=dest_path)
            upload_success = await provider.rsync_upload(
                instance.instance_id,
                str(source_path) + "/",
                dest_path + "/",
                exclude=upload_config.get("exclude", []),
            )

        if not upload_success:
            error_msg = "File upload failed"
            if not quiet:
                console.print(f"[red]{error_msg}![/red]")
                console.print("[yellow]Destroying instance...[/yellow]")
            else:
                logger.error(error_msg)
            await provider.terminate_instance(instance.instance_id)
            return

        if not quiet:
            console.print("[green]Files uploaded successfully![/green]")
        else:
            logger.info("Files uploaded successfully")

    # Upload environment files (conda-pack, env.yml, requirements.txt)
    if env_upload_files:
        for local_file, remote_file in env_upload_files:
            local_path = Path(local_file)
            if not local_path.exists():
                error_msg = f"Environment file not found: {local_path}"
                if not quiet:
                    console.print(f"[red]{error_msg}[/red]")
                else:
                    logger.error(error_msg)
                await provider.terminate_instance(instance.instance_id)
                return

            if not quiet:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                console.print(f"[bold]Uploading environment:[/bold] {local_path.name} ({size_mb:.1f} MB)")
                with console.status("Uploading environment file..."):
                    upload_ok = await provider.rsync_upload(
                        instance.instance_id,
                        str(local_path),
                        remote_file,
                    )
            else:
                logger.info("Uploading environment file", file=local_file, dest=remote_file)
                upload_ok = await provider.rsync_upload(
                    instance.instance_id,
                    str(local_path),
                    remote_file,
                )

            if not upload_ok:
                error_msg = f"Failed to upload environment file: {local_path.name}"
                if not quiet:
                    console.print(f"[red]{error_msg}[/red]")
                else:
                    logger.error(error_msg)
                await provider.terminate_instance(instance.instance_id)
                return

        if not quiet:
            console.print("[green]Environment files uploaded.[/green]")

    # Run environment setup commands via SSH (after file upload)
    # Write commands to a script file, then execute it. This is more robust
    # than sending a long && chain via SSH (which can break on special chars).
    if env_post_upload_setup:
        import base64

        # Write setup script to instance
        # Don't use set -e: conda config commands may fail on some versions
        # and we handle the overall exit code at the CCM level.
        # The last command (conda env create or echo success) determines the exit code.
        script_content = "#!/bin/bash\n" + env_post_upload_setup
        b64 = base64.b64encode(script_content.encode()).decode()
        write_cmd = f"echo {b64} | base64 -d > /workspace/.ccm_env_setup.sh && chmod +x /workspace/.ccm_env_setup.sh"

        if not quiet:
            console.print("[bold]Setting up environment...[/bold]")

        # Deploy the script
        exit_code, _, stderr = await provider.execute_command(
            instance.instance_id, write_cmd, timeout=30,
        )
        if exit_code != 0:
            error_msg = f"Failed to deploy environment setup script: {stderr[:200]}"
            if not quiet:
                console.print(f"[red]{error_msg}[/red]")
            else:
                logger.error(error_msg)
            await provider.terminate_instance(instance.instance_id)
            return

        # Execute the script with long timeout (conda env create can take 15+ min)
        if not quiet:
            with console.status("Running environment setup (this may take 10-15 minutes)..."):
                exit_code, stdout, stderr = await provider.execute_command(
                    instance.instance_id,
                    "bash /workspace/.ccm_env_setup.sh",
                    timeout=1200,
                )
        else:
            logger.info("Running environment setup script")
            exit_code, stdout, stderr = await provider.execute_command(
                instance.instance_id,
                "bash /workspace/.ccm_env_setup.sh",
                timeout=1200,
            )

        if exit_code != 0:
            error_msg = f"Environment setup failed (exit {exit_code}): {stderr[:300]}"
            if not quiet:
                console.print(f"[red]{error_msg}[/red]")
            else:
                logger.error(error_msg)
            await provider.terminate_instance(instance.instance_id)
            return

        if not quiet:
            console.print("[green]Environment ready![/green]")
        else:
            logger.info("Environment setup complete")

    # Update job status from PROVISIONING to RUNNING now that setup is complete
    from sqlmodel import select as sel
    async with get_session() as session:
        stmt = sel(Job).where(Job.job_id == early_job_id)
        result = await session.execute(stmt)
        db_job = result.scalar_one_or_none()
        if db_job:
            db_job.status = JobStatus.RUNNING
            session.add(db_job)

    if not quiet:
        console.print(f"\n[bold green]Job ready to start![/bold green]")
        console.print(f"  Job ID: {early_job_id}")
        console.print(f"  Instance: {instance.instance_id}")
    else:
        logger.info("Job ready", job_id=early_job_id, instance_id=instance.instance_id)

    # Start the job
    if effective_command:
        if not quiet:
            console.print("\n[bold]Starting job...[/bold]")
        else:
            logger.info("Starting job", job_id=job.job_id)

        # Deploy and run SIGTERM-aware wrapper script
        from cloudcomputemanager.core.wrapper import build_deploy_commands
        setup_cmd, run_cmd = build_deploy_commands(effective_command)

        exit_code, stdout, stderr = await provider.execute_command(
            instance.instance_id, setup_cmd
        )

        if exit_code != 0:
            error_msg = f"Failed to create job script: {stderr}"
            if not quiet:
                console.print(f"[red]{error_msg}[/red]")
            else:
                logger.error(error_msg)
        else:
            # nohup ... & should return immediately; use short timeout
            exit_code, stdout, stderr = await provider.execute_command(
                instance.instance_id, run_cmd, timeout=15
            )

            if exit_code == 0:
                if not quiet:
                    console.print("[green]Job started in background[/green]")
                else:
                    logger.info("Job started in background", job_id=job.job_id)
            elif exit_code == -1 and "timed out" in stderr.lower():
                # nohup timeout is common on slow SSH connections but the job
                # usually starts successfully. Verify by checking for the process.
                # Fixes: https://github.com/fl-sean03/cloudcomputemanager/issues/7
                verify_code, verify_out, _ = await provider.execute_command(
                    instance.instance_id,
                    "pgrep -f run_job.sh > /dev/null 2>&1 && echo RUNNING || echo NOT_RUNNING",
                    timeout=15,
                )
                if "RUNNING" in verify_out:
                    if not quiet:
                        console.print("[green]Job started in background[/green] [dim](nohup SSH timed out but process verified)[/dim]")
                    else:
                        logger.info("Job started (verified after nohup timeout)", job_id=job.job_id)
                else:
                    error_msg = f"Failed to start job: nohup timed out and process not found"
                    if not quiet:
                        console.print(f"[red]{error_msg}[/red]")
                    else:
                        logger.error(error_msg)
            else:
                error_msg = f"Failed to start job: {stderr}"
                if not quiet:
                    console.print(f"[red]{error_msg}[/red]")
                else:
                    logger.error(error_msg)

    # Show sync info
    if sync_config.get("enabled", True):
        settings = get_settings()
        sync_dir = settings.sync_local_path / job.job_id
        if not quiet:
            console.print(f"\n[bold]Sync directory:[/bold] {sync_dir}")
        else:
            logger.info("Sync directory", path=str(sync_dir))

    # Auto-terminate setting from config (default: True)
    auto_terminate = config.get("auto_terminate", True)

    if wait:
        if not quiet:
            console.print("\n[dim]Waiting for job completion... (Ctrl+C to detach)[/dim]")
        else:
            logger.info("Waiting for job completion", job_id=job.job_id)
        try:
            completed, exit_code = await wait_for_job_completion(
                provider, job, instance.instance_id, timeout=budget.get("max_hours", 24) * 3600
            )

            if completed:
                # Update job status
                async with get_session() as session:
                    from sqlmodel import select
                    stmt = select(Job).where(Job.job_id == job.job_id)
                    result = await session.execute(stmt)
                    db_job = result.scalar_one_or_none()
                    if db_job:
                        db_job.exit_code = exit_code
                        db_job.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                        db_job.completed_at = datetime.utcnow()
                        session.add(db_job)

                if exit_code == 0:
                    if not quiet:
                        console.print("[bold green]Job completed successfully![/bold green]")
                    else:
                        logger.info("Job completed successfully", job_id=job.job_id)
                else:
                    if not quiet:
                        console.print(f"[bold red]Job failed with exit code {exit_code}[/bold red]")
                    else:
                        logger.error("Job failed", job_id=job.job_id, exit_code=exit_code)

                # Final sync
                if not quiet:
                    console.print("\n[bold]Syncing final results...[/bold]")
                else:
                    logger.info("Syncing final results", job_id=job.job_id)
                sync_dir = settings.sync_local_path / job.job_id
                sync_dir.mkdir(parents=True, exist_ok=True)
                await provider.rsync_download(
                    instance.instance_id,
                    sync_config.get("source", "/workspace") + "/",
                    str(sync_dir) + "/",
                    exclude=sync_config.get("exclude_patterns", []),
                )
                if not quiet:
                    console.print(f"[green]Results synced to: {sync_dir}[/green]")
                else:
                    logger.info("Results synced", path=str(sync_dir))

                # Auto-terminate if enabled
                if auto_terminate:
                    if not quiet:
                        console.print("\n[bold]Terminating instance...[/bold]")
                    await provider.terminate_instance(instance.instance_id)
                    if not quiet:
                        console.print("[green]Instance terminated.[/green]")
                    else:
                        logger.info("Instance terminated", instance_id=instance.instance_id)
        except KeyboardInterrupt:
            if not quiet:
                console.print("\n[yellow]Detached from job. Job continues running.[/yellow]")
                console.print(f"Check status with: ccm jobs status {job.job_id}")
            else:
                logger.info("Detached from job", job_id=job.job_id)


async def list_jobs(
    status: Optional[str],
    project: Optional[str],
    limit: int,
) -> None:
    """List all jobs."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).order_by(Job.created_at.desc()).limit(limit)

        if status:
            stmt = stmt.where(Job.status == JobStatus(status.upper()))
        if project:
            stmt = stmt.where(Job.project == project)

        result = await session.execute(stmt)
        jobs = result.scalars().all()

    if not jobs:
        console.print("[yellow]No jobs found.[/yellow]")
        return

    table = Table(title="Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Instance", style="dim")
    table.add_column("Created", style="dim")

    status_colors = {
        JobStatus.PENDING: "yellow",
        JobStatus.PROVISIONING: "yellow",
        JobStatus.RUNNING: "green",
        JobStatus.CHECKPOINTING: "blue",
        JobStatus.RECOVERING: "yellow",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.CANCELLED: "dim",
    }

    for job in jobs:
        color = status_colors.get(job.status, "white")
        table.add_row(
            job.job_id,
            job.name,
            f"[{color}]{job.status.value}[/{color}]",
            job.instance_id or "-",
            job.created_at.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)


async def show_job_status(job_id: str, watch: bool) -> None:
    """Show detailed job status."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        return

    # Build status panel
    status_color = {
        JobStatus.RUNNING: "green",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
    }.get(job.status, "yellow")

    panel_content = f"""
[bold]Name:[/bold] {job.name}
[bold]Status:[/bold] [{status_color}]{job.status.value}[/{status_color}]
[bold]Instance:[/bold] {job.instance_id or 'None'}
[bold]Attempt:[/bold] {job.attempt_number + 1}
[bold]Created:[/bold] {job.created_at.strftime('%Y-%m-%d %H:%M:%S')}
[bold]Started:[/bold] {job.started_at.strftime('%Y-%m-%d %H:%M:%S') if job.started_at else 'Not started'}
[bold]Cost:[/bold] ${job.total_cost_usd:.2f}
"""

    if job.error_message:
        panel_content += f"\n[bold red]Error:[/bold red] {job.error_message}"

    console.print(Panel(panel_content, title=f"Job: {job.job_id}"))

    # Show instance details if running
    if job.instance_id and job.status == JobStatus.RUNNING:
        provider = VastProvider()
        instance = await provider.get_instance(job.instance_id)

        if instance:
            console.print(f"\n[bold]Instance Details:[/bold]")
            console.print(f"  SSH: ssh -p {instance.ssh_port} {instance.ssh_user}@{instance.ssh_host}")
            console.print(f"  GPU: {instance.gpu_type} x{instance.gpu_count}")
            console.print(f"  Rate: ${instance.hourly_rate:.3f}/hr")


async def show_job_logs(job_id: str, tail: int, follow: bool) -> None:
    """Show job logs."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job or not job.instance_id:
        console.print(f"[red]Job not found or no instance: {job_id}[/red]")
        return

    provider = VastProvider()

    cmd = f"tail -n {tail} /workspace/job.log 2>/dev/null || cat /workspace/log.lammps 2>/dev/null || echo 'No logs found'"
    exit_code, stdout, stderr = await provider.execute_command(job.instance_id, cmd)

    if stdout:
        console.print(Panel(stdout, title=f"Logs: {job_id}"))
    else:
        console.print("[yellow]No logs available.[/yellow]")


async def cancel_job(job_id: str, force: bool) -> None:
    """Cancel a running job."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            console.print(f"[red]Job not found: {job_id}[/red]")
            return

        if job.status not in [JobStatus.RUNNING, JobStatus.PROVISIONING, JobStatus.RECOVERING]:
            console.print(f"[yellow]Job is not running: {job.status.value}[/yellow]")
            return

        # Terminate instance if exists
        if job.instance_id:
            provider = VastProvider()
            with console.status("Terminating instance..."):
                await provider.terminate_instance(job.instance_id)

        # Update job status
        job.status = JobStatus.CANCELLED
        session.add(job)

    console.print(f"[green]Job cancelled: {job_id}[/green]")


async def sync_job(job_id: str) -> None:
    """Trigger immediate sync for a job."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job or not job.instance_id:
        console.print(f"[red]Job not found or no instance: {job_id}[/red]")
        return

    provider = VastProvider()
    sync_config = job.sync_config

    if not sync_config:
        console.print("[yellow]No sync configuration for this job.[/yellow]")
        return

    settings = get_settings()
    local_dest = settings.sync_local_path / job_id
    local_dest.mkdir(parents=True, exist_ok=True)

    with console.status("Syncing..."):
        success = await provider.rsync_download(
            job.instance_id,
            sync_config.source + "/",
            str(local_dest) + "/",
            exclude=sync_config.exclude_patterns,
        )

    if success:
        console.print(f"[green]Sync completed![/green]")
        console.print(f"  Local path: {local_dest}")
    else:
        console.print("[red]Sync failed![/red]")


async def checkpoint_job(job_id: str) -> None:
    """Trigger immediate checkpoint for a job."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job or not job.instance_id:
        console.print(f"[red]Job not found or no instance: {job_id}[/red]")
        return

    from cloudcomputemanager.checkpoint import CheckpointOrchestrator

    provider = VastProvider()
    orchestrator = CheckpointOrchestrator(provider)

    with console.status("Creating checkpoint..."):
        checkpoint = await orchestrator.save_checkpoint(
            job.job_id,
            job.instance_id,
            job.checkpoint_config,
        )

    if checkpoint:
        console.print(f"[green]Checkpoint created![/green]")
        console.print(f"  ID: {checkpoint.checkpoint_id}")
        console.print(f"  Location: {checkpoint.location}")
        console.print(f"  Size: {checkpoint.size_bytes} bytes")
        console.print(f"  Verified: {checkpoint.verified}")
    else:
        console.print("[red]Failed to create checkpoint![/red]")


async def wait_for_existing_job(job_id: str, timeout: int, auto_terminate: bool) -> None:
    """Wait for an existing job to complete."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

    if not job:
        console.print(f"[red]Job not found: {job_id}[/red]")
        return

    if not job.instance_id:
        console.print(f"[red]Job has no instance: {job_id}[/red]")
        return

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        console.print(f"[yellow]Job already finished: {job.status.value}[/yellow]")
        return

    provider = VastProvider()
    settings = get_settings()

    console.print(f"\n[bold]Waiting for job:[/bold] {job_id}")
    console.print(f"  Instance: {job.instance_id}")
    console.print(f"  Timeout: {timeout}s")
    console.print(f"  Auto-terminate: {auto_terminate}\n")

    try:
        completed, exit_code = await wait_for_job_completion(
            provider, job, job.instance_id, timeout=timeout
        )

        if completed:
            # Update job status
            async with get_session() as session:
                stmt = select(Job).where(Job.job_id == job_id)
                result = await session.execute(stmt)
                db_job = result.scalar_one_or_none()
                if db_job:
                    db_job.exit_code = exit_code
                    db_job.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
                    db_job.completed_at = datetime.utcnow()
                    session.add(db_job)

            if exit_code == 0:
                console.print("[bold green]Job completed successfully![/bold green]")
            else:
                console.print(f"[bold red]Job failed with exit code {exit_code}[/bold red]")

            # Final sync
            sync_config = job.sync_config or {}
            if sync_config:
                console.print("\n[bold]Syncing final results...[/bold]")
                sync_dir = settings.sync_local_path / job_id
                sync_dir.mkdir(parents=True, exist_ok=True)
                await provider.rsync_download(
                    job.instance_id,
                    sync_config.get("source", "/workspace") + "/",
                    str(sync_dir) + "/",
                    exclude=sync_config.get("exclude_patterns", []),
                )
                console.print(f"[green]Results synced to: {sync_dir}[/green]")

            # Auto-terminate if enabled
            if auto_terminate:
                console.print("\n[bold]Terminating instance...[/bold]")
                await provider.terminate_instance(job.instance_id)
                console.print("[green]Instance terminated.[/green]")
        else:
            console.print("[yellow]Timeout reached. Job still running.[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Detached from job. Job continues running.[/yellow]")


async def complete_job(job_id: str, status: str, terminate: bool) -> None:
    """Mark a job as complete, sync results, and optionally terminate instance."""
    await init_db()

    from sqlmodel import select

    async with get_session() as session:
        stmt = select(Job).where(Job.job_id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()

        if not job:
            console.print(f"[red]Job not found: {job_id}[/red]")
            return

        # Update status
        if status == "completed":
            job.status = JobStatus.COMPLETED
        elif status == "failed":
            job.status = JobStatus.FAILED
        else:
            console.print(f"[red]Invalid status: {status}. Use 'completed' or 'failed'[/red]")
            return

        job.completed_at = datetime.utcnow()
        session.add(job)

    console.print(f"[green]Job {job_id} marked as {status}[/green]")

    provider = VastProvider()
    settings = get_settings()

    # Sync if instance exists
    if job.instance_id:
        sync_config = job.sync_config or {}
        if sync_config:
            console.print("\n[bold]Syncing results...[/bold]")
            sync_dir = settings.sync_local_path / job_id
            sync_dir.mkdir(parents=True, exist_ok=True)
            try:
                await provider.rsync_download(
                    job.instance_id,
                    sync_config.get("source", "/workspace") + "/",
                    str(sync_dir) + "/",
                    exclude=sync_config.get("exclude_patterns", []),
                )
                console.print(f"[green]Results synced to: {sync_dir}[/green]")
            except Exception as e:
                console.print(f"[yellow]Sync failed: {e}[/yellow]")

        # Terminate if requested
        if terminate:
            console.print("\n[bold]Terminating instance...[/bold]")
            try:
                await provider.terminate_instance(job.instance_id)
                console.print("[green]Instance terminated.[/green]")
            except Exception as e:
                console.print(f"[yellow]Terminate failed: {e}[/yellow]")


async def recover_jobs(job_id: Optional[str] = None) -> None:
    """Manually trigger recovery for failed/preempted jobs."""
    await init_db()

    from sqlmodel import select
    from cloudcomputemanager.core.recovery import RecoveryManager

    recovery_manager = RecoveryManager()

    if job_id:
        # Recover specific job
        async with get_session() as session:
            stmt = select(Job).where(Job.job_id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()

        if not job:
            console.print(f"[red]Job not found: {job_id}[/red]")
            return

        if job.status not in [JobStatus.FAILED, JobStatus.RECOVERING]:
            console.print(f"[yellow]Job is not in recoverable state: {job.status.value}[/yellow]")
            # Ask if they want to force recovery
            console.print("[dim]Set job to RECOVERING state first if you want to retry[/dim]")
            return

        console.print(f"\n[bold]Recovering job:[/bold] {job_id}")
        console.print(f"  Name: {job.name}")
        console.print(f"  Attempt: {job.attempt_number + 1}")

        with console.status("Recovering..."):
            result = await recovery_manager.recover_job(job)

        if result.success:
            console.print(f"\n[bold green]Recovery successful![/bold green]")
            console.print(f"  New instance: {result.new_instance_id}")
            console.print(f"  Checkpoint restored: {result.checkpoint_restored}")
        else:
            console.print(f"\n[bold red]Recovery failed[/bold red]")
            console.print(f"  Error: {result.error}")

    else:
        # Recover all jobs in RECOVERING state
        async with get_session() as session:
            stmt = select(Job).where(Job.status == JobStatus.RECOVERING)
            result = await session.execute(stmt)
            jobs = result.scalars().all()

        if not jobs:
            console.print("[yellow]No jobs in RECOVERING state[/yellow]")
            return

        console.print(f"\n[bold]Recovering {len(jobs)} jobs...[/bold]")

        for job in jobs:
            console.print(f"\n  Recovering: {job.name} ({job.job_id})")
            try:
                result = await recovery_manager.recover_job(job)
                if result.success:
                    console.print(f"    [green]Success[/green] - Instance: {result.new_instance_id}")
                else:
                    console.print(f"    [red]Failed[/red] - {result.error}")
            except Exception as e:
                console.print(f"    [red]Error[/red] - {e}")

        console.print("\n[bold]Recovery complete[/bold]")


async def reconnect_jobs(job_id: Optional[str] = None) -> None:
    """Reconnect to jobs after daemon downtime.

    Checks each active job's instance status, reads exit codes,
    syncs results if completed, and updates DB state. Works
    entirely without the daemon — talks directly to Vast.ai API.
    """
    await init_db()

    from sqlmodel import select

    provider = VastProvider()
    settings = get_settings()

    # Find target jobs
    async with get_session() as session:
        if job_id:
            stmt = select(Job).where(Job.job_id == job_id)
        else:
            stmt = select(Job).where(Job.status.in_([
                JobStatus.RUNNING, JobStatus.RECOVERING,
                JobStatus.CHECKPOINTING, JobStatus.PROVISIONING,
            ]))
        result = await session.execute(stmt)
        jobs = result.scalars().all()

    if not jobs:
        console.print("[yellow]No active jobs to reconnect.[/yellow]")
        return

    console.print(f"\n[bold]Reconnecting {len(jobs)} job(s)...[/bold]\n")

    for job in jobs:
        await _reconnect_single_job(job, provider, settings)


async def _reconnect_single_job(job: Job, provider: VastProvider, settings) -> None:
    """Reconnect to a single job, rehydrating state from its instance."""
    console.print(f"[bold]{job.name}[/bold] ({job.job_id})")

    if not job.instance_id:
        console.print("  [red]No instance ID — cannot reconnect[/red]\n")
        return

    console.print(f"  Instance: {job.instance_id}")

    # Step 1: Check instance status via Vast.ai API
    try:
        instance = await provider.get_instance(job.instance_id)
    except Exception as e:
        console.print(f"  [red]Cannot reach Vast.ai API: {e}[/red]\n")
        return

    if instance is None:
        console.print("  [red]Instance not found (destroyed/expired)[/red]")
        async with get_session() as session:
            from sqlmodel import select as sel
            stmt = sel(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                db_job.status = JobStatus.FAILED
                db_job.error_message = "Instance not found on reconnect"
                db_job.completed_at = datetime.utcnow()
                session.add(db_job)
        sync_dir = settings.sync_local_path / job.job_id
        if sync_dir.exists():
            console.print(f"  [yellow]Local sync data at: {sync_dir}[/yellow]")
        console.print()
        return

    console.print(f"  Instance status: {instance.status.value}")

    from cloudcomputemanager.providers.base import ProviderStatus
    if instance.status not in (ProviderStatus.RUNNING, ProviderStatus.STARTING):
        console.print(f"  [yellow]Instance is {instance.status.value} — marking job for recovery[/yellow]")
        async with get_session() as session:
            from sqlmodel import select as sel
            stmt = sel(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job and db_job.attempt_number < 5:
                db_job.status = JobStatus.RECOVERING
                db_job.error_message = f"Instance {instance.status.value} on reconnect"
                session.add(db_job)
                console.print("  [yellow]Marked as RECOVERING — run `ccm jobs recover` to retry[/yellow]")
            elif db_job:
                db_job.status = JobStatus.FAILED
                db_job.error_message = f"Instance {instance.status.value}, max attempts reached"
                db_job.completed_at = datetime.utcnow()
                session.add(db_job)
                console.print("  [red]Marked as FAILED (max recovery attempts)[/red]")
        console.print()
        return

    # Step 2: Instance is running — check exit code sentinel
    completed, exit_code = await check_job_completion(provider, job.instance_id)

    if completed:
        status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
        color = "green" if exit_code == 0 else "red"
        console.print(f"  [{color}]Job finished with exit code {exit_code}[/{color}]")

        # Check preemption marker
        try:
            rc, stdout, _ = await provider.execute_command(
                job.instance_id,
                "cat /workspace/.ccm_preempted 2>/dev/null || echo 'none'",
            )
            if stdout.strip() != "none":
                console.print(f"  [yellow]Preemption detected: {stdout.strip()}[/yellow]")
        except Exception:
            pass

        # Sync results
        console.print("  Syncing results...")
        sync_config = job.sync_config or {}
        sync_dir = settings.sync_local_path / job.job_id
        sync_dir.mkdir(parents=True, exist_ok=True)
        success = await provider.rsync_download(
            job.instance_id,
            sync_config.get("source", "/workspace") + "/",
            str(sync_dir) + "/",
            exclude=sync_config.get("exclude_patterns", []),
        )
        if success:
            console.print(f"  [green]Synced to: {sync_dir}[/green]")
        else:
            console.print("  [red]Sync failed[/red]")

        # Update DB
        async with get_session() as session:
            from sqlmodel import select as sel
            stmt = sel(Job).where(Job.job_id == job.job_id)
            result = await session.execute(stmt)
            db_job = result.scalar_one_or_none()
            if db_job:
                db_job.status = status
                db_job.exit_code = exit_code
                db_job.completed_at = datetime.utcnow()
                session.add(db_job)

        console.print(f"  [dim]Terminate with: ccm instances terminate {job.instance_id}[/dim]")
    else:
        # Job is still running
        console.print("  [green]Job is still running[/green]")

        # Read heartbeat
        try:
            rc, stdout, _ = await provider.execute_command(
                job.instance_id,
                "cat /workspace/.ccm_heartbeat 2>/dev/null || echo 'no heartbeat'",
            )
            console.print(f"  Heartbeat: {stdout.strip()}")
        except Exception:
            pass

        # Read last log line
        try:
            rc, stdout, _ = await provider.execute_command(
                job.instance_id,
                "tail -1 /workspace/job.log 2>/dev/null || echo 'no logs'",
            )
            console.print(f"  Last log: {stdout.strip()[:100]}")
        except Exception:
            pass

    console.print()
