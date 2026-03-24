"""Vast.ai provider implementation.

Uses the vastai CLI for API operations with SSH-based command execution
and rsync file transfers.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

import structlog

SSH_MAX_RETRIES = 3
SSH_RETRY_BACKOFF = [2, 4, 8]  # seconds between retries

from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.providers.base import (
    CloudProvider,
    ProviderInstance,
    ProviderOffer,
    ProviderStatus,
)

logger = structlog.get_logger(__name__)


class VastProvider(CloudProvider):
    """Vast.ai cloud provider implementation.

    Uses the vastai CLI for API operations. The vastai Python SDK can also
    be used, but the CLI provides more stable behavior.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Vast.ai provider.

        Args:
            api_key: Vast.ai API key (uses settings if not provided)
        """
        settings = get_settings()
        self._api_key = api_key or settings.get_vast_api_key()
        self._ssh_key_path = settings.ssh_key_path

    @property
    def name(self) -> str:
        return "vast"

    async def _run_vastai_cmd(
        self,
        *args: str,
        parse_json: bool = True,
    ) -> dict | list | str:
        """Run a vastai CLI command.

        Args:
            *args: Command arguments
            parse_json: Whether to parse output as JSON

        Returns:
            Parsed JSON or raw output
        """
        cmd = ["vastai", "--api-key", self._api_key, *args]

        logger.debug("Running vastai command", cmd=" ".join(["vastai", "--api-key", "***"] + list(args)))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            logger.error("Vastai command failed", error=error_msg, returncode=proc.returncode)
            raise RuntimeError(f"vastai command failed: {error_msg}")

        output = stdout.decode().strip()

        if parse_json and output:
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                # Some commands return non-JSON output
                return output

        return output

    async def search_offers(
        self,
        gpu_type: Optional[str] = None,
        gpu_count: int = 1,
        gpu_memory_min: Optional[int] = None,
        disk_gb_min: int = 50,
        max_hourly_rate: Optional[float] = None,
        interruptible: bool = True,
        cpu_cores_min: Optional[int] = None,
    ) -> list[ProviderOffer]:
        """Search for available GPU offers on Vast.ai.

        Args:
            gpu_type: GPU type to search for (e.g., "RTX_4090"). If None, any GPU type.
            gpu_count: Number of GPUs required.
            gpu_memory_min: Minimum GPU memory in GB. If None, defaults to 8GB.
            disk_gb_min: Minimum disk space in GB.
            max_hourly_rate: Maximum hourly rate. If None, no price limit.
            interruptible: Whether to search for interruptible (spot) instances.
            cpu_cores_min: Minimum number of CPU cores. If None, no CPU filter.

        Returns:
            List of ProviderOffer objects sorted by score.
        """
        # Use sensible default for gpu_memory_min if not specified
        effective_gpu_memory = gpu_memory_min if gpu_memory_min is not None else 8

        # Build search query
        query_parts = [
            f"num_gpus={gpu_count}",
            f"gpu_ram>={effective_gpu_memory}",
            f"disk_space>={disk_gb_min}",
            "verified=true",
            "rented=false",
        ]

        if gpu_type:
            # Vast.ai requires underscores instead of spaces in GPU names
            gpu_type_normalized = gpu_type.replace(" ", "_")
            query_parts.append(f"gpu_name={gpu_type_normalized}")

        if max_hourly_rate:
            price_field = "dph_total" if interruptible else "dph_base"
            query_parts.append(f"{price_field}<={max_hourly_rate}")

        if interruptible:
            query_parts.append("rentable=true")

        if cpu_cores_min:
            # Filter by minimum CPU cores (effective cores)
            query_parts.append(f"cpu_cores_effective>={cpu_cores_min}")

        query = " ".join(query_parts)

        logger.info("Searching Vast.ai offers", query=query)

        # Run search
        results = await self._run_vastai_cmd(
            "search", "offers", query, "--raw", "--order", "dph_total"
        )

        if not isinstance(results, list):
            logger.warning("Unexpected search result format", result=results)
            return []

        # Convert to ProviderOffer objects
        offers = []
        for r in results[:20]:  # Limit to top 20 offers
            try:
                offers.append(
                    ProviderOffer(
                        offer_id=str(r["id"]),
                        provider="vast",
                        gpu_type=r.get("gpu_name", "Unknown"),
                        gpu_count=r.get("num_gpus", 1),
                        gpu_memory_gb=int(r.get("gpu_ram", 0) / 1024),  # MB to GB
                        cpu_cores=r.get("cpu_cores_effective", 0),
                        memory_gb=int(r.get("cpu_ram", 0) / 1024),  # MB to GB
                        disk_gb=int(r.get("disk_space", 0)),
                        hourly_rate=r.get("dph_total", 0),
                        location=r.get("geolocation", "Unknown"),
                        reliability_score=r.get("reliability2", 0.9),
                        interruptible=interruptible,
                        cuda_version=r.get("cuda_max_good"),
                        driver_version=r.get("driver_version"),
                    )
                )
            except (KeyError, TypeError) as e:
                logger.warning("Failed to parse offer", error=str(e), offer=r)

        # Sort by score
        offers.sort(key=lambda o: o.score)

        logger.info("Found offers", count=len(offers))
        return offers

    async def create_instance(
        self,
        offer_id: str,
        image: str,
        disk_gb: int = 50,
        ssh_public_key: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
        startup_script: Optional[str] = None,
    ) -> ProviderInstance:
        """Create a new instance on Vast.ai."""
        logger.info("Creating instance", offer_id=offer_id, image=image, disk_gb=disk_gb)

        # Load SSH public key if not provided
        if not ssh_public_key and self._ssh_key_path.exists():
            pub_key_path = Path(str(self._ssh_key_path) + ".pub")
            if pub_key_path.exists():
                ssh_public_key = pub_key_path.read_text().strip()

        # Build create command
        args = [
            "create", "instance",
            offer_id,
            "--image", image,
            "--disk", str(disk_gb),
            "--raw",
        ]

        if ssh_public_key:
            # Use --ssh for Vast.ai managed SSH key setup (uses registered SSH keys)
            args.extend(["--ssh", "--direct"])

        # Build onstart script
        # Note: SSH key setup is handled by Vast.ai via --ssh flag and registered keys
        # We don't need to add the key via onstart script
        onstart_parts = [
            "touch ~/.no_auto_tmux",
            f'echo "{self._api_key}" > ~/.vast_api_key && chmod 600 ~/.vast_api_key',
        ]

        if env_vars:
            for key, value in env_vars.items():
                onstart_parts.append(f'export {key}="{value}"')
                onstart_parts.append(f'echo "export {key}=\"{value}\"" >> ~/.bashrc')

        if startup_script:
            # Convert multiline setup script to single-line shell commands.
            # Vast.ai onstart-cmd is passed as a single string, so newlines
            # need to be converted to command separators.
            lines = [line.strip() for line in startup_script.strip().split("\n") if line.strip()]
            onstart_parts.extend(lines)

        # Write a sentinel file when setup completes so CCM can detect readiness
        onstart_parts.append("touch /tmp/.ccm_setup_done")

        # Start a background heartbeat that writes instance health to disk every 60s.
        # This lets CCM reconnect and determine when the instance was last alive,
        # even if the daemon was down for hours.
        heartbeat_cmd = (
            "(while true; do "
            "date -u +%Y-%m-%dT%H:%M:%SZ > /workspace/.ccm_heartbeat; "
            "sleep 60; done) &"
        )
        onstart_parts.append(heartbeat_cmd)

        onstart = " && ".join(onstart_parts)
        args.extend(["--onstart-cmd", onstart])

        # Create the instance
        result = await self._run_vastai_cmd(*args)

        if isinstance(result, dict) and "new_contract" in result:
            instance_id = str(result["new_contract"])
        elif isinstance(result, str) and "Started" in result:
            # Parse "Started. Instance ID is XXXXX"
            instance_id = result.split()[-1]
        else:
            raise RuntimeError(f"Unexpected create result: {result}")

        logger.info("Instance created", instance_id=instance_id)

        # Wait briefly and get instance details
        await asyncio.sleep(5)
        instance = await self.get_instance(instance_id)

        if not instance:
            raise RuntimeError(f"Failed to get created instance {instance_id}")

        return instance

    def _parse_instance(self, data: dict) -> ProviderInstance:
        """Parse instance data from Vast.ai API."""
        # Map status
        # Handle None value (API can return null for actual_status)
        vast_status = (data.get("actual_status") or "unknown").lower()
        status_map = {
            "loading": ProviderStatus.STARTING,
            "running": ProviderStatus.RUNNING,
            "exited": ProviderStatus.STOPPED,
            "stopped": ProviderStatus.STOPPED,
            "error": ProviderStatus.ERROR,
        }
        status = status_map.get(vast_status, ProviderStatus.PENDING)

        # Get SSH info
        # Note: ssh_host + ssh_port are for the SSH proxy (sshX.vast.ai:port)
        # The ports mapping is for direct connections to the host IP
        # When using SSH proxy (which is the default), use ssh_host + ssh_port
        ssh_host = data.get("ssh_host", "")
        ssh_port = data.get("ssh_port", 22)

        # Only use port mapping if not using SSH proxy (i.e., direct connection)
        # SSH proxy hosts look like "sshX.vast.ai"
        ports = data.get("ports", {})
        if not ssh_host.endswith(".vast.ai") and ports and "22/tcp" in ports:
            port_info = ports["22/tcp"]
            if isinstance(port_info, list) and port_info:
                ssh_port = int(port_info[0].get("HostPort", ssh_port))
            elif isinstance(port_info, dict):
                ssh_port = int(port_info.get("HostPort", ssh_port))

        # Jupyter URL
        jupyter_url = None
        if ports and "8888/tcp" in ports:
            port_info = ports["8888/tcp"]
            jupyter_port = ssh_port  # Default fallback
            if isinstance(port_info, list) and port_info:
                jupyter_port = int(port_info[0].get("HostPort", jupyter_port))
            jupyter_url = f"http://{ssh_host}:{jupyter_port}"

        return ProviderInstance(
            instance_id=str(data["id"]),
            provider="vast",
            status=status,
            gpu_type=data.get("gpu_name", "Unknown"),
            gpu_count=data.get("num_gpus", 1),
            gpu_memory_gb=int(data.get("gpu_ram", 0) / 1024),
            cpu_cores=data.get("cpu_cores_effective", 0),
            memory_gb=int(data.get("cpu_ram", 0) / 1024),
            disk_gb=int(data.get("disk_space", 0)),
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_user="root",
            hourly_rate=data.get("dph_total", 0),
            interruptible=data.get("is_bid", True),
            jupyter_url=jupyter_url,
            internal_ip=data.get("local_ipaddrs"),
            external_ip=data.get("public_ipaddr"),
        )

    async def get_instance(self, instance_id: str) -> Optional[ProviderInstance]:
        """Get instance details from Vast.ai."""
        try:
            result = await self._run_vastai_cmd("show", "instance", instance_id, "--raw")

            if isinstance(result, dict):
                return self._parse_instance(result)

            # Sometimes returns a list with one item
            if isinstance(result, list) and result:
                return self._parse_instance(result[0])

            return None

        except RuntimeError as e:
            error_str = str(e).lower()
            if "not found" in error_str:
                return None
            # Vast.ai CLI crashes with TypeError when instance is still booting
            # (start_date is None). Treat as "not ready yet" instead of fatal.
            if "typeerror" in error_str or "nonetype" in error_str:
                logger.debug("Instance not ready yet (CLI error)", instance_id=instance_id)
                return None
            raise

    async def list_instances(self) -> list[ProviderInstance]:
        """List all instances on Vast.ai."""
        result = await self._run_vastai_cmd("show", "instances", "--raw")

        if not isinstance(result, list):
            return []

        return [self._parse_instance(data) for data in result]

    async def start_instance(self, instance_id: str) -> bool:
        """Start a stopped instance."""
        try:
            await self._run_vastai_cmd("start", "instance", instance_id)
            return True
        except RuntimeError:
            return False

    async def stop_instance(self, instance_id: str) -> bool:
        """Stop a running instance."""
        try:
            await self._run_vastai_cmd("stop", "instance", instance_id)
            return True
        except RuntimeError:
            return False

    async def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance."""
        try:
            await self._run_vastai_cmd("destroy", "instance", instance_id)
            return True
        except RuntimeError:
            return False

    def _build_ssh_cmd(
        self,
        host: str,
        port: int,
        user: str,
        command: str,
        timeout: int = 60,
    ) -> list[str]:
        """Build an SSH command list."""
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={min(timeout, 30)}",
            "-p", str(port),
            f"{user}@{host}",
            command,
        ]

        if self._ssh_key_path.exists():
            ssh_cmd.insert(1, "-i")
            ssh_cmd.insert(2, str(self._ssh_key_path))

        return ssh_cmd

    async def execute_command(
        self,
        instance_id: str,
        command: str,
        timeout: int = 60,
    ) -> tuple[int, str, str]:
        """Execute a command on an instance via SSH with retry on connection failures."""
        instance = await self.get_instance(instance_id)
        if not instance:
            raise RuntimeError(f"Instance {instance_id} not found")

        if instance.status != ProviderStatus.RUNNING:
            raise RuntimeError(f"Instance {instance_id} is not running")

        ssh_cmd = self._build_ssh_cmd(
            instance.ssh_host, instance.ssh_port, instance.ssh_user, command, timeout
        )

        last_error = ""
        for attempt in range(SSH_MAX_RETRIES):
            logger.debug(
                "Executing SSH command",
                host=instance.ssh_host,
                command=command[:50],
                attempt=attempt + 1,
            )

            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return -1, "", "Command timed out"

            returncode = proc.returncode or 0
            stderr_str = stderr.decode()

            # SSH connection errors warrant a retry; command failures do not
            if returncode == 255 and attempt < SSH_MAX_RETRIES - 1:
                # Exit code 255 = SSH connection failure
                backoff = SSH_RETRY_BACKOFF[min(attempt, len(SSH_RETRY_BACKOFF) - 1)]
                last_error = stderr_str
                logger.warning(
                    "SSH connection failed, retrying",
                    host=instance.ssh_host,
                    attempt=attempt + 1,
                    backoff=backoff,
                    error=stderr_str[:100],
                )
                await asyncio.sleep(backoff)
                continue

            return returncode, stdout.decode(), stderr_str

        # All retries exhausted
        return 255, "", f"SSH connection failed after {SSH_MAX_RETRIES} attempts: {last_error}"

    async def rsync_download(
        self,
        instance_id: str,
        remote_path: str,
        local_path: str,
        exclude: Optional[list[str]] = None,
    ) -> bool:
        """Download files from instance using rsync.

        Args:
            instance_id: Instance ID
            remote_path: Path on instance
            local_path: Local destination path
            exclude: Patterns to exclude

        Returns:
            True if successful
        """
        instance = await self.get_instance(instance_id)
        if not instance or instance.status != ProviderStatus.RUNNING:
            return False

        # Build rsync command
        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e", f"ssh -o StrictHostKeyChecking=no -p {instance.ssh_port}",
        ]

        if self._ssh_key_path.exists():
            rsync_cmd[-1] = f"ssh -i {self._ssh_key_path} -o StrictHostKeyChecking=no -p {instance.ssh_port}"

        if exclude:
            for pattern in exclude:
                rsync_cmd.extend(["--exclude", pattern])

        rsync_cmd.extend([
            f"{instance.ssh_user}@{instance.ssh_host}:{remote_path}",
            local_path,
        ])

        logger.info("Running rsync download", remote=remote_path, local=local_path)

        for attempt in range(SSH_MAX_RETRIES):
            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode == 0:
                return True

            stderr_str = stderr.decode()
            if attempt < SSH_MAX_RETRIES - 1:
                backoff = SSH_RETRY_BACKOFF[min(attempt, len(SSH_RETRY_BACKOFF) - 1)]
                logger.warning("Rsync download failed, retrying", attempt=attempt + 1, backoff=backoff, error=stderr_str[:100])
                await asyncio.sleep(backoff)
            else:
                logger.error("Rsync download failed after retries", error=stderr_str)

        return False

    async def rsync_upload(
        self,
        instance_id: str,
        local_path: str,
        remote_path: str,
        exclude: Optional[list[str]] = None,
    ) -> bool:
        """Upload files to instance using rsync.

        Args:
            instance_id: Instance ID
            local_path: Local source path
            remote_path: Path on instance
            exclude: Patterns to exclude

        Returns:
            True if successful
        """
        instance = await self.get_instance(instance_id)
        if not instance or instance.status != ProviderStatus.RUNNING:
            return False

        # Build rsync command
        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e", f"ssh -o StrictHostKeyChecking=no -p {instance.ssh_port}",
        ]

        if self._ssh_key_path.exists():
            rsync_cmd[-1] = f"ssh -i {self._ssh_key_path} -o StrictHostKeyChecking=no -p {instance.ssh_port}"

        if exclude:
            for pattern in exclude:
                rsync_cmd.extend(["--exclude", pattern])

        rsync_cmd.extend([
            local_path,
            f"{instance.ssh_user}@{instance.ssh_host}:{remote_path}",
        ])

        logger.info("Running rsync upload", local=local_path, remote=remote_path)

        for attempt in range(SSH_MAX_RETRIES):
            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode == 0:
                return True

            stderr_str = stderr.decode()
            if attempt < SSH_MAX_RETRIES - 1:
                backoff = SSH_RETRY_BACKOFF[min(attempt, len(SSH_RETRY_BACKOFF) - 1)]
                logger.warning("Rsync upload failed, retrying", attempt=attempt + 1, backoff=backoff, error=stderr_str[:100])
                await asyncio.sleep(backoff)
            else:
                logger.error("Rsync upload failed after retries", error=stderr_str)

        return False
