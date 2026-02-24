"""Vast.ai provider implementation.

Uses the vastai-sdk for direct API access, with SkyPilot integration for
job orchestration when available.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

import structlog

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
        gpu_memory_min: int = 16,
        disk_gb_min: int = 50,
        max_hourly_rate: Optional[float] = None,
        interruptible: bool = True,
    ) -> list[ProviderOffer]:
        """Search for available GPU offers on Vast.ai."""
        # Build search query
        query_parts = [
            f"num_gpus={gpu_count}",
            f"gpu_ram>={gpu_memory_min}",
            f"disk_space>={disk_gb_min}",
            "verified=true",
            "rented=false",
        ]

        if gpu_type:
            query_parts.append(f"gpu_name={gpu_type}")

        if max_hourly_rate:
            price_field = "dph_total" if interruptible else "dph_base"
            query_parts.append(f"{price_field}<={max_hourly_rate}")

        if interruptible:
            query_parts.append("rentable=true")

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
            f'echo "{self._api_key}" > ~/.vast_api_key',
        ]

        if env_vars:
            for key, value in env_vars.items():
                onstart_parts.append(f'export {key}="{value}"')
                onstart_parts.append(f'echo "export {key}=\"{value}\"" >> ~/.bashrc')

        if startup_script:
            onstart_parts.append(startup_script)

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
            if "not found" in str(e).lower():
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

    async def execute_command(
        self,
        instance_id: str,
        command: str,
        timeout: int = 60,
    ) -> tuple[int, str, str]:
        """Execute a command on an instance via SSH."""
        instance = await self.get_instance(instance_id)
        if not instance:
            raise RuntimeError(f"Instance {instance_id} not found")

        if instance.status != ProviderStatus.RUNNING:
            raise RuntimeError(f"Instance {instance_id} is not running")

        # Build SSH command
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={min(timeout, 30)}",
            "-p", str(instance.ssh_port),
            f"{instance.ssh_user}@{instance.ssh_host}",
            command,
        ]

        if self._ssh_key_path.exists():
            ssh_cmd.insert(1, "-i")
            ssh_cmd.insert(2, str(self._ssh_key_path))

        logger.debug("Executing SSH command", host=instance.ssh_host, command=command[:50])

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

        return proc.returncode or 0, stdout.decode(), stderr.decode()

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

        proc = await asyncio.create_subprocess_exec(
            *rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("Rsync failed", error=stderr.decode())
            return False

        return True

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

        proc = await asyncio.create_subprocess_exec(
            *rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error("Rsync failed", error=stderr.decode())
            return False

        return True
