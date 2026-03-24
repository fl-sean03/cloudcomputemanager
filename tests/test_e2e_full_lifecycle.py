"""Full End-to-End Integration Test for CCM.

This test runs the COMPLETE job lifecycle against real Vast.ai:
1. Search for GPU offers
2. Create instance
3. Wait for ready
4. Execute job
5. Monitor completion
6. Sync results
7. Terminate instance

WARNING: This test COSTS MONEY! Run only after all other tests pass.

Usage:
    # Run only after all unit tests pass
    pytest tests/ -v --ignore=tests/test_e2e_full_lifecycle.py && \
    pytest tests/test_e2e_full_lifecycle.py -v --run-e2e

    # Or explicitly:
    pytest tests/test_e2e_full_lifecycle.py -v --run-e2e -s
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from cloudcomputemanager.providers.vast import VastProvider
from cloudcomputemanager.providers.base import ProviderStatus
from cloudcomputemanager.core.config import get_settings
from cloudcomputemanager.core.templates import load_config_with_template, normalize_resources


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: mark test as full end-to-end test (requires --run-e2e, costs money)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-e2e", default=False):
        skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run (costs money!)")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--run-e2e",
            action="store_true",
            default=False,
            help="run full end-to-end tests (costs money!)",
        )
    except ValueError:
        # Option already added by conftest.py
        pass


# Track instances for emergency cleanup
_test_instances: list[str] = []


@pytest.fixture(scope="module")
def vast_provider() -> VastProvider:
    """Create VastProvider with real API key."""
    settings = get_settings()
    api_key = settings.get_vast_api_key()
    if not api_key:
        pytest.skip("No Vast.ai API key configured")
    return VastProvider(api_key=api_key)


@pytest.fixture(scope="module", autouse=True)
def emergency_cleanup(vast_provider: VastProvider):
    """Emergency cleanup of any created instances."""
    yield
    # Cleanup after all tests
    for instance_id in _test_instances:
        try:
            asyncio.get_event_loop().run_until_complete(
                vast_provider.terminate_instance(instance_id)
            )
            print(f"[CLEANUP] Terminated instance {instance_id}")
        except Exception as e:
            print(f"[CLEANUP ERROR] Failed to terminate {instance_id}: {e}")


@pytest.mark.e2e
class TestFullJobLifecycle:
    """Complete end-to-end job lifecycle test.

    This test creates a real instance, runs a real job, and verifies:
    - GPU type respects user config (no default override)
    - Job executes successfully
    - Results are synced back
    - Instance is properly terminated
    """

    @pytest.mark.timeout(600)  # 10 minute timeout
    async def test_complete_job_lifecycle(self, vast_provider: VastProvider, tmp_path: Path):
        """Test the complete job lifecycle from submit to results."""
        print("\n" + "="*60)
        print("FULL E2E LIFECYCLE TEST - THIS COSTS MONEY!")
        print("="*60)

        # Test parameters - use cheap GPU
        gpu_type = "RTX_3060"
        max_hourly_rate = 0.10
        job_command = """
echo "E2E Test Job Started at $(date)"
echo "Testing CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "No torch"
echo "Creating test output..."
echo "E2E test output" > /workspace/e2e_output.txt
date >> /workspace/e2e_output.txt
hostname >> /workspace/e2e_output.txt
echo "E2E Test Job Completed at $(date)"
"""

        # Step 1: Verify GPU query uses correct type (not default RTX_4090)
        print("\n[STEP 1] Searching for GPU offers...")
        offers = await vast_provider.search_offers(
            gpu_type=gpu_type,
            gpu_memory_min=8,
            max_hourly_rate=max_hourly_rate,
        )

        assert len(offers) > 0, f"No {gpu_type} offers found under ${max_hourly_rate}/hr"
        best_offer = offers[0]

        print(f"  Found {len(offers)} offers")
        print(f"  Best: {best_offer.gpu_type} at ${best_offer.hourly_rate}/hr")
        print(f"  Location: {best_offer.location}")

        # Verify we got the right GPU type
        assert gpu_type.replace("_", " ") in best_offer.gpu_type or gpu_type in best_offer.gpu_type, \
            f"Expected {gpu_type}, got {best_offer.gpu_type}"

        # Step 2: Create instance
        print("\n[STEP 2] Creating instance...")
        instance = await vast_provider.create_instance(
            offer_id=best_offer.offer_id,
            image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
            disk_gb=20,
        )
        _test_instances.append(instance.instance_id)

        print(f"  Instance ID: {instance.instance_id}")
        print(f"  SSH: ssh -p {instance.ssh_port} {instance.ssh_user}@{instance.ssh_host}")

        # Step 3: Wait for instance to be ready
        print("\n[STEP 3] Waiting for instance to be ready...")
        ready = await vast_provider.wait_for_ready(
            instance.instance_id,
            timeout=300,  # 5 minutes
        )

        if not ready:
            # Cleanup and fail
            await vast_provider.terminate_instance(instance.instance_id)
            _test_instances.remove(instance.instance_id)
            pytest.fail("Instance failed to become ready within 5 minutes")

        print("  Instance is ready!")

        # Step 4: Execute the job
        print("\n[STEP 4] Executing job...")

        # Create wrapper script
        import base64
        wrapper_script = f'''#!/bin/bash
set -e
cd /workspace
{job_command}
echo $? > /workspace/.e2e_exit_code
'''
        script_b64 = base64.b64encode(wrapper_script.encode()).decode()

        # Upload and execute script
        setup_cmd = f"echo {script_b64} | base64 -d > /workspace/e2e_job.sh && chmod +x /workspace/e2e_job.sh"
        exit_code, stdout, stderr = await vast_provider.execute_command(
            instance.instance_id, setup_cmd
        )
        assert exit_code == 0, f"Failed to create job script: {stderr}"

        # Run the job (not in background so we wait for it)
        run_cmd = "cd /workspace && bash /workspace/e2e_job.sh"
        exit_code, stdout, stderr = await vast_provider.execute_command(
            instance.instance_id, run_cmd, timeout=120
        )

        print(f"  Job output:\n{stdout}")
        if stderr:
            print(f"  Job stderr:\n{stderr}")

        assert exit_code == 0, f"Job failed with exit code {exit_code}"

        # Step 5: Verify output file exists
        print("\n[STEP 5] Verifying output...")
        exit_code, stdout, stderr = await vast_provider.execute_command(
            instance.instance_id, "cat /workspace/e2e_output.txt"
        )

        assert exit_code == 0, "Output file not found"
        assert "E2E test output" in stdout, f"Expected output not found: {stdout}"
        print(f"  Output verified: {stdout.strip()}")

        # Step 6: Sync results
        print("\n[STEP 6] Syncing results...")
        sync_dir = tmp_path / "e2e_results"
        sync_dir.mkdir()

        success = await vast_provider.rsync_download(
            instance.instance_id,
            "/workspace/",
            str(sync_dir) + "/",
        )

        assert success, "Failed to sync results"

        # Verify synced files
        output_file = sync_dir / "e2e_output.txt"
        assert output_file.exists(), f"Output file not synced. Contents: {list(sync_dir.iterdir())}"
        content = output_file.read_text()
        assert "E2E test output" in content

        print(f"  Results synced to: {sync_dir}")
        print(f"  Files: {[f.name for f in sync_dir.iterdir()]}")

        # Step 7: Terminate instance
        print("\n[STEP 7] Terminating instance...")
        terminated = await vast_provider.terminate_instance(instance.instance_id)
        _test_instances.remove(instance.instance_id)

        assert terminated, "Failed to terminate instance"
        print("  Instance terminated successfully!")

        # Final summary
        print("\n" + "="*60)
        print("E2E TEST PASSED!")
        print(f"  GPU: {best_offer.gpu_type}")
        print(f"  Cost: ~${best_offer.hourly_rate * 0.1:.4f} (estimated)")
        print("="*60)


@pytest.mark.e2e
class TestAgentSDKE2E:
    """End-to-end test using the Agent SDK."""

    @pytest.mark.timeout(600)
    async def test_agent_sdk_job_lifecycle(self, tmp_path: Path):
        """Test complete lifecycle through Agent SDK."""
        from cloudcomputemanager.agents.sdk import CloudComputeManagerAgent, JobSpec

        print("\n" + "="*60)
        print("AGENT SDK E2E TEST")
        print("="*60)

        async with CloudComputeManagerAgent() as agent:
            # Track events
            events = []
            agent.on_event(lambda e: events.append(e))

            # Create job spec with non-default GPU
            spec = JobSpec(
                name="e2e-agent-test",
                command="echo 'Agent SDK E2E Test' && python -c 'print(1+1)'",
                image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                gpu_type="RTX_3060",  # Explicit, not default
                gpu_count=1,
                disk_gb=20,
                max_hourly_rate=0.10,
                max_cost_usd=0.10,
                checkpoint_enabled=False,  # Skip for speed
                sync_enabled=False,
            )

            print(f"\n[SUBMIT] Submitting job: {spec.name}")
            print(f"  GPU: {spec.gpu_type}")

            try:
                job = await agent.submit(spec, wait_for_start=True)
                _test_instances.append(job.instance_id)

                print(f"  Job ID: {job.job_id}")
                print(f"  Instance: {job.instance_id}")

                # Wait for completion (short timeout since it's a quick job)
                print("\n[WAIT] Waiting for completion...")
                result = await agent.wait_for_completion(
                    job.job_id,
                    timeout=180,  # 3 minutes
                    poll_interval=10,
                )

                print(f"  Result: {result.status}")
                print(f"  Success: {result.success}")

                # Verify events were captured
                event_types = [e.type.value for e in events]
                print(f"\n[EVENTS] Captured: {event_types}")

                assert "instance.created" in event_types
                assert "job.submitted" in event_types

                # Cancel/cleanup
                await agent.cancel(job.job_id)
                if job.instance_id in _test_instances:
                    _test_instances.remove(job.instance_id)

                print("\n" + "="*60)
                print("AGENT SDK E2E TEST PASSED!")
                print("="*60)

            except Exception as e:
                # Emergency cleanup
                print(f"\n[ERROR] {e}")
                raise


@pytest.mark.e2e
class TestConfigNormalizationE2E:
    """E2E test that config normalization works in real submission."""

    @pytest.mark.timeout(300)
    async def test_gpu_memory_gb_normalized_in_real_search(self, vast_provider: VastProvider):
        """Verify gpu_memory_gb is normalized to gpu_memory_min in real API calls."""
        print("\n" + "="*60)
        print("CONFIG NORMALIZATION E2E TEST")
        print("="*60)

        # Simulate loading a config with alternate key names
        test_config = {
            "resources": {
                "gpu_type": "RTX_3060",
                "gpu_memory_gb": 8,  # Alternate name
                "disk": 20,  # Alternate name
            }
        }

        # Normalize like load_config_with_template does
        normalized = normalize_resources(test_config["resources"])

        print(f"  Original: {test_config['resources']}")
        print(f"  Normalized: {normalized}")

        assert "gpu_memory_min" in normalized
        assert normalized["gpu_memory_min"] == 8
        assert "gpu_memory_gb" not in normalized

        # Now use normalized values in real search
        offers = await vast_provider.search_offers(
            gpu_type=normalized.get("gpu_type"),
            gpu_memory_min=normalized.get("gpu_memory_min"),
        )

        assert len(offers) > 0, "No offers found"
        print(f"  Found {len(offers)} offers using normalized config")
        print("  CONFIG NORMALIZATION VERIFIED!")
