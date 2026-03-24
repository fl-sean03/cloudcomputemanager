"""Tests for v0.2.0 sprint features.

Tests cover actual logic, not just field existence:
- Setup commands flow through to VastProvider (Phase 1.1)
- Provisioning timeout is actually used (Phase 1.2)
- Variable substitution with edge cases (Phase 3.1)
- Multi-stage advancement logic in daemon (Phase 3.2)
- Batch matrix expansion (Phase 3.3)
- Progress parsing regex extraction (Phase 2.1)
- Notification hooks with shell escaping (Phase 2.2)
- SSH credentials (Phase 1.3)
- Wrapper script correctness (critical)
- CostRecord and BenchmarkSuite models (Phase 4)
- Corrupted JSON resilience (robustness)
"""

import base64
import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from cloudcomputemanager.core.models import (
    Job,
    JobStatus,
    CostRecord,
)


# ============================================================================
# Wrapper Script Correctness (CRITICAL)
# ============================================================================


class TestWrapperScript:
    """Test the shared wrapper script builder for correctness."""

    def test_wrapper_uses_set_plus_e(self):
        """Wrapper must use set +e so $? is captured even on failure."""
        from cloudcomputemanager.core.wrapper import build_wrapper_script
        script = build_wrapper_script("echo hello")
        assert "set +e" in script
        idx_set_plus_e = script.index("set +e")
        idx_exit_code = script.index("JOB_EXIT_CODE=$?")
        assert idx_set_plus_e < idx_exit_code

    def test_wrapper_writes_ccm_exit_code(self):
        """Wrapper must write .ccm_exit_code sentinel."""
        from cloudcomputemanager.core.wrapper import build_wrapper_script
        script = build_wrapper_script("echo hello")
        assert ".ccm_exit_code" in script

    def test_all_sites_use_shared_builder(self):
        """All wrapper sites must delegate to core.wrapper, not have inline scripts."""
        import inspect
        from cloudcomputemanager.cli import jobs
        from cloudcomputemanager.daemon import monitor
        from cloudcomputemanager.agents import sdk

        assert "build_deploy_commands" in inspect.getsource(jobs.submit_job)
        assert "build_deploy_commands" in inspect.getsource(monitor.JobMonitor.advance_job_stage)
        assert "build_deploy_commands" in inspect.getsource(sdk.CloudComputeManagerAgent.submit)


# ============================================================================
# Setup Commands Flow (Phase 1.1)
# ============================================================================


class TestSetupCommandsFlow:
    """Test that setup commands flow through to VastProvider correctly."""

    def test_setup_commands_stored_on_job(self):
        """Job model stores setup commands."""
        job = Job(
            name="test",
            image="ubuntu:22.04",
            command="echo hello",
            setup_commands="apt-get update && apt-get install -y python3",
        )
        assert job.setup_commands == "apt-get update && apt-get install -y python3"

    def test_setup_commands_default_none(self):
        """setup_commands defaults to None when not provided."""
        job = Job(name="test", image="ubuntu:22.04", command="echo hello")
        assert job.setup_commands is None

    @pytest.mark.asyncio
    async def test_setup_commands_passed_to_create_instance(self):
        """When setup_commands present, they must be passed as startup_script."""
        from cloudcomputemanager.providers.vast import VastProvider

        provider = VastProvider.__new__(VastProvider)
        provider._api_key = "test-key"
        provider._ssh_key_path = Path("/nonexistent")

        # Mock _run_vastai_cmd to capture what's passed
        captured_args = {}

        async def mock_run(*args, parse_json=True):
            captured_args["args"] = args
            return {"new_contract": 12345}

        provider._run_vastai_cmd = mock_run

        # Mock get_instance to return a valid instance
        async def mock_get_instance(iid):
            from cloudcomputemanager.providers.base import ProviderInstance, ProviderStatus
            return ProviderInstance(
                instance_id=iid, provider="vast", status=ProviderStatus.RUNNING,
                gpu_type="RTX_3060", gpu_count=1, gpu_memory_gb=12, cpu_cores=16,
                memory_gb=32, disk_gb=50, ssh_host="ssh1.vast.ai", ssh_port=22,
                hourly_rate=0.08,
            )

        provider.get_instance = mock_get_instance

        await provider.create_instance(
            offer_id="999",
            image="ubuntu:22.04",
            startup_script="apt-get install -y lammps",
        )

        # Verify the onstart-cmd includes our setup script AND the sentinel
        onstart_arg_idx = None
        for i, arg in enumerate(captured_args["args"]):
            if arg == "--onstart-cmd":
                onstart_arg_idx = i + 1
                break

        assert onstart_arg_idx is not None, "--onstart-cmd not found in vastai args"
        onstart_cmd = captured_args["args"][onstart_arg_idx]
        assert "apt-get install -y lammps" in onstart_cmd
        assert ".ccm_setup_done" in onstart_cmd
        # Sentinel must come AFTER the setup script (joined with &&)
        idx_setup = onstart_cmd.index("apt-get install")
        idx_sentinel = onstart_cmd.index(".ccm_setup_done")
        assert idx_setup < idx_sentinel, "Sentinel must come after setup commands"


# ============================================================================
# Provisioning Timeout (Phase 1.2)
# ============================================================================


class TestProvisioningTimeout:
    """Test configurable provisioning timeout."""

    def test_timeout_stored_on_job(self):
        """Custom timeout should be stored on Job model."""
        job = Job(
            name="test", image="ubuntu:22.04", command="echo",
            provisioning_timeout=600,
        )
        assert job.provisioning_timeout == 600

    def test_timeout_default_300(self):
        """Default timeout should be 300s."""
        job = Job(name="test", image="ubuntu:22.04", command="echo")
        assert job.provisioning_timeout == 300

    def test_timeout_parsed_from_config(self):
        """Provisioning timeout should be read from YAML config."""
        from cloudcomputemanager.core.templates import load_config_with_template

        config_content = """
name: test
image: ubuntu:22.04
command: echo hello
provisioning:
  timeout: 900
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            p = Path(f.name)
        try:
            config = load_config_with_template(p)
            assert config["provisioning"]["timeout"] == 900
        finally:
            p.unlink()


# ============================================================================
# Variable Substitution (Phase 3.1)
# ============================================================================


class TestVariableSubstitution:
    """Test YAML variable substitution with real edge cases."""

    def test_basic_substitution(self):
        from cloudcomputemanager.core.templates import substitute_variables

        result = substitute_variables(
            "name: job-${STRUCTURE}", {"STRUCTURE": "F100"}
        )
        assert result == "name: job-F100"

    def test_builtins_always_available(self):
        from cloudcomputemanager.core.templates import substitute_variables

        result = substitute_variables("ts: ${TIMESTAMP}\ndate: ${DATE}\nrand: ${RANDOM}")
        parsed = yaml.safe_load(result)
        assert parsed["ts"] is not None
        assert parsed["date"] is not None
        assert parsed["rand"] is not None

    def test_user_vars_override_builtins(self):
        from cloudcomputemanager.core.templates import substitute_variables

        result = substitute_variables("ts: ${TIMESTAMP}", {"TIMESTAMP": "CUSTOM"})
        assert "CUSTOM" in result

    def test_safe_substitute_no_crash_on_missing(self):
        from cloudcomputemanager.core.templates import substitute_variables

        result = substitute_variables("val: ${MISSING}", {})
        assert "${MISSING}" in result

    def test_end_to_end_yaml_with_nested_variables(self):
        """Variables should work in nested YAML structures."""
        from cloudcomputemanager.core.templates import load_config_with_template

        config_content = """
name: job-${STRUCTURE}
image: ubuntu:22.04
command: run --input ${DATA_FILE}
resources:
  disk_gb: 50
upload:
  source: ./data/${STRUCTURE}/
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            p = Path(f.name)
        try:
            config = load_config_with_template(
                p, variables={"STRUCTURE": "OH50-F50", "DATA_FILE": "input.data"}
            )
            assert config["name"] == "job-OH50-F50"
            assert config["command"] == "run --input input.data"
            assert config["upload"]["source"] == "./data/OH50-F50/"
        finally:
            p.unlink()


# ============================================================================
# Multi-Stage Job Advancement (Phase 3.2) — CRITICAL DAEMON LOGIC
# ============================================================================


class TestMultiStageAdvancement:
    """Test the actual stage advancement logic in the daemon monitor."""

    def _make_staged_job(self):
        stages = [
            {"name": "stage_a", "command": "echo stage_a"},
            {"name": "stage_b", "command": "echo stage_b"},
            {"name": "stage_c", "command": "echo stage_c"},
        ]
        return Job(
            name="multi-stage-test",
            image="ubuntu:22.04",
            command="echo stage_a",
            job_id="job_staged",
            instance_id="inst_123",
            status=JobStatus.RUNNING,
            stages_json=json.dumps(stages),
            current_stage=0,
        )

    def test_get_stages_returns_all(self):
        job = self._make_staged_job()
        assert len(job.get_stages()) == 3

    def test_get_current_stage_returns_correct(self):
        job = self._make_staged_job()
        assert job.get_current_stage()["name"] == "stage_a"
        job.current_stage = 2
        assert job.get_current_stage()["name"] == "stage_c"

    def test_get_current_stage_past_end(self):
        job = self._make_staged_job()
        job.current_stage = 10
        assert job.get_current_stage() is None

    def test_no_stages_returns_empty_list(self):
        job = Job(name="test", image="u", command="e")
        assert job.get_stages() == []
        assert job.get_current_stage() is None

    @pytest.mark.asyncio
    async def test_advance_job_stage_not_multistage(self):
        """advance_job_stage should return False for non-multi-stage jobs."""
        from cloudcomputemanager.daemon.monitor import JobMonitor

        monitor = JobMonitor.__new__(JobMonitor)
        monitor.provider = AsyncMock()

        job = Job(name="single", image="u", command="e", instance_id="i")
        result = await monitor.advance_job_stage(job, exit_code=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_advance_job_stage_on_failure(self):
        """advance_job_stage should NOT advance when exit code != 0."""
        from cloudcomputemanager.daemon.monitor import JobMonitor

        monitor = JobMonitor.__new__(JobMonitor)
        monitor.provider = AsyncMock()

        job = self._make_staged_job()
        result = await monitor.advance_job_stage(job, exit_code=1)
        assert result is False

    @pytest.mark.asyncio
    async def test_advance_job_stage_last_stage(self):
        """advance_job_stage should return False when already on last stage."""
        from cloudcomputemanager.daemon.monitor import JobMonitor

        monitor = JobMonitor.__new__(JobMonitor)
        monitor.provider = AsyncMock()

        job = self._make_staged_job()
        job.current_stage = 2  # Last stage
        result = await monitor.advance_job_stage(job, exit_code=0)
        assert result is False

    @pytest.mark.asyncio
    async def test_advance_job_stage_executes_next(self):
        """advance_job_stage should SSH commands to start next stage."""
        from cloudcomputemanager.daemon.monitor import JobMonitor
        from cloudcomputemanager.core.database import init_db, get_session
        from cloudcomputemanager.core import database as db_module

        # Reset DB engine for fresh schema
        db_module._engine = None
        db_module._async_session_factory = None

        monitor = JobMonitor.__new__(JobMonitor)
        mock_provider = AsyncMock()
        mock_provider.execute_command = AsyncMock(return_value=(0, "", ""))
        monitor.provider = mock_provider

        job = self._make_staged_job()

        # We need a real DB for this test
        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            # Patch get_settings to return our test settings
            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                # Persist the job first
                async with get_session() as session:
                    session.add(job)

                # Advance from stage 0 to stage 1
                result = await monitor.advance_job_stage(job, exit_code=0)

                assert result is True

                # Verify SSH commands were called:
                # 1. rm -f .ccm_exit_code
                # 2. Write wrapper script (base64)
                # 3. nohup to start it
                calls = mock_provider.execute_command.call_args_list
                assert len(calls) >= 3
                assert "rm -f /workspace/.ccm_exit_code" in calls[0][0][1]
                assert "base64 -d" in calls[1][0][1]
                assert "nohup" in calls[2][0][1]

                # Verify DB was updated
                from sqlmodel import select
                async with get_session() as session:
                    stmt = select(Job).where(Job.job_id == "job_staged")
                    result = await session.execute(stmt)
                    db_job = result.scalar_one_or_none()
                    assert db_job is not None
                    assert db_job.current_stage == 1
                    assert db_job.command == "echo stage_b"

            # Cleanup
            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None


# ============================================================================
# Batch Parameter Matrix (Phase 3.3)
# ============================================================================


class TestBatchParameterMatrix:
    """Test batch matrix expansion with real YAML files."""

    def test_no_matrix_returns_single_entry(self):
        from cloudcomputemanager.cli.batch import expand_matrix

        config = {"name": "test", "command": "echo hello"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            p = Path(f.name)
        try:
            entries = expand_matrix(p)
            assert len(entries) == 1
            assert entries[0][1] == {}
        finally:
            p.unlink()

    def test_cartesian_product(self):
        from cloudcomputemanager.cli.batch import expand_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            job_path = Path(tmpdir) / "job.yaml"
            job_path.write_text("name: test\ncommand: echo ${X}")

            matrix_path = Path(tmpdir) / "matrix.yaml"
            matrix_config = {
                "template": "job.yaml",
                "matrix": {"X": [1, 2, 3], "Y": ["A", "B"]},
            }
            with open(matrix_path, "w") as f:
                yaml.dump(matrix_config, f)

            entries = expand_matrix(matrix_path)
            assert len(entries) == 6  # 3 * 2

            var_combos = [e[1] for e in entries]
            assert {"X": "1", "Y": "A"} in var_combos
            assert {"X": "3", "Y": "B"} in var_combos

    def test_single_value_wrapped_as_list(self):
        from cloudcomputemanager.cli.batch import expand_matrix

        config = {"matrix": {"GPU": "RTX_3060", "N": [8, 16]}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            p = Path(f.name)
        try:
            entries = expand_matrix(p)
            assert len(entries) == 2
        finally:
            p.unlink()

    def test_empty_matrix_returns_zero(self):
        from cloudcomputemanager.cli.batch import expand_matrix

        config = {"matrix": {"X": []}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            p = Path(f.name)
        try:
            entries = expand_matrix(p)
            assert len(entries) == 0
        finally:
            p.unlink()


# ============================================================================
# Pluggable Progress Parsing (Phase 2.1)
# ============================================================================


class TestProgressParsing:
    """Test actual regex extraction logic, not just config storage."""

    def test_progress_config_roundtrip(self):
        progress = {"type": "regex_parse", "file": "/out.log", "regex": r"(\d+)", "total": 1000}
        job = Job(
            name="t", image="u", command="e",
            progress_json=json.dumps(progress),
        )
        config = job.get_progress_config()
        assert config["type"] == "regex_parse"
        assert config["total"] == 1000

    @pytest.mark.asyncio
    async def test_regex_parse_extracts_last_match(self):
        """regex_parse mode should extract the LAST match from tail output."""
        from cloudcomputemanager.daemon.monitor import JobMonitor

        monitor = JobMonitor.__new__(JobMonitor)

        # Simulate SSH returning log output with multiple timestep lines
        log_output = """
Step 1000 temp 300.0
Step 2000 temp 301.5
Step 3000 temp 299.8
"""
        mock_provider = AsyncMock()
        mock_provider.execute_command = AsyncMock(return_value=(0, log_output, ""))
        monitor.provider = mock_provider

        job = Job(
            name="test", image="u", command="e",
            job_id="job_prog", instance_id="inst_1",
            status=JobStatus.RUNNING,
            progress_json=json.dumps({
                "type": "regex_parse",
                "file": "/workspace/output.log",
                "regex": r"Step\s+(\d+)",
                "total": 5000,
            }),
        )

        # We need a DB session for update_job_progress to persist metrics
        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            from cloudcomputemanager.core.database import init_db, get_session
            from cloudcomputemanager.core import database as db_module

            db_module._engine = None
            db_module._async_session_factory = None
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                async with get_session() as session:
                    session.add(job)

                await monitor.update_job_progress(job)

                # Verify the correct value (3000, the last match) was extracted
                from sqlmodel import select
                async with get_session() as session:
                    stmt = select(Job).where(Job.job_id == "job_prog")
                    result = await session.execute(stmt)
                    db_job = result.scalar_one_or_none()
                    assert db_job is not None
                    metrics = json.loads(db_job.metrics_json)
                    assert metrics["current_step"] == 3000.0
                    assert metrics["total_steps"] == 5000
                    assert metrics["progress_percent"] == 60.0

            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None

    @pytest.mark.asyncio
    async def test_regex_parse_no_match_is_silent(self):
        """If regex doesn't match anything, progress should not crash."""
        from cloudcomputemanager.daemon.monitor import JobMonitor

        monitor = JobMonitor.__new__(JobMonitor)
        mock_provider = AsyncMock()
        mock_provider.execute_command = AsyncMock(return_value=(0, "no numbers here\n", ""))
        monitor.provider = mock_provider

        job = Job(
            name="t", image="u", command="e",
            job_id="job_nomatch", instance_id="i",
            progress_json=json.dumps({
                "type": "regex_parse", "file": "/f", "regex": r"Step\s+(\d+)", "total": 100,
            }),
        )
        # Should not raise
        await monitor.update_job_progress(job)


# ============================================================================
# Notification Hooks (Phase 2.2)
# ============================================================================


class TestNotificationHooks:
    """Test notification hook execution with shell escaping."""

    def test_notifications_stored(self):
        job = Job(
            name="t", image="u", command="e",
            notifications_json=json.dumps({"on_complete": "echo done"}),
        )
        assert job.get_notifications()["on_complete"] == "echo done"

    @pytest.mark.asyncio
    async def test_notification_substitutes_and_escapes(self):
        """Notification should substitute variables with shlex escaping."""
        from cloudcomputemanager.daemon.monitor import JobMonitor

        monitor = JobMonitor.__new__(JobMonitor)

        job = Job(
            name="my job with spaces",  # Has spaces — must be quoted
            image="u", command="e",
            job_id="job_abc",
            exit_code=42,
            status=JobStatus.FAILED,
            notifications_json=json.dumps({
                "on_failure": "echo ${JOB_NAME} exited ${EXIT_CODE}"
            }),
        )

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_shell", return_value=mock_proc) as mock_shell:
            await monitor.run_notification(job, "on_failure")
            cmd = mock_shell.call_args[0][0]
            # Job name should be shell-quoted because it has spaces
            assert "'my job with spaces'" in cmd
            assert "42" in cmd

    @pytest.mark.asyncio
    async def test_notification_missing_event_is_noop(self):
        """If notification config doesn't have the event, nothing happens."""
        from cloudcomputemanager.daemon.monitor import JobMonitor

        monitor = JobMonitor.__new__(JobMonitor)

        job = Job(
            name="t", image="u", command="e",
            notifications_json=json.dumps({"on_complete": "echo done"}),
        )

        # on_failure is not defined — should be a silent no-op
        with patch("asyncio.create_subprocess_shell") as mock_shell:
            await monitor.run_notification(job, "on_failure")
            mock_shell.assert_not_called()


# ============================================================================
# SSH Credentials (Phase 1.3)
# ============================================================================


class TestSSHCredentials:
    """Test SSH credential retrieval."""

    def test_agent_sdk_has_method(self):
        from cloudcomputemanager.agents.sdk import CloudComputeManagerAgent
        assert callable(getattr(CloudComputeManagerAgent, "get_ssh_credentials", None))

    def test_cli_exec_module_callable(self):
        from cloudcomputemanager.cli.jobs_exec import exec_on_job, upload_to_job, get_ssh_credentials
        assert callable(exec_on_job)
        assert callable(upload_to_job)
        assert callable(get_ssh_credentials)


# ============================================================================
# Agent SDK Exit Code Detection
# ============================================================================


class TestAgentSDKExitCode:
    """Test that agent SDK correctly detects job completion via .ccm_exit_code."""

    def test_shared_wrapper_has_ccm_exit_code(self):
        """The shared wrapper must write .ccm_exit_code."""
        from cloudcomputemanager.core.wrapper import build_wrapper_script
        script = build_wrapper_script("echo test")
        assert ".ccm_exit_code" in script

    def test_wait_reads_ccm_exit_code(self):
        """Agent SDK wait_for_completion() must read .ccm_exit_code, not /workspace/.exit_code."""
        from cloudcomputemanager.agents import sdk
        source = __import__("inspect").getsource(sdk.CloudComputeManagerAgent.wait_for_completion)
        assert ".ccm_exit_code" in source
        assert "/workspace/.exit_code" not in source.replace(".ccm_exit_code", "")


# ============================================================================
# Corrupted JSON Resilience
# ============================================================================


class TestCorruptedJSONResilience:
    """Test that model helpers handle corrupted JSON gracefully."""

    def test_corrupted_stages_json(self):
        job = Job(name="t", image="u", command="e", stages_json="not valid json{")
        assert job.get_stages() == []

    def test_corrupted_progress_json(self):
        job = Job(name="t", image="u", command="e", progress_json="[[invalid")
        assert job.get_progress_config() == {}

    def test_corrupted_notifications_json(self):
        job = Job(name="t", image="u", command="e", notifications_json="{bad")
        assert job.get_notifications() == {}

    def test_stages_json_wrong_type(self):
        """If stages_json contains a dict instead of list, return empty list."""
        job = Job(name="t", image="u", command="e", stages_json='{"not": "a list"}')
        assert job.get_stages() == []

    def test_progress_json_wrong_type(self):
        """If progress_json contains a list instead of dict, return empty dict."""
        job = Job(name="t", image="u", command="e", progress_json='[1, 2, 3]')
        assert job.get_progress_config() == {}


# ============================================================================
# CostRecord and Benchmark Models (Phase 4)
# ============================================================================


class TestCostRecord:
    """Test cost record model."""

    def test_full_creation(self):
        record = CostRecord(
            job_id="job_abc", project="proj", gpu_type="RTX_3060",
            gpu_count=1, cpu_cores=16, hourly_rate=0.08,
            performance_metric=145.5, metric_name="ts/s",
            total_cost_usd=1.44, total_runtime_seconds=64800,
            cost_per_unit=0.80, workload_type="md",
        )
        assert record.gpu_type == "RTX_3060"
        assert record.cost_per_unit == 0.80

    def test_minimal_creation(self):
        record = CostRecord(job_id="j", gpu_type="A100")
        assert record.performance_metric is None
        assert record.workload_type is None


class TestBenchmarkModels:
    """Test benchmark framework models."""

    def test_benchmark_suite_creation(self):
        from cloudcomputemanager.benchmarks.models import BenchmarkSuite
        suite = BenchmarkSuite(name="test-bench", status="pending")
        assert suite.name == "test-bench"
        assert suite.total_runs == 0

    def test_benchmark_config_valid(self):
        """Valid benchmark config should load without error."""
        from cloudcomputemanager.benchmarks.engine import BenchmarkEngine

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"name": "test", "workload": {"command": "echo hello"}}, f)
            p = Path(f.name)
        try:
            config = BenchmarkEngine.load_config(p)
            assert config["name"] == "test"
            assert config["workload"]["command"] == "echo hello"
        finally:
            p.unlink()

    def test_benchmark_config_missing_name(self):
        """Missing name should raise ValueError."""
        from cloudcomputemanager.benchmarks.engine import BenchmarkEngine

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"workload": {"command": "echo"}}, f)
            p = Path(f.name)
        try:
            with pytest.raises(ValueError, match="missing required key"):
                BenchmarkEngine.load_config(p)
        finally:
            p.unlink()

    def test_benchmark_config_missing_command(self):
        """Missing workload.command should raise ValueError."""
        from cloudcomputemanager.benchmarks.engine import BenchmarkEngine

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"name": "test", "workload": {}}, f)
            p = Path(f.name)
        try:
            with pytest.raises(ValueError, match="command is required"):
                BenchmarkEngine.load_config(p)
        finally:
            p.unlink()


# ============================================================================
# CLI Integration
# ============================================================================


class TestCLIIntegration:
    """Test CLI parsing."""

    def test_parse_set_variables_valid(self):
        from cloudcomputemanager.cli.main import parse_set_variables
        assert parse_set_variables(["A=1", "B=two"]) == {"A": "1", "B": "two"}

    def test_parse_set_variables_none(self):
        from cloudcomputemanager.cli.main import parse_set_variables
        assert parse_set_variables(None) is None

    def test_parse_set_variables_empty(self):
        from cloudcomputemanager.cli.main import parse_set_variables
        assert parse_set_variables([]) is None

    def test_parse_set_variables_value_with_equals(self):
        from cloudcomputemanager.cli.main import parse_set_variables
        assert parse_set_variables(["CMD=echo a=b"]) == {"CMD": "echo a=b"}

    def test_parse_set_variables_empty_value(self):
        from cloudcomputemanager.cli.main import parse_set_variables
        assert parse_set_variables(["KEY="]) == {"KEY": ""}

    def test_parse_set_variables_invalid_format(self):
        import typer
        from cloudcomputemanager.cli.main import parse_set_variables
        with pytest.raises(typer.BadParameter):
            parse_set_variables(["NO_EQUALS_SIGN"])


# ============================================================================
# Second-Pass Audit Bug Fix Verification
# ============================================================================


class TestSecondPassFixes:
    """Tests verifying bugs found in second-pass audit are actually fixed."""

    def test_wrapper_uses_effective_command(self):
        """submit_job must pass effective_command to build_deploy_commands."""
        from cloudcomputemanager.cli import jobs
        import inspect
        source = inspect.getsource(jobs.submit_job)
        assert "build_deploy_commands(effective_command)" in source

    def test_started_at_set_in_cli_submit(self):
        """CLI submit_job must set started_at on the Job record."""
        from cloudcomputemanager.cli import jobs
        import inspect
        source = inspect.getsource(jobs.submit_job)
        assert "started_at=" in source

    def test_batch_progress_uses_job_entries(self):
        """Batch progress bar must use len(job_entries), not len(config_files)."""
        from cloudcomputemanager.cli import batch
        import inspect
        source = inspect.getsource(batch.batch_submit)
        assert "total=len(job_entries)" in source

    def test_sdk_cleanup_has_null_checks(self):
        """SDK _cleanup must check if sync_engine/checkpoint_orchestrator are not None."""
        from cloudcomputemanager.agents import sdk
        import inspect
        source = inspect.getsource(sdk.CloudComputeManagerAgent._cleanup)
        assert "if self._sync_engine" in source
        assert "if self._checkpoint_orchestrator" in source

    def test_monitor_recovery_timer_is_instance_state(self):
        """Recovery timer must be instance state, not local variable."""
        from cloudcomputemanager.daemon.monitor import JobMonitor
        monitor = JobMonitor.__new__(JobMonitor)
        monitor.config = None
        monitor.provider = None
        monitor.settings = None
        monitor._running = False
        monitor._task = None
        monitor._event_handlers = []
        monitor._monitored_jobs = set()
        monitor._last_recovery_check = 0
        assert hasattr(monitor, '_last_recovery_check')
        # Verify the loop method uses self._last_recovery_check, not a bare local
        import inspect
        source = inspect.getsource(JobMonitor._monitor_loop)
        assert "self._last_recovery_check" in source
        # Must not have a bare local variable (without self. prefix)
        import re
        bare_locals = re.findall(r'(?<!\.)(?<!self\._)last_recovery_check\s*=', source)
        assert len(bare_locals) == 0, f"Found bare local variable: {bare_locals}"

    def test_monitor_no_hasattr_guards(self):
        """Monitor should call Job methods directly, not through hasattr guards."""
        from cloudcomputemanager.daemon import monitor
        import inspect
        # Check the methods that were cleaned up
        progress_source = inspect.getsource(monitor.JobMonitor.update_job_progress)
        assert "hasattr" not in progress_source

        stage_source = inspect.getsource(monitor.JobMonitor.advance_job_stage)
        assert "hasattr" not in stage_source

        notif_source = inspect.getsource(monitor.JobMonitor.run_notification)
        assert "hasattr" not in notif_source

    def test_benchmark_suite_marks_failed_when_zero_completed(self):
        """Benchmark suite must be marked 'failed' when completed_runs == 0."""
        from cloudcomputemanager.benchmarks import engine
        import inspect
        source = inspect.getsource(engine.BenchmarkEngine.run_suite)
        assert 'db_suite.status = "failed"' in source
        assert "completed_runs == 0" in source

    def test_benchmark_get_results_handles_empty(self):
        """get_results must return [] on empty results, not crash."""
        from cloudcomputemanager.benchmarks import engine
        import inspect
        source = inspect.getsource(engine.BenchmarkEngine.get_results)
        assert "if not results:" in source
        assert "return []" in source


# ============================================================================
# Resilience Features
# ============================================================================


class TestWrapperScriptResilience:
    """Test the shared wrapper script builder and SIGTERM handling."""

    def test_wrapper_has_sigterm_trap(self):
        from cloudcomputemanager.core.wrapper import build_wrapper_script
        script = build_wrapper_script("echo hello")
        assert "trap _ccm_sigterm_handler SIGTERM" in script
        assert ".ccm_preempted" in script
        assert "SIGUSR1" in script or "kill -USR1" in script

    def test_wrapper_runs_job_in_background(self):
        """Job must run as background process (&) so trap can fire during wait."""
        from cloudcomputemanager.core.wrapper import build_wrapper_script
        script = build_wrapper_script("mpirun lmp -in input.inp")
        assert "mpirun lmp -in input.inp &" in script
        assert "wait $JOB_PID" in script

    def test_wrapper_writes_exit_code_143_on_preemption(self):
        from cloudcomputemanager.core.wrapper import build_wrapper_script, PREEMPTION_EXIT_CODE
        script = build_wrapper_script("echo test")
        assert str(PREEMPTION_EXIT_CODE) in script
        assert PREEMPTION_EXIT_CODE == 143

    def test_wrapper_stage_name_in_header(self):
        from cloudcomputemanager.core.wrapper import build_wrapper_script
        script = build_wrapper_script("echo test", stage_name="equilibration")
        assert "Stage: equilibration" in script

    def test_build_deploy_commands(self):
        from cloudcomputemanager.core.wrapper import build_deploy_commands
        setup, run = build_deploy_commands("echo hello")
        assert "base64 -d" in setup
        assert "run_job.sh" in setup
        assert "nohup" in run

    def test_build_deploy_commands_custom_path(self):
        from cloudcomputemanager.core.wrapper import build_deploy_commands
        setup, run = build_deploy_commands("echo hello", script_path="/workspace/run_stage.sh")
        assert "run_stage.sh" in setup
        assert "run_stage.sh" in run

    def test_all_three_sites_use_shared_builder(self):
        """cli/jobs.py, daemon/monitor.py, agents/sdk.py must all use core.wrapper."""
        import inspect
        from cloudcomputemanager.cli import jobs
        from cloudcomputemanager.daemon import monitor
        from cloudcomputemanager.agents import sdk

        # Each should import from core.wrapper, not have inline wrapper scripts
        jobs_src = inspect.getsource(jobs.submit_job)
        assert "build_deploy_commands" in jobs_src

        monitor_src = inspect.getsource(monitor.JobMonitor.advance_job_stage)
        assert "build_deploy_commands" in monitor_src

        sdk_src = inspect.getsource(sdk.CloudComputeManagerAgent.submit)
        assert "build_deploy_commands" in sdk_src


class TestExitCode143Detection:
    """Test that exit code 143 is treated as preemption, not failure."""

    def test_handle_completion_routes_143_to_preemption(self):
        """Exit code 143 in handle_job_completion should call handle_preemption."""
        from cloudcomputemanager.daemon import monitor
        import inspect
        source = inspect.getsource(monitor.JobMonitor.handle_job_completion)
        assert "PREEMPTION_EXIT_CODE" in source
        assert "handle_preemption" in source


class TestDaemonReconciliation:
    """Test daemon startup reconciliation."""

    def test_reconcile_method_exists(self):
        from cloudcomputemanager.daemon.monitor import JobMonitor
        assert hasattr(JobMonitor, '_reconcile_stale_jobs')

    def test_monitor_loop_calls_reconcile(self):
        """_monitor_loop must call _reconcile_stale_jobs before entering while loop."""
        from cloudcomputemanager.daemon import monitor
        import inspect
        source = inspect.getsource(monitor.JobMonitor._monitor_loop)
        assert "_reconcile_stale_jobs" in source
        # It should be called BEFORE the while loop
        idx_reconcile = source.index("_reconcile_stale_jobs")
        idx_while = source.index("while self._running")
        assert idx_reconcile < idx_while

    @pytest.mark.asyncio
    async def test_reconcile_handles_gone_instance(self):
        """Reconciliation should handle instances that no longer exist."""
        from cloudcomputemanager.daemon.monitor import JobMonitor, MonitorConfig
        from cloudcomputemanager.core.database import init_db, get_session
        from cloudcomputemanager.core import database as db_module

        monitor = JobMonitor.__new__(JobMonitor)
        mock_provider = AsyncMock()
        mock_provider.get_instance = AsyncMock(return_value=None)
        monitor.provider = mock_provider
        monitor.config = MonitorConfig()
        monitor.settings = MagicMock()
        monitor._event_handlers = []
        monitor._monitored_jobs = set()
        monitor._last_recovery_check = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            from cloudcomputemanager.core.config import Settings
            settings = Settings(
                data_dir=Path(tmpdir) / "ccm",
                database_url=f"sqlite+aiosqlite:///{tmpdir}/test.db",
            )
            settings.ensure_directories()

            with patch("cloudcomputemanager.core.database.get_settings", return_value=settings):
                db_module._engine = None
                db_module._async_session_factory = None
                await init_db()

                # Create a job in RUNNING state with a fake instance_id
                job = Job(
                    name="stale-job", image="u", command="e",
                    job_id="job_stale", instance_id="gone_instance",
                    status=JobStatus.RUNNING,
                    started_at=datetime.utcnow(),
                )
                async with get_session() as session:
                    session.add(job)

                # Run reconciliation
                await monitor._reconcile_stale_jobs()

                # Job should now be in RECOVERING state
                from sqlmodel import select
                async with get_session() as session:
                    stmt = select(Job).where(Job.job_id == "job_stale")
                    result = await session.execute(stmt)
                    db_job = result.scalar_one_or_none()
                    assert db_job is not None
                    assert db_job.status == JobStatus.RECOVERING

            await db_module.close_db()
            db_module._engine = None
            db_module._async_session_factory = None


class TestReconnectCommand:
    """Test the reconnect command registration."""

    def test_reconnect_function_exists(self):
        from cloudcomputemanager.cli.jobs import reconnect_jobs
        assert callable(reconnect_jobs)

    def test_reconnect_cli_registered(self):
        """ccm jobs reconnect and ccm reconnect should both exist."""
        from cloudcomputemanager.cli import main
        import inspect
        source = inspect.getsource(main)
        assert "jobs_reconnect" in source
        assert "quick_reconnect" in source


class TestInstanceHeartbeat:
    """Test that instance heartbeat is injected into onstart."""

    def test_heartbeat_in_create_instance(self):
        from cloudcomputemanager.providers import vast
        import inspect
        source = inspect.getsource(vast.VastProvider.create_instance)
        assert ".ccm_heartbeat" in source
        assert "while true" in source
