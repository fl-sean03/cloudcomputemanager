"""Tests for checkpoint detection and orchestration."""

import pytest

from cloudcomputemanager.checkpoint.detectors import (
    LAMMPSDetector,
    FilePatternDetector,
    CheckpointInfo,
)


class TestLAMMPSDetector:
    """Tests for LAMMPS checkpoint detector."""

    @pytest.fixture
    def detector(self):
        return LAMMPSDetector()

    def test_detector_name(self, detector):
        """Test detector name."""
        assert detector.name == "lammps"

    @pytest.mark.asyncio
    async def test_find_checkpoints_empty(self, detector):
        """Test finding checkpoints when none exist."""

        async def execute_fn(cmd):
            return (1, "", "")

        checkpoints = await detector.find_checkpoints(execute_fn, "/workspace")
        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_find_checkpoints_with_files(self, detector):
        """Test finding checkpoints with restart files."""

        async def execute_fn(cmd):
            if "find" in cmd:
                return (0, "/workspace/restart.1000.bin\n/workspace/restart.2000.bin", "")
            elif "stat" in cmd:
                if "1000" in cmd:
                    return (0, "1024 1700000000", "")
                else:
                    return (0, "2048 1700001000", "")
            return (1, "", "")

        checkpoints = await detector.find_checkpoints(execute_fn, "/workspace")

        assert len(checkpoints) == 2
        # Should be sorted by iteration (descending)
        assert checkpoints[0].iteration == 2000
        assert checkpoints[1].iteration == 1000

    @pytest.mark.asyncio
    async def test_get_current_timestep(self, detector):
        """Test parsing current timestep from LAMMPS log."""

        async def execute_fn(cmd):
            if "grep" in cmd:
                return (0, "    50000", "")
            return (1, "", "")

        timestep = await detector.get_current_timestep(execute_fn)
        assert timestep == 50000

    @pytest.mark.asyncio
    async def test_get_progress(self, detector):
        """Test getting simulation progress."""

        async def execute_fn(cmd):
            if "grep" in cmd:
                return (0, "    250000", "")
            return (1, "", "")

        progress = await detector.get_progress(execute_fn, total_steps=1000000)

        assert progress is not None
        assert progress["current_step"] == 250000
        assert progress["percent_complete"] == 25.0


class TestFilePatternDetector:
    """Tests for file pattern checkpoint detector."""

    @pytest.fixture
    def detector(self):
        return FilePatternDetector(
            patterns=["*.pt", "*.pth", "checkpoint_*"],
            iteration_regex=r"(\d+)",
        )

    def test_detector_name(self, detector):
        """Test detector name."""
        assert detector.name == "file_pattern"

    @pytest.mark.asyncio
    async def test_find_pytorch_checkpoints(self, detector):
        """Test finding PyTorch checkpoint files."""

        async def execute_fn(cmd):
            if "find" in cmd:
                return (0, "/workspace/model_100.pt\n/workspace/model_200.pt", "")
            elif "stat" in cmd:
                if "100" in cmd:
                    return (0, "5000000 1700000000", "")
                else:
                    return (0, "5100000 1700001000", "")
            return (1, "", "")

        checkpoints = await detector.find_checkpoints(execute_fn, "/workspace")

        assert len(checkpoints) == 2
        assert checkpoints[0].iteration == 200
        assert checkpoints[1].iteration == 100
