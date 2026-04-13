"""Tests for restart adapters.

Tests detection, checkpoint file handling, and restart command generation
for all application-specific restart adapters.
"""

import pytest
from pathlib import Path

from cloudcomputemanager.checkpoint.restart_adapters import (
    RestartResult,
    RestartAdapter,
    NAMDRestartAdapter,
    GROMACSRestartAdapter,
    LAMMPSRestartAdapter,
    QuantumEspressoRestartAdapter,
    VASPRestartAdapter,
    PyTorchLightningRestartAdapter,
    HFTrainerRestartAdapter,
    GenericRestartAdapter,
    get_restart_adapter,
    RESTART_ADAPTERS,
)


# =============================================================================
# RestartResult
# =============================================================================


class TestRestartResult:
    def test_basic_result(self):
        r = RestartResult(command="echo hello", description="test")
        assert r.command == "echo hello"
        assert r.description == "test"
        assert r.files_written == []

    def test_with_files(self):
        r = RestartResult(
            command="run.sh",
            files_written=["/tmp/restart.conf"],
            description="restart",
        )
        assert r.files_written == ["/tmp/restart.conf"]


# =============================================================================
# NAMDRestartAdapter
# =============================================================================


class TestNAMDRestartAdapter:
    def test_detect_namd3(self):
        a = NAMDRestartAdapter()
        assert a.detect("namd3 +p8 +devices 0 config.namd")
        assert a.detect("cd /workspace && namd3 +p8 +devices 0 production.namd > job.log 2>&1")

    def test_detect_namd2(self):
        a = NAMDRestartAdapter()
        assert a.detect("namd2 +p4 config.namd")

    def test_detect_namd_word_boundary(self):
        a = NAMDRestartAdapter()
        assert a.detect("namd +p8 config.namd")
        # Should NOT match substrings like "dynamics" containing "namd"
        # The regex uses word boundary \bnamd\b
        assert not a.detect("some-dynamics-tool --flag")

    def test_detect_case_insensitive(self):
        a = NAMDRestartAdapter()
        assert a.detect("NAMD3 +p8 config.namd")

    def test_detect_negative(self):
        a = NAMDRestartAdapter()
        assert not a.detect("python train.py")
        assert not a.detect("gmx mdrun -deffnm prod")
        assert not a.detect("lmp -in input.in")

    def test_name(self):
        assert NAMDRestartAdapter().name == "namd"

    def test_prepare_restart_no_files(self, tmp_path):
        a = NAMDRestartAdapter()
        result = a.prepare_restart("namd3 +p8 config.namd", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_partial_files(self, tmp_path):
        """Only some restart files present — should return None."""
        (tmp_path / "simulation.restart.xsc").write_text(
            "# NAMD xsc\n#$LABELS step\n500000 72.95 0 0 0 72.95 0 0 0 72.95 36 36 36\n"
        )
        (tmp_path / "simulation.restart.coor").write_bytes(b"\x00" * 100)
        # Missing .vel
        a = NAMDRestartAdapter()
        result = a.prepare_restart("namd3 +p8 config.namd", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_with_files(self, tmp_path):
        """All restart files present — should generate config."""
        (tmp_path / "simulation.restart.xsc").write_text(
            "# NAMD xsc\n#$LABELS step\n5000000 72.95 0 0 0 72.95 0 0 0 72.95 36 36 36\n"
        )
        (tmp_path / "simulation.restart.coor").write_bytes(b"\x00" * 100)
        (tmp_path / "simulation.restart.vel").write_bytes(b"\x00" * 100)

        a = NAMDRestartAdapter()
        result = a.prepare_restart("namd3 +p8 config.namd", tmp_path, "job_1")

        assert result is not None
        assert "simulation_restart.namd" in result.command
        assert (tmp_path / "simulation_restart.namd").exists()
        assert "5000000" in result.description


# =============================================================================
# GROMACSRestartAdapter
# =============================================================================


class TestGROMACSRestartAdapter:
    def test_detect_gmx(self):
        a = GROMACSRestartAdapter()
        assert a.detect("gmx mdrun -deffnm production")
        assert a.detect("gmx_mpi mdrun -s topol.tpr")

    def test_detect_mdrun(self):
        a = GROMACSRestartAdapter()
        assert a.detect("mdrun -s topol.tpr -deffnm prod")

    def test_detect_gromacs(self):
        a = GROMACSRestartAdapter()
        assert a.detect("gromacs_wrapper run.sh")

    def test_detect_negative(self):
        a = GROMACSRestartAdapter()
        assert not a.detect("namd3 +p8 config.namd")
        assert not a.detect("python train.py")

    def test_name(self):
        assert GROMACSRestartAdapter().name == "gromacs"

    def test_prepare_restart_no_checkpoint(self, tmp_path):
        a = GROMACSRestartAdapter()
        result = a.prepare_restart("gmx mdrun -deffnm prod", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_injects_cpi(self, tmp_path):
        (tmp_path / "state.cpt").write_bytes(b"\x00" * 100)
        a = GROMACSRestartAdapter()
        result = a.prepare_restart("gmx mdrun -deffnm prod", tmp_path, "job_1")
        assert result is not None
        assert "-cpi state.cpt" in result.command
        assert "mdrun" in result.command

    def test_prepare_restart_already_has_cpi(self, tmp_path):
        (tmp_path / "state.cpt").write_bytes(b"\x00" * 100)
        a = GROMACSRestartAdapter()
        result = a.prepare_restart("gmx mdrun -cpi state.cpt -deffnm prod", tmp_path, "job_1")
        assert result is not None
        # Should NOT double-add -cpi
        assert result.command.count("-cpi") == 1

    def test_prepare_restart_uses_latest_cpt(self, tmp_path):
        """If multiple .cpt files, use the most recently modified."""
        import time
        (tmp_path / "state_prev.cpt").write_bytes(b"\x00" * 50)
        time.sleep(0.05)  # Ensure different mtime
        (tmp_path / "state.cpt").write_bytes(b"\x00" * 100)
        a = GROMACSRestartAdapter()
        result = a.prepare_restart("gmx mdrun -deffnm prod", tmp_path, "job_1")
        assert result is not None
        assert "state.cpt" in result.command


# =============================================================================
# LAMMPSRestartAdapter
# =============================================================================


class TestLAMMPSRestartAdapter:
    def test_detect_lmp(self):
        a = LAMMPSRestartAdapter()
        assert a.detect("lmp -in input.in -sf gpu -pk gpu 1")
        assert a.detect("lmp_mpi -in simulation.in")
        assert a.detect("lmp_gpu -in input.lammps")

    def test_detect_lammps(self):
        a = LAMMPSRestartAdapter()
        assert a.detect("lammps -in input.in")

    def test_detect_negative(self):
        a = LAMMPSRestartAdapter()
        assert not a.detect("gmx mdrun")
        assert not a.detect("python train.py")
        # "example" contains "lmp" but not as a word
        assert not a.detect("example_program --flag")

    def test_name(self):
        assert LAMMPSRestartAdapter().name == "lammps"

    def test_prepare_restart_no_files(self, tmp_path):
        a = LAMMPSRestartAdapter()
        result = a.prepare_restart("lmp -in input.in", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_with_restart_bin(self, tmp_path):
        (tmp_path / "restart.500000.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "restart.1000000.bin").write_bytes(b"\x00" * 100)
        a = LAMMPSRestartAdapter()
        result = a.prepare_restart("lmp -in input.in -sf gpu", tmp_path, "job_1")
        assert result is not None
        assert "ccm_restart_lammps.sh" in result.command
        assert (tmp_path / "ccm_restart_lammps.sh").exists()
        # Wrapper should contain original command
        wrapper = (tmp_path / "ccm_restart_lammps.sh").read_text()
        assert "lmp -in input.in -sf gpu" in wrapper

    def test_prepare_restart_with_dot_restart(self, tmp_path):
        (tmp_path / "system.restart").write_bytes(b"\x00" * 100)
        a = LAMMPSRestartAdapter()
        result = a.prepare_restart("lmp -in input.in", tmp_path, "job_1")
        assert result is not None


# =============================================================================
# QuantumEspressoRestartAdapter
# =============================================================================


class TestQuantumEspressoRestartAdapter:
    def test_detect_pwx(self):
        a = QuantumEspressoRestartAdapter()
        assert a.detect("pw.x -i scf.in")
        assert a.detect("mpirun -np 4 pw.x -input relax.in")

    def test_detect_phx(self):
        a = QuantumEspressoRestartAdapter()
        assert a.detect("ph.x -i ph.in")

    def test_detect_negative(self):
        a = QuantumEspressoRestartAdapter()
        assert not a.detect("python train.py")
        assert not a.detect("vasp_std")

    def test_name(self):
        assert QuantumEspressoRestartAdapter().name == "quantum_espresso"

    def test_prepare_restart_no_save_dir(self, tmp_path):
        a = QuantumEspressoRestartAdapter()
        result = a.prepare_restart("pw.x -i scf.in", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_rewrites_input(self, tmp_path):
        (tmp_path / "pwscf.save").mkdir()
        input_content = """&CONTROL
  calculation = 'relax',
  restart_mode = 'from_scratch',
  outdir = './tmp/',
/
"""
        (tmp_path / "relax.in").write_text(input_content)

        a = QuantumEspressoRestartAdapter()
        result = a.prepare_restart("pw.x -i relax.in", tmp_path, "job_1")
        assert result is not None
        modified = (tmp_path / "relax.in").read_text()
        assert "restart_mode = 'restart'" in modified
        assert "from_scratch" not in modified

    def test_prepare_restart_input_not_in_sync(self, tmp_path):
        """Save dir exists but input file not synced — still return result."""
        (tmp_path / "pwscf.save").mkdir()
        a = QuantumEspressoRestartAdapter()
        result = a.prepare_restart("pw.x -i scf.in", tmp_path, "job_1")
        assert result is not None
        assert "input not in sync" in result.description

    def test_prepare_restart_already_restart_mode(self, tmp_path):
        (tmp_path / "pwscf.save").mkdir()
        input_content = "  restart_mode = 'restart',\n"
        (tmp_path / "scf.in").write_text(input_content)

        a = QuantumEspressoRestartAdapter()
        result = a.prepare_restart("pw.x -i scf.in", tmp_path, "job_1")
        assert result is not None


# =============================================================================
# VASPRestartAdapter
# =============================================================================


class TestVASPRestartAdapter:
    def test_detect(self):
        a = VASPRestartAdapter()
        assert a.detect("vasp_std")
        assert a.detect("mpirun -np 4 vasp_gam")
        assert a.detect("vasp_ncl 2>&1 | tee output.log")

    def test_detect_negative(self):
        a = VASPRestartAdapter()
        assert not a.detect("python train.py")
        assert not a.detect("lmp -in input.in")

    def test_name(self):
        assert VASPRestartAdapter().name == "vasp"

    def test_prepare_restart_no_wavecar(self, tmp_path):
        a = VASPRestartAdapter()
        result = a.prepare_restart("vasp_std", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_empty_wavecar(self, tmp_path):
        (tmp_path / "WAVECAR").write_bytes(b"")
        a = VASPRestartAdapter()
        result = a.prepare_restart("vasp_std", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_wavecar_only(self, tmp_path):
        (tmp_path / "WAVECAR").write_bytes(b"\x00" * 1000)
        a = VASPRestartAdapter()
        result = a.prepare_restart("vasp_std", tmp_path, "job_1")
        assert result is not None
        assert result.command == "vasp_std"
        assert "WAVECAR" in result.description

    def test_prepare_restart_contcar_copied(self, tmp_path):
        (tmp_path / "WAVECAR").write_bytes(b"\x00" * 1000)
        (tmp_path / "CONTCAR").write_text("CONTCAR geometry\n")
        a = VASPRestartAdapter()
        result = a.prepare_restart("vasp_std", tmp_path, "job_1")
        assert result is not None
        assert (tmp_path / "POSCAR").exists()
        assert (tmp_path / "POSCAR").read_text() == "CONTCAR geometry\n"
        assert "CONTCAR" in result.description


# =============================================================================
# PyTorchLightningRestartAdapter
# =============================================================================


class TestPyTorchLightningRestartAdapter:
    def test_detect_lightning(self):
        a = PyTorchLightningRestartAdapter()
        assert a.detect("python -m lightning fit --config config.yaml")
        assert a.detect("python train.py --trainer.fit")

    def test_detect_pl_prefix(self):
        a = PyTorchLightningRestartAdapter()
        assert a.detect("python -m pl_bolts --config config.yaml")

    def test_detect_trainer_fit(self):
        a = PyTorchLightningRestartAdapter()
        assert a.detect("python script.py trainer.fit --accelerator gpu")

    def test_detect_negative(self):
        a = PyTorchLightningRestartAdapter()
        assert not a.detect("python train.py --epochs 10")
        assert not a.detect("lmp -in input.in")

    def test_name(self):
        assert PyTorchLightningRestartAdapter().name == "pytorch_lightning"

    def test_prepare_restart_no_checkpoints(self, tmp_path):
        a = PyTorchLightningRestartAdapter()
        result = a.prepare_restart("python -m lightning fit", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_appends_ckpt_path(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "epoch=2-step=1500.ckpt").write_bytes(b"\x00" * 100)
        a = PyTorchLightningRestartAdapter()
        result = a.prepare_restart("python -m lightning fit --config c.yaml", tmp_path, "job_1")
        assert result is not None
        assert "--ckpt_path last" in result.command

    def test_prepare_restart_already_has_ckpt_path(self, tmp_path):
        (tmp_path / "last.ckpt").write_bytes(b"\x00" * 100)
        a = PyTorchLightningRestartAdapter()
        result = a.prepare_restart(
            "python -m lightning fit --ckpt_path last", tmp_path, "job_1"
        )
        assert result is not None
        assert result.command.count("--ckpt_path") == 1


# =============================================================================
# HFTrainerRestartAdapter
# =============================================================================


class TestHFTrainerRestartAdapter:
    def test_detect_do_train(self):
        a = HFTrainerRestartAdapter()
        assert a.detect("python run_clm.py --do_train --model_name gpt2")

    def test_detect_run_scripts(self):
        a = HFTrainerRestartAdapter()
        assert a.detect("python run_mlm.py --model bert-base")
        assert a.detect("python run_glue.py --task sst2")

    def test_detect_negative(self):
        a = HFTrainerRestartAdapter()
        assert not a.detect("python train.py --epochs 10")
        assert not a.detect("lmp -in input.in")

    def test_name(self):
        assert HFTrainerRestartAdapter().name == "hf_trainer"

    def test_prepare_restart_no_checkpoints(self, tmp_path):
        a = HFTrainerRestartAdapter()
        result = a.prepare_restart("python run_clm.py --do_train", tmp_path, "job_1")
        assert result is None

    def test_prepare_restart_appends_flag(self, tmp_path):
        (tmp_path / "checkpoint-1000").mkdir()
        (tmp_path / "checkpoint-2000").mkdir()
        a = HFTrainerRestartAdapter()
        result = a.prepare_restart("python run_clm.py --do_train", tmp_path, "job_1")
        assert result is not None
        assert "--resume_from_checkpoint True" in result.command

    def test_prepare_restart_already_has_flag(self, tmp_path):
        (tmp_path / "checkpoint-1000").mkdir()
        a = HFTrainerRestartAdapter()
        result = a.prepare_restart(
            "python run_clm.py --do_train --resume_from_checkpoint True",
            tmp_path, "job_1",
        )
        assert result is not None
        assert result.command.count("--resume_from_checkpoint") == 1


# =============================================================================
# GenericRestartAdapter
# =============================================================================


class TestGenericRestartAdapter:
    def test_detect_always_true(self):
        a = GenericRestartAdapter()
        assert a.detect("literally anything")
        assert a.detect("")
        assert a.detect("python train.py")
        assert a.detect("./my-custom-simulator --config run.conf")

    def test_name(self):
        assert GenericRestartAdapter().name == "generic"

    def test_prepare_restart_always_none(self, tmp_path):
        a = GenericRestartAdapter()
        # Even with files present, returns None
        (tmp_path / "some_checkpoint.dat").write_bytes(b"\x00" * 100)
        result = a.prepare_restart("./custom-app", tmp_path, "job_1")
        assert result is None


# =============================================================================
# Adapter Chain (get_restart_adapter)
# =============================================================================


class TestAdapterChain:
    def test_namd_matched(self):
        adapter = get_restart_adapter("namd3 +p8 +devices 0 config.namd")
        assert adapter.name == "namd"

    def test_gromacs_matched(self):
        adapter = get_restart_adapter("gmx mdrun -deffnm production")
        assert adapter.name == "gromacs"

    def test_lammps_matched(self):
        adapter = get_restart_adapter("lmp -in input.in -sf gpu -pk gpu 1")
        assert adapter.name == "lammps"

    def test_qe_matched(self):
        adapter = get_restart_adapter("mpirun -np 4 pw.x -i scf.in")
        assert adapter.name == "quantum_espresso"

    def test_vasp_matched(self):
        adapter = get_restart_adapter("mpirun -np 4 vasp_std")
        assert adapter.name == "vasp"

    def test_lightning_matched(self):
        adapter = get_restart_adapter("python -m lightning fit --config c.yaml")
        assert adapter.name == "pytorch_lightning"

    def test_hf_trainer_matched(self):
        adapter = get_restart_adapter("python run_clm.py --do_train --model gpt2")
        assert adapter.name == "hf_trainer"

    def test_unknown_falls_to_generic(self):
        adapter = get_restart_adapter("./my-custom-binary --flag value")
        assert adapter.name == "generic"

    def test_python_script_falls_to_generic(self):
        adapter = get_restart_adapter("python train.py --epochs 100")
        assert adapter.name == "generic"

    def test_generic_is_last(self):
        assert RESTART_ADAPTERS[-1] is GenericRestartAdapter

    def test_all_adapters_have_unique_names(self):
        names = [cls().name for cls in RESTART_ADAPTERS]
        assert len(names) == len(set(names))

    def test_empty_command_falls_to_generic(self):
        adapter = get_restart_adapter("")
        assert adapter.name == "generic"


# =============================================================================
# Cross-Adapter Edge Cases
# =============================================================================


class TestEdgeCases:
    def test_command_with_multiple_tools(self):
        """A command that mentions multiple tools — first adapter wins."""
        # This has both "gmx" and "lmp" — GROMACS comes first
        adapter = get_restart_adapter("bash -c 'gmx mdrun && lmp -in input.in'")
        assert adapter.name == "gromacs"

    def test_case_insensitivity(self):
        """All adapters should handle case variations."""
        assert get_restart_adapter("NAMD3 +p8 config.namd").name == "namd"
        assert get_restart_adapter("GMX mdrun").name == "gromacs"
        assert get_restart_adapter("VASP_STD").name == "vasp"

    def test_adapter_returns_none_means_original_command(self, tmp_path):
        """When adapter returns None, recovery uses original command."""
        # NAMD adapter with no restart files
        adapter = NAMDRestartAdapter()
        result = adapter.prepare_restart("namd3 config.namd", tmp_path, "job_1")
        assert result is None
        # This signals: detected NAMD but no checkpoint — use original command

    def test_all_adapters_are_instantiable(self):
        """Smoke test: all adapters can be instantiated."""
        for cls in RESTART_ADAPTERS:
            adapter = cls()
            assert isinstance(adapter, RestartAdapter)
            assert isinstance(adapter.name, str)
            assert len(adapter.name) > 0
