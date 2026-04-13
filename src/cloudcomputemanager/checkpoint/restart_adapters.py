"""Restart adapters for different application types.

When a job is preempted and recovered onto a new instance, a RestartAdapter
determines how to resume the job from its checkpoint files. Adapters are
auto-detected from the job command string and tried in priority order.

The adapter chain:
    1. User-defined restart (from job YAML restart: section) — handled in recovery.py
    2. Auto-detected adapter (matched from command string) — this module
    3. Original command fallback (GenericRestartAdapter returns None)

Each adapter's prepare_restart() runs locally on the CCM host, reading/writing
only to the sync_dir (already rsynced from the old instance). No SSH, no async.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RestartResult:
    """Result of a restart adapter's prepare_restart() call."""

    command: str
    """The command to run on the recovered instance."""

    files_written: list[str] = field(default_factory=list)
    """Paths of files written to sync_dir (for logging)."""

    description: str = ""
    """Human-readable description for logs."""


class RestartAdapter(ABC):
    """Adapts recovery behavior for a specific application type.

    Subclasses detect their application from the command string and prepare
    a restart command using checkpoint files in the local sync directory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'namd', 'gromacs', 'lammps'."""
        ...

    @abstractmethod
    def detect(self, command: str) -> bool:
        """Return True if this adapter handles the given command string.

        Called with the job's command. First adapter that returns True wins.
        Must be fast (string matching only, no I/O).
        """
        ...

    @abstractmethod
    def prepare_restart(
        self,
        command: str,
        sync_dir: Path,
        job_id: str,
    ) -> Optional[RestartResult]:
        """Inspect sync_dir for checkpoint files and prepare restart.

        May write files to sync_dir (e.g., restart config).
        Returns RestartResult with the command to run, or None if
        no checkpoint files found (fall through to original command).

        This runs locally (on the CCM host), not on the instance.
        sync_dir has already been rsynced from the old instance.
        """
        ...


# =============================================================================
# Adapter Implementations
# =============================================================================


class NAMDRestartAdapter(RestartAdapter):
    """NAMD molecular dynamics — generates restart config from checkpoint.

    NAMD requires a new config file with binCoordinates/binVelocities/
    extendedSystem pointing to checkpoint files, plus firsttimestep set
    to the correct step number. Delegates to namd_restart.py for the
    actual config generation.
    """

    @property
    def name(self) -> str:
        return "namd"

    def detect(self, command: str) -> bool:
        cmd = command.lower()
        return "namd3" in cmd or "namd2" in cmd or re.search(r"\bnamd\b", cmd) is not None

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        # Find NAMD restart files by glob — NAMD's outputName can be anything
        # (simulation, npt_equil, prod_453K, etc.), and the restart files are
        # named <outputName>.restart.{coor,vel,xsc}. Prefer the matched triplet.
        xsc_files = sorted(sync_dir.glob("*.restart.xsc"))
        # Skip rotation .old files that NAMD writes as backups
        xsc_files = [f for f in xsc_files if not f.name.endswith(".xsc.old")]
        xsc = coor = vel = None
        for candidate in xsc_files:
            stem = candidate.name[:-len(".restart.xsc")]  # e.g. "npt_equil"
            c_coor = sync_dir / f"{stem}.restart.coor"
            c_vel = sync_dir / f"{stem}.restart.vel"
            if c_coor.exists() and c_vel.exists():
                xsc, coor, vel = candidate, c_coor, c_vel
                break

        if xsc is None:
            logger.debug("No NAMD restart file triplet in sync dir", job_id=job_id)
            return None

        try:
            from cloudcomputemanager.checkpoint.namd_restart import (
                generate_recovery_command,
                generate_restart_config,
            )

            result = generate_restart_config(restart_xsc=str(xsc))

            if result["remaining"] <= 0:
                logger.info("NAMD simulation already complete", job_id=job_id)
                return None

            if result["config"]:
                config_path = sync_dir / "simulation_restart.namd"
                config_path.write_text(result["config"])
                logger.info(
                    "Generated NAMD restart config",
                    job_id=job_id,
                    step=result["step_number"],
                    remaining=result["remaining"],
                )
                return RestartResult(
                    command=generate_recovery_command(has_restart_config=True),
                    files_written=[str(config_path)],
                    description=(
                        f"NAMD restart from step {result['step_number']}, "
                        f"{result['remaining']} steps remaining"
                    ),
                )
        except Exception as e:
            logger.warning(
                "Failed to generate NAMD restart config",
                job_id=job_id,
                error=str(e),
            )

        return None


class GROMACSRestartAdapter(RestartAdapter):
    """GROMACS molecular dynamics — injects -cpi flag for checkpoint restart.

    GROMACS is nearly self-healing: it auto-checkpoints every 15 minutes,
    handles SIGTERM gracefully, and resumes with `gmx mdrun -cpi state.cpt`.
    This adapter just ensures -cpi is present in the command.
    """

    @property
    def name(self) -> str:
        return "gromacs"

    def detect(self, command: str) -> bool:
        cmd = command.lower()
        return "gmx" in cmd or "mdrun" in cmd or "gromacs" in cmd

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        # Find checkpoint file
        cpt_files = sorted(sync_dir.glob("*.cpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cpt_files:
            return None

        cpt_name = cpt_files[0].name

        # If -cpi already in command, GROMACS will auto-detect
        if "-cpi" in command:
            return RestartResult(
                command=command,
                description=f"GROMACS restart ({cpt_name} present, -cpi already set)",
            )

        # Inject -cpi flag after mdrun
        if "mdrun" in command:
            restart_cmd = re.sub(
                r"(mdrun\b)", rf"\1 -cpi {cpt_name}", command, count=1,
            )
        else:
            restart_cmd = f"{command} -cpi {cpt_name}"

        return RestartResult(
            command=restart_cmd,
            description=f"GROMACS restart with checkpoint {cpt_name}",
        )


class LAMMPSRestartAdapter(RestartAdapter):
    """LAMMPS molecular dynamics — generates restart wrapper script.

    LAMMPS needs `read_restart` instead of `read_data` to resume. This
    adapter generates a wrapper shell script that detects the latest
    restart file and passes it as a LAMMPS variable, allowing input
    scripts that support `-var restart_file` to resume automatically.

    For input scripts that don't support this pattern, users can define
    a custom restart command in the job YAML.
    """

    @property
    def name(self) -> str:
        return "lammps"

    def detect(self, command: str) -> bool:
        cmd = command.lower()
        # Match lmp, lmp_mpi, lmp_gpu, lmp_serial, lammps, etc.
        # Use (?<!\w) instead of \b to allow lmp_mpi (underscore is a word char)
        return bool(re.search(r"(?<!\w)lmp(?:\b|_)", cmd) or re.search(r"\blammps\b", cmd))

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        # Find latest restart file
        restart_files = sorted(sync_dir.glob("restart.*.bin"), reverse=True)
        if not restart_files:
            restart_files = sorted(sync_dir.glob("*.restart"), reverse=True)
        if not restart_files:
            restart_files = sorted(sync_dir.glob("restart.bin"), reverse=True)
        if not restart_files:
            return None

        latest = restart_files[0].name

        # Generate a wrapper script that detects restart files at runtime.
        # This handles the case where the restart file name might differ
        # on the instance vs what we see in sync_dir.
        wrapper = f"""#!/bin/bash
# CCM LAMMPS restart wrapper — auto-generated
cd /workspace

# Find the most recent restart file
RESTART_FILE=$(ls -t restart.*.bin 2>/dev/null | head -1)
if [ -z "$RESTART_FILE" ]; then
    RESTART_FILE=$(ls -t *.restart 2>/dev/null | head -1)
fi
if [ -z "$RESTART_FILE" ]; then
    RESTART_FILE=$(ls -t restart.bin 2>/dev/null | head -1)
fi

if [ -n "$RESTART_FILE" ]; then
    echo "CCM: Found LAMMPS restart file: $RESTART_FILE"
    echo "CCM: Running with -var restart_file $RESTART_FILE"
    {command} -var restart_file "$RESTART_FILE"
else
    echo "CCM: No LAMMPS restart file found, running original command"
    {command}
fi
"""
        wrapper_path = sync_dir / "ccm_restart_lammps.sh"
        wrapper_path.write_text(wrapper)

        return RestartResult(
            command="bash /workspace/ccm_restart_lammps.sh",
            files_written=[str(wrapper_path)],
            description=f"LAMMPS restart from {latest}",
        )


class QuantumEspressoRestartAdapter(RestartAdapter):
    """Quantum ESPRESSO — rewrites restart_mode in input file.

    QE writes checkpoint data to outdir/prefix.save/ automatically.
    To resume, the input file needs `restart_mode = 'restart'` instead
    of `restart_mode = 'from_scratch'`. This adapter finds the input
    file from the command and modifies it in-place in sync_dir.
    """

    @property
    def name(self) -> str:
        return "quantum_espresso"

    def detect(self, command: str) -> bool:
        cmd = command.lower()
        return "pw.x" in cmd or "ph.x" in cmd or "cp.x" in cmd

    def _find_input_file(self, command: str, sync_dir: Path) -> Optional[Path]:
        """Extract input file path from QE command."""
        # QE uses -i or -input or < for input
        match = re.search(r"-(?:i|input)\s+(\S+)", command)
        if match:
            return sync_dir / Path(match.group(1)).name
        # Check for stdin redirect
        match = re.search(r"<\s*(\S+)", command)
        if match:
            return sync_dir / Path(match.group(1)).name
        return None

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        # Check for QE save directory (indicates prior run made progress)
        save_dirs = list(sync_dir.glob("*.save"))
        xml_files = list(sync_dir.glob("*.xml"))
        if not save_dirs and not xml_files:
            return None

        # Find and modify input file
        input_file = self._find_input_file(command, sync_dir)
        if input_file is None or not input_file.exists():
            logger.debug("QE input file not found in sync_dir", job_id=job_id)
            # Still return original command — QE save files are present,
            # and the input file on the instance may already have restart_mode
            return RestartResult(
                command=command,
                description="QE restart (save dir present, input not in sync)",
            )

        # Read and modify the input file
        content = input_file.read_text()
        modified = re.sub(
            r"restart_mode\s*=\s*['\"]from_scratch['\"]",
            "restart_mode = 'restart'",
            content,
            flags=re.IGNORECASE,
        )

        if modified != content:
            input_file.write_text(modified)
            return RestartResult(
                command=command,
                files_written=[str(input_file)],
                description="QE restart (restart_mode set to 'restart')",
            )

        # restart_mode might already be 'restart' or not present
        return RestartResult(
            command=command,
            description="QE restart (save dir present)",
        )


class VASPRestartAdapter(RestartAdapter):
    """VASP electronic structure — auto-detects WAVECAR, copies CONTCAR.

    VASP auto-reads WAVECAR if present (ISTART default). For ionic
    relaxation, CONTCAR (latest geometry) needs to be copied to POSCAR.
    """

    @property
    def name(self) -> str:
        return "vasp"

    def detect(self, command: str) -> bool:
        cmd = command.lower()
        return "vasp" in cmd

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        wavecar = sync_dir / "WAVECAR"
        contcar = sync_dir / "CONTCAR"

        if not wavecar.exists() or wavecar.stat().st_size == 0:
            return None

        files_written = []
        desc_parts = ["VASP restart (WAVECAR present)"]

        # For ionic relaxation: copy CONTCAR to POSCAR
        if contcar.exists() and contcar.stat().st_size > 0:
            poscar = sync_dir / "POSCAR"
            poscar.write_text(contcar.read_text())
            files_written.append(str(poscar))
            desc_parts.append("CONTCAR → POSCAR")

        return RestartResult(
            command=command,
            files_written=files_written,
            description=", ".join(desc_parts),
        )


class PyTorchLightningRestartAdapter(RestartAdapter):
    """PyTorch Lightning — appends --ckpt_path flag.

    Lightning auto-saves checkpoints and can resume with
    `trainer.fit(ckpt_path="last")`. When launched via CLI, this
    corresponds to `--ckpt_path last`.
    """

    @property
    def name(self) -> str:
        return "pytorch_lightning"

    def detect(self, command: str) -> bool:
        cmd = command.lower()
        return "lightning" in cmd or "pl_" in cmd or "trainer.fit" in cmd

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        # Look for Lightning checkpoint files
        ckpt_files = list(sync_dir.glob("**/*.ckpt"))
        if not ckpt_files:
            return None

        # If --ckpt_path already specified, don't double-add
        if "--ckpt_path" in command or "ckpt_path=" in command:
            return RestartResult(
                command=command,
                description=f"Lightning restart ({len(ckpt_files)} checkpoints, --ckpt_path already set)",
            )

        # Append --ckpt_path last
        restart_cmd = f"{command} --ckpt_path last"
        return RestartResult(
            command=restart_cmd,
            description=f"Lightning restart ({len(ckpt_files)} checkpoints, --ckpt_path last appended)",
        )


class HFTrainerRestartAdapter(RestartAdapter):
    """Hugging Face Transformers Trainer — appends --resume_from_checkpoint.

    The HF Trainer saves checkpoint-{step}/ directories and can resume
    with `--resume_from_checkpoint True` (auto-detects latest).
    """

    @property
    def name(self) -> str:
        return "hf_trainer"

    def detect(self, command: str) -> bool:
        cmd = command.lower()
        # Look for HF Trainer patterns — be specific to avoid false positives
        return "--do_train" in cmd or "run_clm" in cmd or "run_mlm" in cmd or "run_glue" in cmd

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        # Look for HF checkpoint directories
        ckpt_dirs = sorted(sync_dir.glob("checkpoint-*"), reverse=True)
        if not ckpt_dirs:
            # Also check output_dir subdirectory
            ckpt_dirs = sorted(sync_dir.glob("**/checkpoint-*"), reverse=True)
        if not ckpt_dirs:
            return None

        # If --resume_from_checkpoint already specified, don't double-add
        if "--resume_from_checkpoint" in command:
            return RestartResult(
                command=command,
                description=f"HF Trainer restart ({len(ckpt_dirs)} checkpoints, flag already set)",
            )

        # Append flag — "True" tells HF to auto-detect latest
        restart_cmd = f"{command} --resume_from_checkpoint True"
        return RestartResult(
            command=restart_cmd,
            description=f"HF Trainer restart ({len(ckpt_dirs)} checkpoints, auto-resume enabled)",
        )


class GenericRestartAdapter(RestartAdapter):
    """Fallback adapter — matches everything, does nothing.

    Returns None from prepare_restart(), signaling that recovery should
    use the original command. This is the industry-standard behavior
    (SkyPilot, Modal, Ray all do the same thing).

    The job's checkpoint files are still present in /workspace from the
    sync overlay — self-healing applications (GROMACS with -cpi already
    set, PyTorch scripts that check for checkpoints) will pick them up.
    """

    @property
    def name(self) -> str:
        return "generic"

    def detect(self, command: str) -> bool:
        return True

    def prepare_restart(
        self, command: str, sync_dir: Path, job_id: str,
    ) -> Optional[RestartResult]:
        return None


# =============================================================================
# Adapter Registry
# =============================================================================

# Ordered by specificity — first match wins. Generic is always last.
RESTART_ADAPTERS: list[type[RestartAdapter]] = [
    NAMDRestartAdapter,
    GROMACSRestartAdapter,
    LAMMPSRestartAdapter,
    QuantumEspressoRestartAdapter,
    VASPRestartAdapter,
    PyTorchLightningRestartAdapter,
    HFTrainerRestartAdapter,
    GenericRestartAdapter,
]


def get_restart_adapter(command: str) -> RestartAdapter:
    """Find the first adapter that matches the command.

    Returns an instantiated adapter. GenericRestartAdapter is guaranteed
    to match if nothing else does.
    """
    for adapter_cls in RESTART_ADAPTERS:
        adapter = adapter_cls()
        if adapter.detect(command):
            logger.debug("Restart adapter matched", adapter=adapter.name, command=command[:80])
            return adapter
    # Should never reach here — GenericRestartAdapter.detect() returns True
    return GenericRestartAdapter()
