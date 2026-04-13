#!/usr/bin/env python3
"""Validate restart adapters against real checkpoint files from completed jobs.

Run AFTER integration test jobs complete and their sync dirs are populated.

Usage:
    python validate_adapters.py [--sync-base ~/.cloudcomputemanager/sync]
"""

import sys
import argparse
from pathlib import Path

# Add CCM to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cloudcomputemanager.checkpoint.restart_adapters import (
    get_restart_adapter,
    RESTART_ADAPTERS,
)


def validate_lammps(sync_dir: Path) -> bool:
    """Validate LAMMPS adapter with real restart files."""
    print("\n=== LAMMPS Adapter Validation ===")
    command = "lmp -in input.in -log log.lammps"
    adapter = get_restart_adapter(command)
    print(f"  Adapter matched: {adapter.name}")
    assert adapter.name == "lammps", f"Expected lammps, got {adapter.name}"

    # Check for restart files
    restart_files = list(sync_dir.glob("restart.*.bin"))
    print(f"  Restart files found: {len(restart_files)}")
    for f in sorted(restart_files):
        print(f"    {f.name} ({f.stat().st_size} bytes)")

    result = adapter.prepare_restart(command, sync_dir, "test_lammps")
    if result:
        print(f"  Restart command: {result.command}")
        print(f"  Description: {result.description}")
        print(f"  Files written: {result.files_written}")
        # Verify wrapper script was created
        wrapper = sync_dir / "ccm_restart_lammps.sh"
        assert wrapper.exists(), "Wrapper script not created"
        print(f"  Wrapper script exists: {wrapper}")
        print("  PASS")
        return True
    else:
        if not restart_files:
            print("  No restart files — adapter correctly returned None")
            print("  PASS (no checkpoint)")
            return True
        print("  FAIL: restart files exist but adapter returned None")
        return False


def validate_pytorch(sync_dir: Path) -> bool:
    """Validate PyTorch/generic adapter with real checkpoint files."""
    print("\n=== PyTorch (Generic) Adapter Validation ===")
    command = 'python -c "import torch; ..."'
    adapter = get_restart_adapter(command)
    print(f"  Adapter matched: {adapter.name}")
    # Generic python script → falls to generic adapter
    assert adapter.name == "generic", f"Expected generic, got {adapter.name}"

    # Check for checkpoint files
    ckpt_files = list(sync_dir.glob("checkpoint_*.pt"))
    print(f"  Checkpoint files found: {len(ckpt_files)}")
    for f in sorted(ckpt_files):
        print(f"    {f.name} ({f.stat().st_size} bytes)")

    result = adapter.prepare_restart(command, sync_dir, "test_pytorch")
    print(f"  Restart result: {result}")
    # Generic adapter always returns None — job re-runs original command
    # PyTorch script is self-healing (checks for checkpoint on startup)
    assert result is None, "Generic adapter should return None"
    print("  PASS (generic adapter correctly returns None)")
    return True


def validate_gromacs(sync_dir: Path) -> bool:
    """Validate GROMACS adapter with real checkpoint files."""
    print("\n=== GROMACS Adapter Validation ===")
    command = "gmx mdrun -deffnm md -nsteps 50000 -cpt 0.1 -v"
    adapter = get_restart_adapter(command)
    print(f"  Adapter matched: {adapter.name}")
    assert adapter.name == "gromacs", f"Expected gromacs, got {adapter.name}"

    # Check for checkpoint files
    cpt_files = list(sync_dir.glob("*.cpt"))
    print(f"  Checkpoint files found: {len(cpt_files)}")
    for f in sorted(cpt_files):
        print(f"    {f.name} ({f.stat().st_size} bytes)")

    result = adapter.prepare_restart(command, sync_dir, "test_gromacs")
    if result:
        print(f"  Restart command: {result.command}")
        print(f"  Description: {result.description}")
        # Verify -cpi was injected
        assert "-cpi" in result.command, "Expected -cpi in restart command"
        # Verify no double -cpi (original didn't have it)
        assert result.command.count("-cpi") == 1, "Double -cpi detected"
        print("  PASS")
        return True
    else:
        if not cpt_files:
            print("  No checkpoint files — adapter correctly returned None")
            print("  PASS (no checkpoint)")
            return True
        print("  FAIL: checkpoint files exist but adapter returned None")
        return False


def find_sync_dir(job_name_pattern: str, sync_base: Path) -> Path | None:
    """Find the sync directory for a job by name pattern."""
    import sqlite3
    db_path = sync_base.parent / "cloudcomputemanager.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT job_id FROM jobs WHERE name LIKE ? ORDER BY created_at DESC LIMIT 1",
        (f"%{job_name_pattern}%",),
    ).fetchall()
    conn.close()
    if not rows:
        return None
    job_id = rows[0][0]
    sync_dir = sync_base / job_id
    return sync_dir if sync_dir.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Validate restart adapters against real data")
    parser.add_argument("--sync-base", type=Path,
                        default=Path.home() / ".cloudcomputemanager" / "sync")
    args = parser.parse_args()

    print("=" * 60)
    print("Restart Adapter Integration Validation")
    print("=" * 60)
    print(f"Sync base: {args.sync_base}")

    # List all registered adapters
    print(f"\nRegistered adapters ({len(RESTART_ADAPTERS)}):")
    for cls in RESTART_ADAPTERS:
        a = cls()
        print(f"  - {a.name}")

    results = {}

    # Validate each job type
    for name, validator in [
        ("adapter-test-lammps", validate_lammps),
        ("adapter-test-pytorch", validate_pytorch),
        ("adapter-test-gromacs", validate_gromacs),
    ]:
        sync_dir = find_sync_dir(name, args.sync_base)
        if sync_dir:
            print(f"\nFound sync dir for {name}: {sync_dir}")
            results[name] = validator(sync_dir)
        else:
            print(f"\nSync dir not found for {name} — skipping")
            results[name] = None

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else ("SKIP" if passed is None else "FAIL")
        print(f"  {name}: {status}")

    all_tested = [v for v in results.values() if v is not None]
    if all_tested and all(all_tested):
        print("\nAll tested adapters PASSED")
        return 0
    elif not all_tested:
        print("\nNo jobs to validate — submit test jobs first")
        return 1
    else:
        print("\nSome adapters FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
