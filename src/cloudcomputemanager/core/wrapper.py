"""Wrapper script builder for CCM job execution.

Generates SIGTERM-aware bash scripts that:
- Capture exit codes for completion detection (.ccm_exit_code)
- Trap SIGTERM for graceful preemption handling
- Send SIGUSR1 to trigger application checkpoints (e.g., LAMMPS restart files)
- Write preemption marker (.ccm_preempted) for diagnostics
- Use exit code 143 (128+SIGTERM) to distinguish preemption from normal failure
"""

import base64


# Exit code written when SIGTERM is received (128 + 15 = SIGTERM)
PREEMPTION_EXIT_CODE = 143


def build_wrapper_script(command: str, stage_name: str | None = None) -> str:
    """Build a SIGTERM-aware wrapper script for job execution.

    The wrapper runs the command as a background process and uses `wait` so
    that the SIGTERM trap can fire. On SIGTERM (Vast.ai preemption signal):
    1. Writes .ccm_preempted marker with timestamp
    2. Sends SIGUSR1 to job (triggers LAMMPS/GROMACS checkpoint)
    3. Sends SIGTERM to job for graceful shutdown
    4. Waits up to 10s for exit
    5. Writes exit code 143 to .ccm_exit_code

    Args:
        command: The job command to execute
        stage_name: Optional stage name for multi-stage jobs

    Returns:
        Shell script content as string
    """
    label = f"Stage: {stage_name}" if stage_name else "CCM Job"

    return f'''#!/bin/bash
# {label} — CCM wrapper with SIGTERM handling
cd /workspace

_ccm_sigterm_handler() {{
    echo "CCM: SIGTERM received at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /workspace/.ccm_preempted
    if [ -n "$JOB_PID" ]; then
        kill -USR1 $JOB_PID 2>/dev/null
        sleep 2
        kill -TERM $JOB_PID 2>/dev/null
        for i in $(seq 1 10); do
            kill -0 $JOB_PID 2>/dev/null || break
            sleep 1
        done
    fi
    # Only write exit code if not already written by normal completion
    if [ ! -f /workspace/.ccm_exit_code ]; then
        echo {PREEMPTION_EXIT_CODE} > /workspace/.ccm_exit_code
    fi
    echo "CCM: Preemption shutdown complete at $(date)" >> /workspace/job.log
    exit {PREEMPTION_EXIT_CODE}
}}
trap _ccm_sigterm_handler SIGTERM

# Run job in background so trap can fire during wait
set +e
{command} &
JOB_PID=$!
wait $JOB_PID
JOB_EXIT_CODE=$?
set -e

echo $JOB_EXIT_CODE > /workspace/.ccm_exit_code
echo "Job completed with exit code $JOB_EXIT_CODE" >> /workspace/job.log
exit $JOB_EXIT_CODE
'''


def encode_wrapper_script(script: str) -> str:
    """Base64-encode a wrapper script for SSH transfer."""
    return base64.b64encode(script.encode()).decode()


def build_deploy_commands(
    command: str,
    stage_name: str | None = None,
    script_path: str = "/workspace/run_job.sh",
) -> tuple[str, str]:
    """Build SSH commands to deploy and run a wrapper script.

    Args:
        command: The job command to execute
        stage_name: Optional stage name for multi-stage jobs
        script_path: Path on instance to write the script

    Returns:
        (setup_cmd, run_cmd) — setup writes the script, run launches it
    """
    script = build_wrapper_script(command, stage_name)
    b64 = encode_wrapper_script(script)
    setup_cmd = f"echo {b64} | base64 -d > {script_path} && chmod +x {script_path}"
    run_cmd = f"cd /workspace && nohup bash {script_path} > /workspace/job.log 2>&1 &"
    return setup_cmd, run_cmd
