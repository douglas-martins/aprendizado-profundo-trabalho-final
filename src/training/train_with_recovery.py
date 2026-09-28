"""Resume training after recoverable interruptions using persisted run state.

The wrapper restarts the training entry point when the configured failure is
recoverable and keeps state and logs in local generated-output directories.
"""

import collections
import shutil
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MAX_ATTEMPTS = 5
GPU_CRASH_SIGNATURES = ("launch timed out and was terminated", "cudaErrorLaunchTimeout")
_TAIL_LINES = 400


def is_gpu_crash(log_text: str) -> bool:
    """True if `log_text` (the end of an attempt's output) shows the known GPU-crash error."""
    return any(signature in log_text for signature in GPU_CRASH_SIGNATURES)


def split_launcher_args(argv: list[str]) -> tuple[list[str], int, bool]:
    """Split `argv` into `(args for train.py, max_attempts, resume_state)`.

    `max_attempts=` and `resume_state=` belong to this launcher (`resume_state` is
    re-added to `train.py`'s args per attempt, when a state file exists); everything
    else is passed to `train.py` unchanged. Only `resume_state=true`
    (case-insensitive) turns it on.
    """
    train_args: list[str] = []
    max_attempts = DEFAULT_MAX_ATTEMPTS
    resume_state = False
    for arg in argv:
        if arg.startswith("max_attempts="):
            max_attempts = int(arg.split("=", 1)[1])
        elif arg.startswith("resume_state="):
            resume_state = arg.split("=", 1)[1].lower() == "true"
        else:
            train_args.append(arg)
    return train_args, max_attempts, resume_state


def run_label(train_args: list[str]) -> str:
    """`<architecture>-<tier>`, using `train.py`'s own defaults (E1 / mini) when absent."""
    values = dict(arg.split("=", 1) for arg in train_args if "=" in arg)
    return f"{values.get('architecture', 'E1')}-{values.get('tier', 'mini')}"


def default_state_path(train_args: list[str]) -> Path:
    """Where `train.py` keeps this run's state (`run_state/<label>.state.pt`)."""
    return _PROJECT_ROOT / "run_state" / f"{run_label(train_args)}.state.pt"


def _note(log_file, message: str) -> None:
    log_file.write(f"[recovery] {message}\n")
    log_file.flush()


def _run_attempt(command: list[str], log_file) -> tuple[int, str]:
    """Run `command`, streaming its output line by line into `log_file`.

    Returns `(exit code, last _TAIL_LINES lines)` -- the tail is what the crash
    signature is looked for in, since a traceback is always at the end.
    """
    tail: collections.deque[str] = collections.deque(maxlen=_TAIL_LINES)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            tail.append(line)
        returncode = process.wait()
    return returncode, "".join(tail)


def run_with_recovery(
    train_args: list[str],
    *,
    state_path: Path,
    stale_state_dir: Path,
    log_path: Path,
    command_prefix: list[str] | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_resume_state: bool = False,
    gpu_settle_seconds: float = 30.0,
) -> int:
    """Run `train.py` with `train_args`, relaunching after a GPU crash; return the exit code.

    A state file at `state_path` when this starts is a leftover from an earlier run
    and is moved into `stale_state_dir` -- unless `initial_resume_state` (the user
    asked to continue it). After a GPU crash, if a state file exists (this launch's
    own, or the one being resumed) the next attempt gets `resume_state=true` and
    drops any weights-only `resume_from=` (`train.py` rejects both together; the
    state already carries those weights); otherwise it starts over exactly as the
    crashed attempt did. Stops on success, on any failure that isn't the GPU-crash
    signature, or after `max_attempts`.
    """
    prefix = command_prefix or [
        sys.executable,
        "-u",
        str(_PROJECT_ROOT / "src" / "training" / "train.py"),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = 1

    with log_path.open("a", encoding="utf-8") as log:
        if state_path.exists() and not initial_resume_state:
            stale_state_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            moved_to = stale_state_dir / f"{state_path.stem}-{stamp}{state_path.suffix}"
            shutil.move(state_path, moved_to)
            _note(log, f"moved a stale run state aside to {moved_to}")
        resume_state = initial_resume_state
        for attempt in range(1, max_attempts + 1):
            attempt_args = train_args
            if resume_state:
                attempt_args = [a for a in train_args if not a.startswith("resume_from=")]
                attempt_args = [*attempt_args, "resume_state=true"]
            command = [*prefix, *attempt_args]
            _note(log, f"attempt {attempt}/{max_attempts}: {' '.join(command)}")
            returncode, tail = _run_attempt(command, log)

            if returncode == 0:
                _note(log, f"attempt {attempt} finished cleanly")
                return 0
            if not is_gpu_crash(tail):
                _note(
                    log,
                    f"attempt {attempt} failed (exit {returncode}) without the GPU-crash "
                    "signature; not retrying",
                )
                return returncode
            if attempt == max_attempts:
                _note(log, f"attempt {attempt} hit the GPU crash again; out of attempts")
                return returncode

            resume_state = state_path.exists()
            if resume_state:
                _note(
                    log,
                    f"GPU crash in attempt {attempt}; run state found, "
                    "relaunching with resume_state=true",
                )
            else:
                _note(
                    log,
                    f"GPU crash in attempt {attempt} before it saved a run state; "
                    "relaunching from scratch",
                )
            time.sleep(gpu_settle_seconds)
    return returncode


def main() -> int:
    """CLI entry point -- see the module docstring."""
    train_args, max_attempts, initial_resume_state = split_launcher_args(sys.argv[1:])
    return run_with_recovery(
        train_args,
        state_path=default_state_path(train_args),
        stale_state_dir=_PROJECT_ROOT / "run_state" / "stale",
        log_path=_PROJECT_ROOT / "logs" / f"{run_label(train_args)}-recovering.log",
        max_attempts=max_attempts,
        initial_resume_state=initial_resume_state,
    )


if __name__ == "__main__":
    sys.exit(main())
