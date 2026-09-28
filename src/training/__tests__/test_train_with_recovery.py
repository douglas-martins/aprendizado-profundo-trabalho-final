import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import training.train as train
import training.train_with_recovery as train_with_recovery
from training.train_with_recovery import (
    _TAIL_LINES,
    _run_attempt,
    default_state_path,
    is_gpu_crash,
    run_label,
    run_with_recovery,
    split_launcher_args,
)

CRASH = "torch.AcceleratorError: CUDA error: the launch timed out and was terminated"


FAKE_TRAIN = """
import json, sys
from pathlib import Path

base = Path(r"__BASE__")
state = base / "state.json"
n = json.loads(state.read_text())["n"] if state.exists() else 0
state.write_text(json.dumps({"n": n + 1}))
plan = json.loads((base / "plan.json").read_text())
step = plan[min(n, len(plan) - 1)]
with (base / "calls.jsonl").open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\\n")
if step.get("write_state"):
    Path(r"__STATE__").write_text(step["write_state"])
print("fake train output", n + 1)
if step["exit"] != 0:
    print(step["stderr"], file=sys.stderr)
sys.exit(step["exit"])
"""

OK = {"exit": 0}


def _crash(write_state=None):
    return {"exit": 1, "stderr": CRASH, "write_state": write_state}


@pytest.fixture
def fake(tmp_path):
    base = tmp_path / "fake"
    base.mkdir()
    state_path = tmp_path / "run_state" / "E2-raw-full.state.pt"
    state_path.parent.mkdir()
    script = tmp_path / "fake_train.py"
    script.write_text(
        FAKE_TRAIN.replace("__BASE__", str(base)).replace("__STATE__", str(state_path))
    )

    def launch(plan, **kwargs):
        (base / "plan.json").write_text(json.dumps(plan))
        return run_with_recovery(
            kwargs.pop("train_args", ["architecture=E2", "tier=raw-full"]),
            state_path=state_path,
            stale_state_dir=kwargs.pop("stale_state_dir", tmp_path / "stale"),
            log_path=kwargs.pop("log_path", tmp_path / "logs" / "run.log"),
            command_prefix=[sys.executable, str(script)],
            gpu_settle_seconds=0,
            **kwargs,
        )

    def calls():
        path = base / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return SimpleNamespace(
        launch=launch,
        calls=calls,
        state_path=state_path,
        log=tmp_path / "logs" / "run.log",
        stale=tmp_path / "stale",
    )


def _resumes_state(call):
    return "resume_state=true" in call


def _resume_from_arg(call):
    matches = [arg for arg in call if arg.startswith("resume_from=")]
    return matches[0].split("=", 1)[1] if matches else None


class TestRunWithRecovery:
    def test_opens_the_log_as_utf8(self, fake, monkeypatch):
        original_open = Path.open
        encodings = []

        def recording_open(path, *args, **kwargs):
            encodings.append(kwargs.get("encoding"))
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", recording_open)

        assert fake.launch([OK]) == 0
        assert "utf-8" in encodings

    def test_returns_zero_without_retrying_when_the_first_attempt_succeeds(self, fake):
        assert fake.launch([OK]) == 0

        assert len(fake.calls()) == 1
        assert not _resumes_state(fake.calls()[0])

    def test_relaunches_with_an_exact_state_resume_after_a_gpu_crash(self, fake):
        assert fake.launch([_crash(write_state="epoch-3"), OK]) == 0

        first, second = fake.calls()
        assert not _resumes_state(first)
        assert _resumes_state(second)

        assert fake.state_path.read_text() == "epoch-3"

    def test_passes_the_train_args_through_on_every_attempt(self, fake):
        fake.launch([_crash(write_state="s"), OK])

        for call in fake.calls():
            assert call[:2] == ["architecture=E2", "tier=raw-full"]

    def test_does_not_retry_an_ordinary_failure(self, fake):

        plan = [{"exit": 1, "stderr": "ValueError: a real bug"}]

        assert fake.launch(plan) == 1
        assert len(fake.calls()) == 1

    def test_does_not_retry_cuda_out_of_memory(self, fake):
        plan = [{"exit": 1, "stderr": "torch.OutOfMemoryError: CUDA out of memory"}]

        assert fake.launch(plan) == 1
        assert len(fake.calls()) == 1

    def test_gives_up_after_max_attempts(self, fake):
        assert fake.launch([_crash()], max_attempts=3) != 0

        assert len(fake.calls()) == 3

    def test_a_crash_before_any_state_was_saved_relaunches_from_scratch(self, fake):
        fake.launch([_crash(), OK])

        assert not _resumes_state(fake.calls()[1])

    def test_a_stale_state_from_an_earlier_run_is_moved_aside_before_the_first_attempt(self, fake):

        fake.state_path.write_text("old-experiment")

        fake.launch([OK])

        assert not fake.state_path.exists()
        (moved,) = fake.stale.iterdir()
        assert moved.read_text() == "old-experiment"
        assert re.fullmatch(r"E2-raw-full\.state-\d{8}-\d{6}\.pt", moved.name)
        assert not _resumes_state(fake.calls()[0])

    def test_a_stale_state_is_not_resumed_after_a_crash_that_saved_nothing(self, fake):
        fake.state_path.write_text("old-experiment")

        fake.launch([_crash(), OK])

        assert not _resumes_state(fake.calls()[1])

    def test_an_explicit_resume_state_keeps_the_existing_state_and_resumes_it(self, fake):
        fake.state_path.write_text("interrupted-run")

        fake.launch([OK], initial_resume_state=True)

        assert fake.state_path.read_text() == "interrupted-run"
        assert not fake.stale.exists()
        assert _resumes_state(fake.calls()[0])

    def test_a_weights_only_resume_from_is_dropped_once_the_state_takes_over(self, fake):

        args = ["architecture=E2", "tier=raw-full", "resume_from=/x/start.pt"]

        fake.launch([_crash(write_state="s"), OK], train_args=args)

        first, second = fake.calls()
        assert _resume_from_arg(first) == "/x/start.pt"
        assert _resume_from_arg(second) is None
        assert _resumes_state(second)

    def test_a_weights_only_resume_from_is_kept_when_the_crash_saved_no_state(self, fake):
        args = ["architecture=E2", "tier=raw-full", "resume_from=/x/start.pt"]

        fake.launch([_crash(), OK], train_args=args)

        assert _resume_from_arg(fake.calls()[1]) == "/x/start.pt"

    def test_the_log_records_every_attempt_and_the_recovery_decision(self, fake):
        fake.launch([_crash(write_state="s"), OK])

        text = fake.log.read_text()
        assert "attempt 1/" in text
        assert CRASH in text
        assert "attempt 2/" in text
        assert "fake train output 2" in text

    def test_returns_failure_without_launching_anything_when_no_attempts_are_allowed(self, fake):
        assert fake.launch([OK], max_attempts=0) == 1

        assert fake.calls() == []

    def test_creates_missing_parent_directories_for_the_log_and_reuses_existing_ones(
        self, fake, tmp_path
    ):
        log_path = tmp_path / "deep" / "nested" / "logs" / "run.log"

        assert fake.launch([OK], log_path=log_path) == 0

        assert fake.launch([OK], log_path=log_path) == 0

        assert log_path.read_text().count("finished cleanly") == 2

    def test_creates_missing_parent_directories_for_the_stale_state_and_reuses_them(
        self, fake, tmp_path, monkeypatch
    ):
        stale_dir = tmp_path / "deep" / "nested" / "stale"
        ticks = iter(["20260101-000001", "20260101-000002"])
        monkeypatch.setattr(
            train_with_recovery,
            "time",
            SimpleNamespace(
                time=lambda: 0.0, strftime=lambda *a: next(ticks), sleep=lambda seconds: None
            ),
        )

        fake.state_path.write_text("first")
        fake.launch([OK], stale_state_dir=stale_dir)

        fake.state_path.write_text("second")
        fake.launch([OK], stale_state_dir=stale_dir)

        assert sorted(f.read_text() for f in stale_dir.iterdir()) == ["first", "second"]

    def test_default_command_runs_train_py_unbuffered_with_the_current_interpreter(
        self, tmp_path, monkeypatch
    ):
        commands = []

        def fake_run_attempt(command, log_file):
            commands.append(command)
            return 0, ""

        monkeypatch.setattr(train_with_recovery, "_run_attempt", fake_run_attempt)

        code = run_with_recovery(
            ["architecture=E2"],
            state_path=tmp_path / "s.pt",
            stale_state_dir=tmp_path / "stale",
            log_path=tmp_path / "run.log",
        )

        assert code == 0
        assert commands == [
            [
                sys.executable,
                "-u",
                str(train_with_recovery._PROJECT_ROOT / "src" / "training" / "train.py"),
                "architecture=E2",
            ]
        ]

    def test_the_default_settle_delay_is_thirty_seconds(self, tmp_path, monkeypatch):
        sleeps = []
        monkeypatch.setattr(
            train_with_recovery,
            "time",
            SimpleNamespace(time=lambda: 0.0, strftime=lambda *a: "t", sleep=sleeps.append),
        )
        results = iter([(1, CRASH), (0, "")])
        monkeypatch.setattr(train_with_recovery, "_run_attempt", lambda c, f: next(results))

        run_with_recovery(
            [],
            state_path=tmp_path / "s.pt",
            stale_state_dir=tmp_path / "stale",
            log_path=tmp_path / "run.log",
            command_prefix=["x"],
        )

        assert sleeps == [30.0]

    def test_the_log_carries_each_supervisor_note_verbatim(self, fake):
        fake.launch([_crash(write_state="s"), OK])
        text = fake.log.read_text()
        assert "architecture=E2 tier=raw-full" in text
        assert "GPU crash in attempt 1; run state found, relaunching with resume_state=true" in text
        assert "attempt 2 finished cleanly" in text

    def test_the_log_notes_a_crash_before_any_state_was_saved(self, fake):
        fake.launch([_crash(), OK])

        assert (
            "GPU crash in attempt 1 before it saved a run state; relaunching from scratch"
            in fake.log.read_text()
        )

    def test_the_log_notes_a_stale_state_being_moved_aside(self, fake):
        fake.state_path.write_text("old")

        fake.launch([OK])

        assert "moved a stale run state aside to " in fake.log.read_text()

    def test_the_log_notes_an_ordinary_failure_is_not_retried(self, fake):
        fake.launch([{"exit": 3, "stderr": "ValueError: a real bug"}])

        assert (
            "attempt 1 failed (exit 3) without the GPU-crash signature; not retrying"
            in fake.log.read_text()
        )

    def test_the_log_notes_running_out_of_attempts(self, fake):
        fake.launch([_crash()], max_attempts=2)

        assert "attempt 2 hit the GPU crash again; out of attempts" in fake.log.read_text()


class TestIsGpuCrash:
    def test_recognises_the_launch_timeout_message(self):
        assert is_gpu_crash(CRASH) is True

    def test_recognises_the_cuda_error_name(self):
        assert is_gpu_crash("Search for `cudaErrorLaunchTimeout' in docs") is True

    @pytest.mark.parametrize(
        "text",
        [
            "ValueError: bad config",
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB",
            "",
        ],
    )
    def test_rejects_everything_else(self, text):
        assert is_gpu_crash(text) is False


class TestSplitLauncherArgs:
    def test_separates_its_own_args_from_the_ones_passed_to_train(self):
        argv = ["architecture=E2", "resume_state=true", "max_attempts=7", "seed=1"]

        train_args, max_attempts, resume_state = split_launcher_args(argv)

        assert train_args == ["architecture=E2", "seed=1"]
        assert max_attempts == 7
        assert resume_state is True

    def test_a_weights_only_resume_from_stays_a_train_arg(self):
        train_args, _, _ = split_launcher_args(["resume_from=/data/run=1/best.pt"])

        assert train_args == ["resume_from=/data/run=1/best.pt"]

    def test_a_malformed_max_attempts_value_is_rejected_not_truncated(self):
        with pytest.raises(ValueError):
            split_launcher_args(["max_attempts=3=4"])

    def test_defaults_when_the_launcher_args_are_absent(self):
        train_args, max_attempts, resume_state = split_launcher_args(["architecture=E3"])

        assert train_args == ["architecture=E3"]
        assert max_attempts == 5
        assert resume_state is False

    @pytest.mark.parametrize("value", ["false", "False", "0", ""])
    def test_only_true_turns_resume_state_on(self, value):
        _, _, resume_state = split_launcher_args([f"resume_state={value}"])

        assert resume_state is False

    @pytest.mark.parametrize("value", ["true=x", "x=true"])
    def test_a_value_containing_an_equals_sign_is_read_whole_not_truncated(self, value):
        _, _, resume_state = split_launcher_args([f"resume_state={value}"])

        assert resume_state is False

    def test_true_is_case_insensitive(self):
        assert split_launcher_args(["resume_state=TRUE"])[2] is True


class TestStatePathAndLabel:
    def test_default_state_path_matches_where_train_py_saves(self):
        args = ["architecture=E2", "tier=raw-full"]

        assert default_state_path(args) == train._STATE_DIR / "E2-raw-full.state.pt"

    def test_defaults_match_train_pys_own_defaults(self):
        assert default_state_path([]) == train._STATE_DIR / "E1-mini.state.pt"

    def test_run_label_names_the_architecture_and_tier(self):
        assert run_label(["architecture=E3", "tier=raw-full"]) == "E3-raw-full"

    def test_run_label_splits_each_override_on_its_first_equals_only(self):

        assert run_label(["note=a=b", "architecture=E=2"]) == "E=2-mini"


class TestRunAttempt:
    def test_streams_text_with_line_buffering_and_replacement_errors(self, monkeypatch):
        calls = []

        class Process:
            def __init__(self, text):
                self.stdout = iter(["first\\n", "second\\n"] if text else [b"first\\n"])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def wait(self):
                return 7

        def popen(*args, **kwargs):
            calls.append((args, kwargs))
            return Process(kwargs.get("text"))

        monkeypatch.setattr(train_with_recovery.subprocess, "Popen", popen)
        log = io.StringIO()

        result = _run_attempt(["python", "train.py"], log)

        assert result == (7, "first\\nsecond\\n")
        assert log.getvalue() == "first\\nsecond\\n"
        assert calls == [
            (
                (["python", "train.py"],),
                {
                    "stdout": train_with_recovery.subprocess.PIPE,
                    "stderr": train_with_recovery.subprocess.STDOUT,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "bufsize": 1,
                },
            )
        ]

    def test_streams_output_to_the_log_and_returns_the_exit_code_and_the_output_as_the_tail(self):
        log = io.StringIO()
        command = [sys.executable, "-c", "print('one'); print('two'); raise SystemExit(3)"]

        returncode, tail = _run_attempt(command, log)

        assert returncode == 3
        assert log.getvalue() == "one\ntwo\n"
        assert tail == "one\ntwo\n"

    def test_the_tail_keeps_only_the_last_tail_lines_while_the_log_keeps_everything(self):
        log = io.StringIO()
        total = _TAIL_LINES + 50
        command = [sys.executable, "-c", f"[print(i) for i in range({total})]"]

        _, tail = _run_attempt(command, log)

        assert tail.splitlines() == [str(i) for i in range(50, total)]
        assert len(log.getvalue().splitlines()) == total

    def test_undecodable_output_bytes_are_replaced_not_fatal(self):

        log = io.StringIO()
        code = (
            "import sys; sys.stdout.buffer.write(b'bad \\xff byte\\n'); "
            "sys.stdout.buffer.write(b'cudaErrorLaunchTimeout\\n')"
        )

        returncode, tail = _run_attempt([sys.executable, "-c", code], log)

        assert returncode == 0
        assert "bad � byte" in tail
        assert "cudaErrorLaunchTimeout" in tail
