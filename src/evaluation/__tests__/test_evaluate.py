import pytest
import torch

import evaluation.evaluate as evaluate
from evaluation.evaluate import _build_run_name, evaluate_full_metrics, load_checkpoint
from models.architectures import build_e1
from utils.power_meter import EnergyReading


class _FixedLogitModel(torch.nn.Module):
    """Returns pre-baked logits regardless of input -- lets tests control predictions exactly.

    Duplicated from test_train.py's own helper of the same name rather than
    imported -- each test module owns its fixtures here, matching this
    project's established per-file convention (e.g. `_make_scene`/
    `_patches_df` are similarly duplicated, not shared, across test files).
    """

    def __init__(self, logits_by_call: list[torch.Tensor]):
        super().__init__()
        self._logits_by_call = iter(logits_by_call)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return next(self._logits_by_call)


class _RecordingMeter:
    """A meter that records when it is started and stopped, and reports a canned reading."""

    def __init__(self, events, reading):
        self.events = events
        self.reading = reading

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")


class _EventModel(_FixedLogitModel):
    def __init__(self, events, logits_by_call):
        super().__init__(logits_by_call)
        self.events = events

    def forward(self, x):
        self.events.append("forward")
        return super().forward(x)


class TestEvaluateEnergy:
    def _batches(self, n):
        return [{"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)}] * n

    def test_the_meter_covers_the_inference_loop_and_nothing_else(self):
        events = []
        reading = EnergyReading(seconds=1.0, gpu_joules=10.0, cpu_joules=5.0)
        model = _EventModel(events, [torch.zeros(1, 1, 1, 1)] * 2)

        result = evaluate_full_metrics(
            model, self._batches(2), device="cpu", meter=_RecordingMeter(events, reading)
        )

        assert events == ["start", "forward", "forward", "stop"]
        assert result["energy"] == reading

    def test_a_meter_is_created_by_default_and_reports_whatever_this_machine_can_measure(self):
        model = _FixedLogitModel([torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, self._batches(1), device="cpu")

        assert isinstance(result["energy"], EnergyReading)
        assert result["energy"].seconds >= 0


class TestEvaluateFullMetrics:
    def test_synchronizes_before_and_after_cuda_inference(self, monkeypatch):
        syncs = []
        original_zeros = torch.zeros

        def tensor_to(tensor, *args, **kwargs):
            return tensor

        def zeros_without_device(*args, **kwargs):
            kwargs.pop("device", None)
            return original_zeros(*args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", tensor_to)
        monkeypatch.setattr(evaluate.torch.cuda, "synchronize", lambda device: syncs.append(device))
        monkeypatch.setattr(evaluate.torch, "zeros", zeros_without_device)
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)}]

        evaluate_full_metrics(
            _FixedLogitModel([torch.zeros(1, 1, 1, 1)]),
            batches,
            device="cuda",
            pr_thresholds=torch.tensor([0.5]),
        )

        assert syncs == ["cuda", "cuda"]

    def test_allocates_the_sweep_totals_on_the_requested_device(self, monkeypatch):
        calls = []
        original_zeros = torch.zeros

        def recording_zeros(*args, **kwargs):
            calls.append(kwargs.get("device"))
            return original_zeros(*args, **kwargs)

        monkeypatch.setattr(evaluate.torch, "zeros", recording_zeros)
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)}]

        evaluate_full_metrics(
            _FixedLogitModel([torch.zeros(1, 1, 1, 1)]),
            batches,
            device="cpu",
            pr_thresholds=torch.tensor([0.5]),
        )

        assert calls[-1] == "cpu"

    def test_pools_counts_across_batches_instead_of_averaging_per_batch_scores(self):

        batches = [
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            },
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
            },
        ]
        logits_batch1 = torch.tensor([[[[5.0, 5.0], [-5.0, -5.0]]], [[[-5.0, -5.0], [-5.0, -5.0]]]])
        logits_batch2 = torch.full((2, 1, 2, 2), -5.0)
        model = _FixedLogitModel([logits_batch1, logits_batch2])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5
        assert result["confusion_matrix"] == [[13.0, 1.0], [1.0, 1.0]]

    def test_perfect_prediction_scores_one_on_both_precision_and_recall(self):
        batches = [
            {"input": torch.zeros(1, 4, 2, 2), "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])}
        ]
        logits = torch.tensor([[[[5.0, -5.0], [-5.0, -5.0]]]])
        model = _FixedLogitModel([logits])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_respects_a_non_default_threshold(self):

        logit_for_p06 = torch.logit(torch.tensor(0.6))
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}]
        model_default = _FixedLogitModel([torch.full((1, 1, 1, 1), logit_for_p06.item())])
        model_strict = _FixedLogitModel([torch.full((1, 1, 1, 1), logit_for_p06.item())])

        default_result = evaluate_full_metrics(model_default, batches, device="cpu")
        strict_result = evaluate_full_metrics(model_strict, batches, device="cpu", threshold=0.7)

        assert default_result["recall"] == 1.0
        assert strict_result["recall"] == 0.0

    def test_counts_total_patches_processed_across_batches(self):

        batches = [
            {"input": torch.zeros(2, 4, 1, 1), "output": torch.zeros(2, 1, 1, 1)},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)},
        ]
        model = _FixedLogitModel([torch.zeros(2, 1, 1, 1), torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["patches_processed"] == 3

    def test_stops_after_max_batches(self):

        batches = [
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
        ]

        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0), torch.full((1, 1, 1, 1), -5.0)])

        result = evaluate_full_metrics(model, batches, device="cpu", max_batches=1)

        assert result["recall"] == 1.0

    def test_stops_after_exactly_max_batches_out_of_more_than_two(self):

        batches = [
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}
            for _ in range(3)
        ]
        model = _FixedLogitModel([torch.zeros(1, 1, 1, 1)] * 3)

        result = evaluate_full_metrics(model, batches, device="cpu", max_batches=2)

        assert result["patches_processed"] == 2

    def test_moves_inputs_targets_and_pr_thresholds_to_the_requested_device(self, monkeypatch):
        calls = []
        original_to = torch.Tensor.to

        def recording_to(tensor, *args, **kwargs):
            calls.append(args[0] if args else kwargs.get("device"))
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", recording_to)
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)}]

        evaluate_full_metrics(
            _FixedLogitModel([torch.zeros(1, 1, 1, 1)]),
            batches,
            device="cpu",
            pr_thresholds=torch.tensor([0.5]),
        )

        assert calls.count("cpu") == 3
        assert None not in calls

    def test_model_receives_the_batchs_actual_input_tensor(self):

        received = []

        class _RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x):
                received.append(x)
                return torch.zeros(1, 1, 1, 1)

        batch_input = torch.full((1, 4, 1, 1), 3.0)
        batches = [{"input": batch_input, "output": torch.zeros(1, 1, 1, 1)}]

        evaluate_full_metrics(_RecordingModel(), batches, device="cpu")

        assert received[0] is not None
        torch.testing.assert_close(received[0], batch_input)

    def test_probability_exactly_at_threshold_is_not_predicted_positive(self):

        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}]
        model = _FixedLogitModel([torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, batches, device="cpu", threshold=0.5)

        assert result["confusion_matrix"] == [[0.0, 0.0], [1.0, 0.0]]

    def test_wall_clock_is_the_elapsed_duration_not_a_sum_of_timestamps(self, monkeypatch):

        timestamps = iter([1000.0, 1000.25])
        monkeypatch.setattr("evaluation.evaluate.time.perf_counter", lambda: next(timestamps))
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)}]
        model = _FixedLogitModel([torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(
            model, batches, device="cpu", meter=_RecordingMeter([], None)
        )

        assert result["wall_clock_seconds"] == pytest.approx(0.25)


class TestEvaluateFullMetricsPRAUC:
    def test_computes_pr_auc_over_the_given_threshold_sweep(self):

        batches = [
            {
                "input": torch.zeros(4, 4, 1, 1),
                "output": torch.tensor([[[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]]]),
            }
        ]
        probs = torch.tensor([0.9, 0.4, 0.1, 0.6])
        logits = torch.logit(probs).reshape(4, 1, 1, 1)
        model = _FixedLogitModel([logits])

        result = evaluate_full_metrics(
            model, batches, device="cpu", pr_thresholds=torch.tensor([0.0, 0.5, 1.0])
        )

        assert result["pr_auc"] == pytest.approx(0.5, abs=1e-4)
        assert result["precision_recall_curve"] == pytest.approx(
            [(1.0, 0.5), (0.5, 0.5), (0.0, 0.0)], abs=1e-4
        )

    def test_sweep_pools_across_batches_not_just_the_last_one(self):

        batches = [
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])},
        ]
        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0), torch.full((1, 1, 1, 1), -5.0)])

        result = evaluate_full_metrics(
            model, batches, device="cpu", pr_thresholds=torch.tensor([0.5])
        )

        assert result["pr_auc"] == pytest.approx(0.5)

    def test_defaults_to_a_101_point_threshold_sweep(self):
        batches = [{"input": torch.zeros(1, 4, 1, 1), "output": torch.tensor([[[[1.0]]]])}]
        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0)])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert len(result["precision_recall_curve"]) == 101

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_works_when_model_and_batches_are_on_cuda(self):

        batches = [
            {
                "input": torch.zeros(1, 4, 1, 1, device="cuda"),
                "output": torch.tensor([[[[1.0]]]], device="cuda"),
            }
        ]
        model = _FixedLogitModel([torch.full((1, 1, 1, 1), 5.0, device="cuda")])

        result = evaluate_full_metrics(model, batches, device="cuda")

        assert result["recall"] == 1.0
        assert result["pr_auc"] > 0.0


class TestEvaluateFullMetricsPerPatchDetection:
    def test_computes_per_patch_detection_counts_and_rate(self):
        batches = [
            {
                "input": torch.zeros(2, 4, 2, 2),
                "output": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[1.0, 0.0], [0.0, 0.0]]]]),
            }
        ]

        logits = torch.tensor([[[[5.0, -5.0], [-5.0, -5.0]]], [[[-5.0, -5.0], [-5.0, -5.0]]]])
        model = _FixedLogitModel([logits])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["per_patch_positive_patches"] == 2
        assert result["per_patch_detected_patches"] == 1
        assert result["per_patch_detection_rate"] == 0.5


class TestEvaluateFullMetricsTiming:
    def test_returns_positive_wall_clock_and_consistent_throughput(self):

        batches = [
            {"input": torch.zeros(2, 4, 1, 1), "output": torch.zeros(2, 1, 1, 1)},
            {"input": torch.zeros(1, 4, 1, 1), "output": torch.zeros(1, 1, 1, 1)},
        ]
        model = _FixedLogitModel([torch.zeros(2, 1, 1, 1), torch.zeros(1, 1, 1, 1)])

        result = evaluate_full_metrics(model, batches, device="cpu")

        assert result["wall_clock_seconds"] > 0
        assert result["patches_per_second"] == pytest.approx(
            result["patches_processed"] / result["wall_clock_seconds"]
        )


class TestBuildRunName:
    def test_same_tier_default_device_has_no_suffix(self):

        assert _build_run_name("E1", "mini", "mini", None, "cuda") == "E1-mini-eval"

    def test_cross_tier_default_device_has_no_suffix(self):
        assert _build_run_name("E2", "raw-full", "mini", None, "cuda") == "E2-mini-on-raw-full-eval"

    def test_explicit_device_appends_a_suffix(self):

        assert _build_run_name("E1", "mini", "mini", "cpu", "cpu") == "E1-mini-eval-cpu"

    def test_cross_tier_with_explicit_device_appends_a_suffix(self):
        assert (
            _build_run_name("E2", "raw-full", "mini", "cpu", "cpu")
            == "E2-mini-on-raw-full-eval-cpu"
        )


class TestLoadCheckpoint:
    def test_passes_the_requested_device_to_checkpoint_loading_and_model(self, monkeypatch):
        calls = []

        class Model(torch.nn.Module):
            def load_state_dict(self, state):
                calls.append(("state", state))

            def to(self, device):
                calls.append(("to", device))
                return self

        monkeypatch.setattr(evaluate, "build_model", lambda architecture: Model())
        monkeypatch.setattr(
            evaluate.torch,
            "load",
            lambda path, **kwargs: calls.append(("load", path, kwargs)) or {"weight": 1},
        )

        loaded = load_checkpoint("E1", "weights.pt", device="cpu")

        assert loaded.training is False
        assert calls == [
            ("load", "weights.pt", {"map_location": "cpu"}),
            ("state", {"weight": 1}),
            ("to", "cpu"),
        ]

    def test_loads_saved_weights_and_reproduces_the_same_forward_pass(self, tmp_path):
        original = build_e1()
        checkpoint_path = tmp_path / "e1.pt"
        torch.save(original.state_dict(), checkpoint_path)

        loaded = load_checkpoint("E1", checkpoint_path, device="cpu")

        sample_input = torch.randn(1, 4, 16, 16)
        original.eval()
        with torch.no_grad():
            assert torch.equal(loaded(sample_input), original(sample_input))
