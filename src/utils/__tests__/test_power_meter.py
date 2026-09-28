import builtins
import os

import pytest

import utils.power_meter as power_meter
from utils.power_meter import (
    EnergyReading,
    PowerMeter,
    RaplReader,
    idle_power,
    nvml_energy_reader,
    reading_to_metrics,
)


class _Clock:
    """Deterministic stand-in for `time.perf_counter`."""

    def __init__(self, *times):
        self._times = iter(times)

    def __call__(self):
        return next(self._times)


def _counter(*values):
    """A cumulative-joules reader that returns `values` in order."""
    iterator = iter(values)
    return lambda: next(iterator)


class TestEnergyReading:
    def test_average_power_is_energy_over_time(self):
        reading = EnergyReading(seconds=2.0, gpu_joules=100.0, cpu_joules=40.0)

        assert reading.gpu_watts == 50.0
        assert reading.cpu_watts == 20.0
        assert reading.total_joules == 140.0
        assert reading.total_watts == 70.0

    def test_a_missing_device_is_none_and_makes_the_total_none(self):
        reading = EnergyReading(seconds=2.0, gpu_joules=100.0, cpu_joules=None)

        assert reading.cpu_watts is None
        assert reading.total_joules is None
        assert reading.total_watts is None
        assert reading.gpu_watts == 50.0

    def test_zero_duration_gives_no_power(self):
        reading = EnergyReading(seconds=0.0, gpu_joules=1.0, cpu_joules=1.0)

        assert reading.gpu_watts is None
        assert reading.total_watts is None


class TestPowerMeter:
    def test_reports_the_energy_used_between_start_and_stop(self):
        meter = PowerMeter(
            readers={"gpu": _counter(1000.0, 1150.0), "cpu": _counter(50.0, 80.0)},
            clock=_Clock(10.0, 12.0),
        )

        meter.start()
        meter.stop()

        assert meter.reading == EnergyReading(seconds=2.0, gpu_joules=150.0, cpu_joules=30.0)

    def test_works_as_a_context_manager(self):
        with PowerMeter(
            readers={"gpu": _counter(0.0, 5.0), "cpu": _counter(0.0, 7.0)},
            clock=_Clock(0.0, 1.0),
        ) as meter:
            pass

        assert meter.reading.gpu_joules == 5.0
        assert meter.reading.cpu_joules == 7.0

    def test_an_exception_inside_the_block_is_not_swallowed_and_the_meter_still_stops(self):
        meter = PowerMeter(
            readers={"gpu": _counter(0.0, 5.0), "cpu": _counter(0.0, 7.0)}, clock=_Clock(0.0, 1.0)
        )

        with pytest.raises(ValueError, match="boom"), meter:
            raise ValueError("boom")

        assert meter.reading.gpu_joules == 5.0

    def test_a_device_that_cannot_be_read_is_reported_as_none(self):
        meter = PowerMeter(
            readers={"gpu": _counter(0.0, 5.0), "cpu": lambda: None}, clock=_Clock(0.0, 1.0)
        )

        meter.start()
        meter.stop()

        assert meter.reading.gpu_joules == 5.0
        assert meter.reading.cpu_joules is None

    def test_a_reader_that_fails_midway_is_reported_as_none_not_raised(self):
        def flaky():
            flaky.calls += 1
            if flaky.calls == 2:
                raise OSError("counter went away")
            return 1.0

        flaky.calls = 0
        meter = PowerMeter(readers={"gpu": flaky, "cpu": lambda: None}, clock=_Clock(0.0, 1.0))

        meter.start()
        meter.stop()

        assert meter.reading.gpu_joules is None

    def test_the_reading_is_unavailable_before_the_meter_has_been_stopped(self):
        meter = PowerMeter(readers={"gpu": lambda: 1.0, "cpu": lambda: 1.0}, clock=_Clock(0.0))
        meter.start()

        assert meter.reading is None

    def test_stopping_a_meter_that_was_never_started_is_an_error(self):
        meter = PowerMeter(readers={"gpu": lambda: 1.0, "cpu": lambda: 1.0})

        with pytest.raises(RuntimeError) as error:
            meter.stop()

        assert str(error.value) == "meter not started"

    def test_initial_state_has_no_reading_and_uses_zero_start_time(self):
        meter = PowerMeter(readers={})

        assert meter.reading is None
        assert meter._started_at == 0.0

    def test_the_default_meter_can_be_created_and_run_on_any_machine(self):

        with PowerMeter() as meter:
            pass

        assert meter.reading.seconds >= 0


class TestRaplReader:
    def _domain(self, root, name, energy_uj, max_range_uj):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "energy_uj").write_text(f"{energy_uj}\n")
        (directory / "max_energy_range_uj").write_text(f"{max_range_uj}\n")
        return directory

    def test_reads_the_package_counter_in_joules(self, tmp_path):
        self._domain(tmp_path, "intel-rapl:0", 2_500_001, 100_000_000)

        assert RaplReader(tmp_path)() == 2.500001

    def test_the_default_root_is_linux_powercap(self):
        assert RaplReader()._root == "/sys/class/powercap"

    def test_uses_the_linux_counter_file_names_exactly(self, monkeypatch):
        values = {"/intel-rapl:0/energy_uj": "100", "/intel-rapl:0/max_energy_range_uj": "1000"}
        calls = []

        class File:
            def __init__(self, value):
                self.value = value

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return self.value

        def open_file(path, *args, **kwargs):
            calls.append(path)
            return File(values[path])

        monkeypatch.setattr(power_meter.glob, "glob", lambda pattern: ["/intel-rapl:0"])
        monkeypatch.setattr(builtins, "open", open_file)

        assert power_meter.RaplReader("/")() == 0.0001
        assert calls == ["/intel-rapl:0/energy_uj", "/intel-rapl:0/max_energy_range_uj"]

    def test_a_missing_counter_file_means_unavailable(self, tmp_path):
        directory = tmp_path / "intel-rapl:0"
        directory.mkdir()
        (directory / "max_energy_range_uj").write_text("100\n")

        assert RaplReader(tmp_path)() is None

    def test_a_missing_range_file_means_unavailable(self, tmp_path):
        directory = tmp_path / "intel-rapl:0"
        directory.mkdir()
        (directory / "energy_uj").write_text("100\n")

        assert RaplReader(tmp_path)() is None

    def test_sums_every_package_and_ignores_sub_domains(self, tmp_path):
        self._domain(tmp_path, "intel-rapl:0", 1_000_000, 100_000_000)
        self._domain(tmp_path, "intel-rapl:1", 2_000_000, 100_000_000)
        self._domain(tmp_path, "intel-rapl:0:0", 900_000, 100_000_000)

        assert RaplReader(tmp_path)() == pytest.approx(3.0)

    def test_a_counter_that_wrapped_is_accumulated_across_the_wrap(self, tmp_path):
        domain = self._domain(tmp_path, "intel-rapl:0", 90_000_000, 100_000_000)
        reader = RaplReader(tmp_path)
        first = reader()
        (domain / "energy_uj").write_text("10000000\n")

        second = reader()

        assert second - first == pytest.approx(20.0)

    def test_a_counter_that_increases_without_wrapping_adds_the_difference(self, tmp_path):
        domain = self._domain(tmp_path, "intel-rapl:0", 90_000_000, 100_000_000)
        reader = RaplReader(tmp_path)
        reader()
        (domain / "energy_uj").write_text("90000001\n")

        assert reader() == 90.000001

    def test_an_unchanged_counter_does_not_wrap(self, tmp_path):
        self._domain(tmp_path, "intel-rapl:0", 90_000_000, 100_000_000)
        reader = RaplReader(tmp_path)

        assert reader() == pytest.approx(90.0)
        assert reader() == pytest.approx(90.0)

    def test_no_domains_means_unavailable(self, tmp_path):
        assert RaplReader(tmp_path)() is None

    def test_a_missing_root_means_unavailable(self, tmp_path):
        assert RaplReader(tmp_path / "does-not-exist")() is None

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read any file")
    def test_an_unreadable_counter_means_unavailable_not_an_error(self, tmp_path):
        domain = self._domain(tmp_path, "intel-rapl:0", 1_000_000, 100_000_000)
        (domain / "energy_uj").chmod(0o000)

        assert RaplReader(tmp_path)() is None


class _FakeNvml:
    def __init__(self, millijoules, fail=False):
        self._millijoules = iter(millijoules)
        self.fail = fail
        self.initialised = False

    def nvmlInit(self):
        if self.fail:
            raise RuntimeError("no driver")
        self.initialised = True

    def nvmlDeviceGetHandleByIndex(self, index):
        return ("handle", index)

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        assert handle == ("handle", 0)
        return next(self._millijoules)


class TestNvmlEnergyReader:
    def test_converts_the_millijoule_counter_to_joules(self):
        reader = nvml_energy_reader(nvml=_FakeNvml([2_500, 4_000]))

        assert reader() == pytest.approx(2.5)
        assert reader() == pytest.approx(4.0)

    def test_the_device_index_is_used(self):
        class _Indexed(_FakeNvml):
            def nvmlDeviceGetHandleByIndex(self, index):
                return ("handle", 0 if index == 3 else -1)

        assert nvml_energy_reader(index=3, nvml=_Indexed([1_000]))() == pytest.approx(1.0)

    def test_a_missing_driver_means_unavailable(self):
        assert nvml_energy_reader(nvml=_FakeNvml([], fail=True))() is None

    def test_a_counter_that_stops_working_means_unavailable(self):
        reader = nvml_energy_reader(nvml=_FakeNvml([]))

        assert reader() is None


class TestReadingToMetrics:
    def test_names_the_energy_power_and_energy_per_item_metrics(self):
        reading = EnergyReading(seconds=4.0, gpu_joules=200.0, cpu_joules=100.0)

        metrics = reading_to_metrics("test", reading, n_items=1000, item="patch")

        assert metrics == {
            "test_energy_gpu_joules": 200.0,
            "test_energy_cpu_joules": 100.0,
            "test_energy_total_joules": 300.0,
            "test_power_gpu_watts": 50.0,
            "test_power_cpu_watts": 25.0,
            "test_power_total_watts": 75.0,
            "test_energy_per_patch_joules": 0.3,
        }

    def test_leaves_out_what_could_not_be_measured(self):
        reading = EnergyReading(seconds=4.0, gpu_joules=200.0, cpu_joules=None)

        metrics = reading_to_metrics("test", reading, n_items=100, item="scene")

        assert metrics == {"test_energy_gpu_joules": 200.0, "test_power_gpu_watts": 50.0}

    def test_a_single_item_still_gets_an_energy_per_item(self):
        reading = EnergyReading(seconds=1.0, gpu_joules=8.0, cpu_joules=4.0)

        metrics = reading_to_metrics("test", reading, n_items=1, item="scene")

        assert metrics["test_energy_per_scene_joules"] == 12.0

    def test_zero_items_has_no_energy_per_item_and_does_not_divide_by_zero(self):
        reading = EnergyReading(seconds=1.0, gpu_joules=8.0, cpu_joules=4.0)

        metrics = reading_to_metrics("test", reading, n_items=0, item="scene")

        assert "test_energy_per_scene_joules" not in metrics
        assert metrics["test_energy_total_joules"] == 12.0

    def test_no_reading_gives_no_metrics(self):
        assert reading_to_metrics("test", None, n_items=10, item="patch") == {}

    def test_every_metric_is_a_plain_float(self):
        reading = EnergyReading(seconds=4, gpu_joules=200, cpu_joules=100)

        metrics = reading_to_metrics("val", reading, n_items=10, item="patch")

        assert all(type(value) is float for value in metrics.values())


class TestIdlePower:
    def test_reports_average_watts_over_the_idle_window(self):
        slept = []
        watts = idle_power(
            seconds=10.0,
            readers={"gpu": _counter(0.0, 120.0), "cpu": _counter(0.0, 50.0)},
            clock=_Clock(0.0, 10.0),
            sleep=slept.append,
        )

        assert watts == {"gpu_watts": 12.0, "cpu_watts": 5.0}
        assert slept == [10.0]

    def test_the_default_idle_window_is_thirty_seconds(self):
        slept = []

        idle_power(
            readers={"gpu": _counter(0.0, 1.0), "cpu": _counter(0.0, 1.0)},
            clock=_Clock(0.0, 30.0),
            sleep=slept.append,
        )

        assert slept == [30.0]

    def test_an_unreadable_device_is_none(self):
        watts = idle_power(
            seconds=1.0,
            readers={"gpu": _counter(0.0, 3.0), "cpu": lambda: None},
            clock=_Clock(0.0, 1.0),
            sleep=lambda seconds: None,
        )

        assert watts == {"gpu_watts": 3.0, "cpu_watts": None}
