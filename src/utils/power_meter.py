"""Electric energy used by the CPU and the GPU over a timed region (GPU-vs-CPU comparison).

Inference comparisons between GPU and CPU require energy as well as speed. This measures, around
the same region the throughput timer covers:

- **GPU**: NVML's cumulative board-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, mJ), so
  the energy of a run is an exact before/after difference, not a sampled estimate. It covers the
  whole GPU board (chip, memory, fans' share of the board power), not the rest of the machine.
- **CPU**: Intel RAPL package energy (`/sys/class/powercap/intel-rapl:N/energy_uj`), summed over
  the packages (sub-domains such as `core` and `dram` are already inside the package figure).
  Linux makes these files readable by root only; when they are not readable the CPU figure is
  simply `None` rather than an error.

Both are **device** energies, not wall-socket energy: they leave out the PSU losses, the mainboard,
RAM outside the package, storage and the display. Anything else running on the machine is counted
too, so runs to be compared must be done on an otherwise idle machine, one at a time. Every reader
returns `None` when its counter cannot be read, and the meter never raises because of one.
"""

import glob
import json
import os
import sys
import time
from collections.abc import Callable
from typing import NamedTuple


class EnergyReading(NamedTuple):
    """Energy used over `seconds`: joules per device, `None` where it could not be measured."""

    seconds: float
    gpu_joules: float | None
    cpu_joules: float | None

    def _watts(self, joules: float | None) -> float | None:
        if joules is None or self.seconds <= 0:
            return None
        return joules / self.seconds

    @property
    def gpu_watts(self) -> float | None:
        """Average GPU board power over the region."""
        return self._watts(self.gpu_joules)

    @property
    def cpu_watts(self) -> float | None:
        """Average CPU package power over the region."""
        return self._watts(self.cpu_joules)

    @property
    def total_joules(self) -> float | None:
        """CPU + GPU energy; `None` unless both were measured (a partial sum would mislead)."""
        if self.gpu_joules is None or self.cpu_joules is None:
            return None
        return self.gpu_joules + self.cpu_joules

    @property
    def total_watts(self) -> float | None:
        """Average CPU + GPU power; `None` unless both were measured."""
        return self._watts(self.total_joules)


def _unavailable() -> float | None:
    return None


class RaplReader:
    """Cumulative CPU package energy in joules from the Linux powercap (RAPL) interface.

    Accumulates across counter wrap-arounds (the raw counter wraps at
    `max_energy_range_uj`, about 26 minutes at 170 W), assuming at most one wrap between two
    reads. Returns `None` when no package domain exists or any counter is unreadable.
    """

    def __init__(self, root: str = "/sys/class/powercap"):
        self._root = str(root)
        self._last: dict[str, int] = {}
        self._accumulated: dict[str, int] = {}

    def __call__(self) -> float | None:
        try:
            domains = sorted(
                path
                for path in glob.glob(os.path.join(self._root, "intel-rapl:*"))
                if os.path.basename(path).count(":") == 1
            )
            if not domains:
                return None
            for domain in domains:
                with open(os.path.join(domain, "energy_uj")) as counter:
                    raw = int(counter.read())
                with open(os.path.join(domain, "max_energy_range_uj")) as limit:
                    wrap = int(limit.read())
                if domain not in self._last:
                    self._accumulated[domain] = raw
                elif raw >= self._last[domain]:
                    self._accumulated[domain] += raw - self._last[domain]
                else:
                    self._accumulated[domain] += wrap - self._last[domain] + raw
                self._last[domain] = raw
            return sum(self._accumulated[domain] for domain in domains) / 1e6
        except (OSError, ValueError):
            return None


def nvml_energy_reader(index: int = 0, nvml=None) -> Callable[[], float | None]:
    """Reader of GPU `index`'s cumulative board energy in joules, `None`-returning if unsupported.

    `nvml` is the `pynvml` module (injectable for tests); it is imported lazily, so a machine
    without it or without an NVIDIA driver just gets a reader that reports nothing.
    """
    try:
        if nvml is None:
            import pynvml as nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
    except Exception:
        return _unavailable

    def read() -> float | None:
        try:
            return nvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000.0
        except Exception:
            return None

    return read


def _safe_read(reader: Callable[[], float | None]) -> float | None:
    try:
        return reader()
    except Exception:
        return None


class PowerMeter:
    """Measures the CPU and GPU energy between `start()` and `stop()` (or a `with` block).

    `readers` maps `"gpu"` and `"cpu"` to functions returning cumulative joules (default:
    NVML and RAPL); `clock` is the time source. The result is `reading`, an `EnergyReading`
    (`None` until stopped).
    """

    def __init__(self, readers: dict[str, Callable[[], float | None]] | None = None, clock=None):
        self._readers = readers or {"gpu": nvml_energy_reader(), "cpu": RaplReader()}
        self._clock = clock or time.perf_counter
        self._start: dict[str, float | None] | None = None
        self._started_at = 0.0
        self.reading: EnergyReading | None = None

    def start(self) -> None:
        """Take the starting counter values and the start time."""
        self.reading = None
        self._start = {name: _safe_read(reader) for name, reader in self._readers.items()}
        self._started_at = self._clock()

    def stop(self) -> None:
        """Take the ending values and compute `reading`."""
        if self._start is None:
            raise RuntimeError("meter not started")
        seconds = self._clock() - self._started_at
        end = {name: _safe_read(reader) for name, reader in self._readers.items()}

        def used(name: str) -> float | None:
            if self._start[name] is None or end[name] is None:
                return None
            return end[name] - self._start[name]

        self.reading = EnergyReading(seconds, used("gpu"), used("cpu"))

    def __enter__(self) -> "PowerMeter":
        self.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        self.stop()
        return False


def reading_to_metrics(
    prefix: str, reading: EnergyReading | None, n_items: int, item: str
) -> dict[str, float]:
    """MLflow-ready metrics (`<prefix>_energy_gpu_joules`, ...) for one measured region.

    Only what was measured is included. `energy_per_<item>_joules` is the CPU + GPU energy per
    processed `item` (patch or scene).
    """
    if reading is None:
        return {}
    candidates = {
        f"{prefix}_energy_gpu_joules": reading.gpu_joules,
        f"{prefix}_energy_cpu_joules": reading.cpu_joules,
        f"{prefix}_energy_total_joules": reading.total_joules,
        f"{prefix}_power_gpu_watts": reading.gpu_watts,
        f"{prefix}_power_cpu_watts": reading.cpu_watts,
        f"{prefix}_power_total_watts": reading.total_watts,
    }
    if reading.total_joules is not None and n_items > 0:
        candidates[f"{prefix}_energy_per_{item}_joules"] = reading.total_joules / n_items
    return {name: float(value) for name, value in candidates.items() if value is not None}


def idle_power(seconds: float = 30.0, readers=None, clock=None, sleep=time.sleep) -> dict:
    """Average CPU and GPU power with nothing running, over `seconds`: the baseline to subtract."""
    meter = PowerMeter(readers=readers, clock=clock)
    meter.start()
    sleep(seconds)
    meter.stop()
    return {"gpu_watts": meter.reading.gpu_watts, "cpu_watts": meter.reading.cpu_watts}


if __name__ == "__main__":
    print(json.dumps(idle_power(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)))
