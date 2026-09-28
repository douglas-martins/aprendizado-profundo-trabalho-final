"""Cost-versus-quality figure: (a) PR-AUC on the real test
split against parameter count, one line per training tier; (b) inference energy per 512x512
scene on GPU and on CPU.

E3 has 12.9% of E2's parameters and uses approximately 3.5x less energy per scene on CPU.
This figure puts those two measurements side by side.

Panel (a) scores every point on the **same** split, `starcop_raw`'s 16,758-patch test set
(`mini` through the cross-tier evaluation), because PR-AUC is only comparable on one split.
Parameter counts live on the *training* runs (`param_count`); the evaluation runs do not log
them. Panel (b) uses E1 as trained on `mini`, E2 and E3 as trained at full scale (R3).
Energy is *device* energy (GPU board + CPU package), not
wall-socket, one execution per cell.

`pr_auc_points`, `energy_rows`, `energy_ratio` and `load_param_count` are the tested logic;
`plot_efficiency` (matplotlib) is thin glue, exercised by running `main()` and inspecting
`figures/efficiency_tradeoff.png`.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.figure_common import (
    FIGURES_DIR,
    IEEE_FULL_WIDTH_INCHES,
    TIER_COLORS,
    TIER_LABELS,
    TIER_MARKERS,
    apply_ieee_style,
    format_decimal_ptbr,
    format_general_ptbr,
    format_integer_ptbr,
    latest_metric,
    open_client,
    resolve_run_id,
)

PR_AUC_RUN_NAMES = {
    "mini": {
        "E1": "E1-mini-on-raw-full-eval",
        "E2": "E2-mini-on-raw-full-eval",
        "E3": "E3-mini-on-raw-full-eval",
    },
    "r2": {"E1": "E1-r2-eval", "E2": "E2-r2-eval", "E3": "E3-r2-eval"},
    "raw-full": {"E2": "E2-raw-full-eval", "E3": "E3-raw-full-eval"},
}


PARAMETER_RUN_NAMES = {"E1": "E1-mini", "E2": "E2-mini", "E3": "E3-mini"}


ENERGY_RUN_NAMES = {
    "E1": {
        "tier": "mini",
        "cuda": "E1-mini-on-raw-full-scene-full_scene-cuda",
        "cpu": "E1-mini-on-raw-full-scene-full_scene-cpu",
    },
    "E2": {
        "tier": "raw-full",
        "cuda": "E2-raw-full-on-raw-full-scene-full_scene-cuda",
        "cpu": "E2-raw-full-on-raw-full-scene-full_scene-cpu",
    },
    "E3": {
        "tier": "raw-full",
        "cuda": "E3-raw-full-on-raw-full-scene-full_scene-cuda",
        "cpu": "E3-raw-full-on-raw-full-scene-full_scene-cpu",
    },
}


@dataclass(frozen=True)
class PrAucPoint:
    """One trained model: its size and its PR-AUC on the real test split."""

    tier: str
    architecture: str
    parameters: int
    pr_auc: float


@dataclass(frozen=True)
class EnergyRow:
    """Device energy per scene of one model on GPU and on CPU, in joules."""

    architecture: str
    tier: str
    gpu_joules: float
    cpu_joules: float

    @property
    def ratio(self) -> float:
        """How many times more energy per scene the CPU run spent."""
        return energy_ratio(self.cpu_joules, self.gpu_joules)


def energy_ratio(cpu_joules: float, gpu_joules: float) -> float:
    """CPU energy divided by GPU energy; refuses a GPU energy that is not positive."""
    if gpu_joules <= 0:
        raise ValueError(f"GPU energy must be positive, got {gpu_joules}")
    return cpu_joules / gpu_joules


def pr_auc_points(
    pr_auc_by_tier: dict[str, dict[str, float]], parameters_by_architecture: dict[str, int]
) -> list[PrAucPoint]:
    """One point per (tier, architecture) present, tiers in the given order, each tier's
    points ascending by parameter count so a line through them runs left to right.

    A tier that lacks an architecture (R3 has no E1) simply has no point for it. Raises
    `KeyError` naming an architecture whose parameter count is unknown.
    """
    points = []
    for tier, pr_auc_by_architecture in pr_auc_by_tier.items():
        tier_points = [
            PrAucPoint(tier, architecture, parameters_by_architecture[architecture], pr_auc)
            for architecture, pr_auc in pr_auc_by_architecture.items()
        ]
        points.extend(sorted(tier_points, key=lambda point: point.parameters))
    return points


def energy_rows(energy_by_architecture: dict[str, dict]) -> list[EnergyRow]:
    """One `EnergyRow` per model, in the order given."""
    return [
        EnergyRow(
            architecture,
            values["tier"],
            gpu_joules=values["gpu_joules"],
            cpu_joules=values["cpu_joules"],
        )
        for architecture, values in energy_by_architecture.items()
    ]


def load_param_count(client, run_id: str) -> int:
    """The `param_count` MLflow parameter of `run_id` (stored as a string) as an integer."""
    return int(client.get_run(run_id).data.params["param_count"])


def plot_efficiency(points: list[PrAucPoint], rows: list[EnergyRow], out_path: Path) -> None:
    """Save the two-panel figure: PR-AUC against parameters (log x), and energy per scene."""
    apply_ieee_style()
    fig, (size_axis, energy_axis) = plt.subplots(
        1, 2, figsize=(IEEE_FULL_WIDTH_INCHES, 2.9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for tier in dict.fromkeys(point.tier for point in points):
        tier_points = [point for point in points if point.tier == tier]
        size_axis.plot(
            [point.parameters for point in tier_points],
            [point.pr_auc for point in tier_points],
            color=TIER_COLORS[tier],
            marker=TIER_MARKERS[tier],
            markersize=5,
            label=TIER_LABELS[tier],
        )
    size_axis.set_xscale("log")
    parameters = sorted({(point.parameters, point.architecture) for point in points})
    size_axis.set_xticks(
        [count for count, _ in parameters],
        [f"{architecture}\n{format_integer_ptbr(count)}" for count, architecture in parameters],
    )
    size_axis.minorticks_off()
    size_axis.set_ylim(0, 0.6)
    size_axis.set_xlabel("Parâmetros (escala logarítmica)")
    size_axis.set_ylabel("PR-AUC no teste real")
    size_axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_general_ptbr(v)))
    size_axis.grid(alpha=0.3, linewidth=0.4)
    size_axis.legend(title="camada de treino", loc="lower right", frameon=False)

    positions = range(len(rows))
    width = 0.36
    energy_axis.bar(
        [p - width / 2 for p in positions],
        [row.gpu_joules for row in rows],
        width,
        color="0.75",
        edgecolor="black",
        linewidth=0.5,
        label="GPU",
    )
    energy_axis.bar(
        [p + width / 2 for p in positions],
        [row.cpu_joules for row in rows],
        width,
        color="0.35",
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        label="CPU",
    )
    for position, row in zip(positions, rows, strict=True):
        energy_axis.text(
            position + width / 2,
            row.cpu_joules,
            f"{format_decimal_ptbr(row.ratio, 1)}×",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    energy_axis.set_xticks(
        list(positions), [f"{row.architecture} ({TIER_LABELS[row.tier]})" for row in rows]
    )
    energy_axis.set_ylim(0, max(row.cpu_joules for row in rows) * 1.15)
    energy_axis.set_ylabel("Energia por cena (J)")
    energy_axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_general_ptbr(v)))
    energy_axis.grid(axis="y", alpha=0.3, linewidth=0.4)
    energy_axis.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    """Read the runs from `mlflow.db` and write `efficiency_tradeoff.png`."""
    client = open_client()
    parameters = {
        architecture: load_param_count(client, resolve_run_id(client, run_name))
        for architecture, run_name in PARAMETER_RUN_NAMES.items()
    }
    pr_auc_by_tier = {
        tier: {
            architecture: latest_metric(client, resolve_run_id(client, run_name), "test_pr_auc")
            for architecture, run_name in by_architecture.items()
        }
        for tier, by_architecture in PR_AUC_RUN_NAMES.items()
    }
    energy = {
        architecture: {
            "tier": runs["tier"],
            "gpu_joules": latest_metric(
                client, resolve_run_id(client, runs["cuda"]), "test_energy_per_scene_joules"
            ),
            "cpu_joules": latest_metric(
                client, resolve_run_id(client, runs["cpu"]), "test_energy_per_scene_joules"
            ),
        }
        for architecture, runs in ENERGY_RUN_NAMES.items()
    }
    points = pr_auc_points(pr_auc_by_tier, parameters)
    rows = energy_rows(energy)
    for point in points:
        print(
            f"{point.tier:9s} {point.architecture} {point.parameters:>9,d} params  "
            f"PR-AUC {point.pr_auc:.4f}"
        )
    for row in rows:
        print(
            f"{row.architecture} ({row.tier}) GPU {row.gpu_joules:.3f} J  "
            f"CPU {row.cpu_joules:.3f} J  ratio {row.ratio:.2f}x"
        )
    out_path = FIGURES_DIR / "efficiency_tradeoff.png"
    plot_efficiency(points, rows, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
