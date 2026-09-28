"""Exploratory analysis and example figures for the local datasets."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.ticker import FuncFormatter
from rasterio.windows import Window

from visualization.class_balance import PARTITIONS, build_class_balance
from visualization.figure_common import (
    FIGURES_DIR,
    IEEE_FULL_WIDTH_INCHES,
    TIER_COLORS,
    apply_ieee_style,
    format_decimal_ptbr,
    format_general_ptbr,
    format_integer_ptbr,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARTITION_LABELS = {"train": "Treino", "val": "Validação", "test": "Teste"}

BALANCE_PANELS = (
    ("pixel_fraction", "(a) Pixels de pluma (%)"),
    ("patch_fraction", "(b) Recortes com pluma (%)"),
)

SCALE_COLORS = {
    "mini": TIER_COLORS["mini"],
    "R2": TIER_COLORS["r2"],
    "R3": TIER_COLORS["raw-full"],
}
EXAMPLE_ROW_LABELS = ("Composição RGB", "mag1c", "Rótulo")


MAG1C_COLOR_MAX = 2000
MAG1C_LABEL = "mag1c (ppm·m)"


def compute_patch_level_balance(patches_df: pd.DataFrame) -> dict:
    """Return {positive, total, positive_fraction} counting patches flagged `has_plume`.

    STARCOP flags a patch when it has 40 plume pixels or more (`frac_positives > 0.00244`), not
    at the first positive pixel. Distinct from stats.py's pixel-level class distribution: a
    flagged patch counts fully here, not fractionally.
    """
    total = len(patches_df)

    positive = int(patches_df["has_plume"].sum()) if total else 0
    return {
        "positive": positive,
        "total": total,
        "positive_fraction": (positive / total) if total else 0.0,
    }


def select_example_patches(
    patches_df: pd.DataFrame,
    n_positive: int,
    n_negative: int,
    seed: int,
) -> pd.DataFrame:
    """Pick `n_positive` + `n_negative` example patches for qualitative visualization.

    One positive slot is always the faintest plume available (lowest
    `frac_positives` among `has_plume` rows) -- a hard/ambiguous example,
    chosen deterministically rather than left to
    chance. Remaining slots are sampled with `seed` for reproducibility.
    """
    positives = patches_df[patches_df["has_plume"]]
    negatives = patches_df[~patches_df["has_plume"]]
    if len(positives) < n_positive:
        raise ValueError(f"Not enough positive patches: need {n_positive}, have {len(positives)}")
    if len(negatives) < n_negative:
        raise ValueError(f"Not enough negative patches: need {n_negative}, have {len(negatives)}")

    hardest = positives.loc[[positives["frac_positives"].idxmin()]]
    remaining_positive = positives.drop(index=hardest.index)
    extra_positive = remaining_positive.sample(n=n_positive - 1, random_state=seed)
    chosen_negative = negatives.sample(n=n_negative, random_state=seed)

    return pd.concat([hardest, extra_positive, chosen_negative], ignore_index=True)


def read_patch_bands(scene_folder: Path, window: Window, bands: list[str]) -> dict[str, np.ndarray]:
    """Read `window` from each of `bands`' GeoTIFFs under `scene_folder`."""
    result = {}
    for band in bands:
        with rasterio.open(scene_folder / f"{band}.tif") as src:
            result[band] = src.read(1, window=window)
    return result


def rgb_composite(bands: dict[str, np.ndarray], r_key: str, g_key: str, b_key: str) -> np.ndarray:
    """Stack three bands into an (H, W, 3) array, each independently scaled to [0, 1].

    Per-band min/max scaling (not a shared scale) so each channel uses its
    own dynamic range -- appropriate here since this is a qualitative
    visualization aid, not the model's actual normalization contract
    (Use STARCOP's fixed offset/factor/clip table for that).
    A constant band (max == min) maps to all-zero rather than dividing by
    zero.
    """

    def _scale(band: np.ndarray) -> np.ndarray:
        band_min, band_max = float(band.min()), float(band.max())
        span = band_max - band_min
        if span == 0:
            return np.zeros_like(band, dtype="float32")
        return ((band - band_min) / span).astype("float32")

    return np.stack([_scale(bands[r_key]), _scale(bands[g_key]), _scale(bands[b_key])], axis=-1)


def partition_series(balance: dict, key: str) -> dict[str, list[float]]:
    """Per scale, the percentage of `key` (a fraction of `class_balance`) for train, val, test."""
    return {
        scale: [by_partition[partition][key] * 100 for partition in PARTITIONS]
        for scale, by_partition in balance.items()
    }


def example_plume_pixels(frac_positives: float, width: int, height: int) -> int:
    """Plume pixels in a `width` x `height` patch, from its `frac_positives` (nearest pixel)."""
    return round(frac_positives * width * height)


def example_column_title(scale: str, has_plume: bool, plume_pixels: int) -> str:
    """Two-line column title for an example patch: its scale and plume, then the pixel count."""
    kind = "com pluma" if has_plume else "sem pluma"
    return f"{scale}, {kind}\n{plume_pixels} pixels"


def plot_class_balance(balance: dict, out_path: Path) -> None:
    """Save the two-panel plume-balance figure: (a) pixels counted once, (b) flagged patches.

    `balance` is `class_balance.build_class_balance`'s result: scale -> partition -> fractions.
    """
    apply_ieee_style()
    scales = list(balance)
    x = np.arange(len(PARTITIONS))
    width = 0.8 / len(scales)
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_FULL_WIDTH_INCHES, 2.4))
    for ax, (key, title) in zip(axes, BALANCE_PANELS):
        series = partition_series(balance, key)
        decimals = 2 if key == "pixel_fraction" else 1
        for i, scale in enumerate(scales):
            bars = ax.bar(
                x + i * width, series[scale], width, color=SCALE_COLORS[scale], label=scale
            )
            for bar, value in zip(bars, series[scale]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    format_decimal_ptbr(value, decimals),
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                )
        ax.set_xticks(x + width * (len(scales) - 1) / 2)
        ax.set_xticklabels([PARTITION_LABELS[p] for p in PARTITIONS])
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(max(values) for values in series.values()) * 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: format_general_ptbr(v)))
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
    axes[0].legend(title="Escala", frameon=False, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_example_patches(
    examples: pd.DataFrame,
    input_products: list[str],
    rgb_keys: tuple[str, str, str],
    mag1c_key: str,
    output_product: str,
    out_path: Path,
) -> None:
    """Render one column per example: RGB composite, mag1c heatmap and ground-truth mask.

    `examples` needs a `scale` column (`mini`, `R2`, `R3`) besides the patch-table columns. Every
    mag1c panel shares one colour scale (0 to `MAG1C_COLOR_MAX`) with a single colour bar.
    """
    n = len(examples)
    apply_ieee_style()

    fig, axes = plt.subplots(
        3,
        n + 1,
        figsize=(IEEE_FULL_WIDTH_INCHES, 4.4),
        gridspec_kw={"width_ratios": [1] * n + [0.06]},
    )
    mag1c_image = None
    for col_idx, (_, row) in enumerate(examples.iterrows()):
        window = Window(
            col_off=row["window_col_off"],
            row_off=row["window_row_off"],
            width=row["window_width"],
            height=row["window_height"],
        )
        bands = read_patch_bands(Path(row["folder"]), window, [*input_products, output_product])
        pixels = example_plume_pixels(
            row["frac_positives"], row["window_width"], row["window_height"]
        )
        axes[0, col_idx].imshow(rgb_composite(bands, *rgb_keys))
        mag1c_image = axes[1, col_idx].imshow(
            bands[mag1c_key], cmap="viridis", vmin=0, vmax=MAG1C_COLOR_MAX
        )
        axes[2, col_idx].imshow(
            bands[output_product], cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        axes[0, col_idx].set_title(
            example_column_title(row["scale"], bool(row["has_plume"]), pixels), fontsize=7
        )
    for row_idx, label in enumerate(EXAMPLE_ROW_LABELS):
        axes[row_idx, 0].set_ylabel(label)
    for ax in axes[:, :n].flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, n].axis("off")
    axes[2, n].axis("off")
    colorbar = fig.colorbar(mag1c_image, cax=axes[1, n])
    colorbar.set_label(MAG1C_LABEL)
    colorbar.set_ticks([0, 500, 1000, 1500, 2000])
    colorbar.ax.set_yticklabels([format_integer_ptbr(v) for v in (0, 500, 1000, 1500, 2000)])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


INPUT_PRODUCTS = ["mag1c", "TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm"]
RGB_KEYS = ("TOA_AVIRIS_640nm", "TOA_AVIRIS_550nm", "TOA_AVIRIS_460nm")
OUTPUT_PRODUCT = "labelbinary"


def _patches_root(dataset: str) -> Path:
    """Resolve the repo-root-relative patches directory for `dataset`."""
    return _PROJECT_ROOT / "data" / "processed" / dataset / "patches"


def main() -> None:
    """Write `balanco_classes.png` and `exemplos_dados.png`; run from the repository root."""
    here = _PROJECT_ROOT
    balance = build_class_balance(
        here / "data" / "manifests" / "r2_manifest_train.csv",
        here / "data" / "manifests" / "r2_manifest_val.csv",
    )
    plot_class_balance(balance, FIGURES_DIR / "balanco_classes.png")

    mini_train = pd.read_csv(_patches_root("starcop_mini") / "train_tiled_128_128.csv")
    mini_examples = select_example_patches(mini_train, n_positive=2, n_negative=2, seed=42)
    raw_train = pd.read_csv(_patches_root("starcop_raw") / "train_tiled_128_128.csv")
    raw_examples = select_example_patches(raw_train, n_positive=1, n_negative=0, seed=42)
    examples = pd.concat(
        [mini_examples.assign(scale="mini"), raw_examples.assign(scale="R3")], ignore_index=True
    )
    plot_example_patches(
        examples,
        INPUT_PRODUCTS,
        RGB_KEYS,
        "mag1c",
        OUTPUT_PRODUCT,
        FIGURES_DIR / "exemplos_dados.png",
    )


if __name__ == "__main__":
    main()
