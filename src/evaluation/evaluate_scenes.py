"""Evaluate a trained checkpoint on complete scenes and save local score artifacts.

Usage: python -m evaluation.evaluate_scenes
"""

import json
import os
import sys
import time

import mlflow
import numpy as np
import pandas as pd
import torch

from data import patch_cache
from data.scene_manifest import load_scene_manifest
from evaluation.evaluate import (
    _CACHE_ROOT,
    _CHECKPOINT_DIR,
    _MLFLOW_EXPERIMENT,
    _REPO_ROOT,
    load_checkpoint,
)
from evaluation.paper_protocol import ALL_THRESHOLDS, AUPRC_GRIDS, auprc_by_grid
from evaluation.scene_inference import save_scene_counts, score_scenes
from evaluation.threshold_selection import (
    apply_threshold,
    oracle_threshold,
    select_with_widening,
)
from utils.power_meter import PowerMeter, reading_to_metrics

_PROJECT_ROOT = str(_REPO_ROOT)
_MLFLOW_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT}/mlflow.db"
_SCORES_DIR = os.path.join(_PROJECT_ROOT, "scene_scores")
_MODES = ("full_scene", "patches")
_BUCKET_GROUPS = ("strong", "weak", "plume_scenes", "all_scenes")

_RAW = "data/processed/starcop_raw/patches"
_MINI = "data/processed/starcop_mini"

EVAL_SOURCES = {
    "raw-full": {
        "dataset": "starcop_raw",
        "test": {
            "labels": "data/starcop_raw/test.csv",
            "patches": f"{_RAW}/test_tiled_128_128.csv",
        },
        "val": {
            "labels": "data/starcop_raw/train.csv",
            "patches": f"{_RAW}/val_tiled_128_128.csv",
        },
    },
    "mini": {
        "dataset": "starcop_mini",
        "test": {
            "labels": f"{_MINI}/splits/test.csv",
            "patches": f"{_MINI}/patches/test_tiled_128_128.csv",
        },
        "val": None,
    },
}


def build_run_name(
    architecture: str,
    checkpoint_tier: str,
    eval_tier: str,
    mode: str,
    device_override: str | None = None,
) -> str:
    """MLflow run name `<arch>-<checkpoint_tier>-on-<eval_tier>-scene-<mode>[-<device>]`."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    name = f"{architecture}-{checkpoint_tier}-on-{eval_tier}-scene-{mode}"
    return f"{name}-{device_override}" if device_override else name


def score_grid(name: str) -> np.ndarray:
    """`"union"`: every threshold the analyses need; `"single"`: the fixed 0.5 only."""
    if name == "union":
        return ALL_THRESHOLDS
    if name == "single":
        return np.array([0.5], dtype=np.float32)
    raise ValueError(f"thresholds must be 'union' or 'single', got {name!r}")


def summarize_test(counts, thresholds, buckets, *, chosen) -> dict:
    """JSON-ready summary of one scored test split.

    `chosen` is the `ThresholdChoice` picked on validation, or `None` for a fixed-threshold run.
    Always reports the fixed 0.5 threshold; with a choice it also applies it once, and computes
    the oracle (best threshold on this same test split) and the optimism gap
    `oracle - validation-picked` in pooled F1. AUPRC on the three grids needs the union grid.
    """

    pixels_per_scene = counts[:, 0, :].sum(axis=1)
    summary = {
        "n_scenes": int(counts.shape[0]),
        "pixels_per_scene": pixels_per_scene.tolist(),
        "pixel_total": int(pixels_per_scene.sum()),
        "at_0p5": apply_threshold(counts, thresholds, buckets, 0.5),
        "validation_choice": None,
        "at_validation_pick": None,
        "oracle_choice": None,
        "oracle_at_threshold": None,
        "optimism_gap_pooled_f1": None,
        "auprc": None,
    }
    if chosen is not None:
        picked = apply_threshold(counts, thresholds, buckets, chosen.threshold)
        oracle = oracle_threshold(counts, thresholds, buckets, objective=chosen.objective)
        oracle_applied = apply_threshold(counts, thresholds, buckets, oracle.threshold)
        summary["validation_choice"] = chosen._asdict()
        summary["at_validation_pick"] = picked
        summary["oracle_choice"] = oracle._asdict()
        summary["oracle_at_threshold"] = oracle_applied
        summary["optimism_gap_pooled_f1"] = (
            oracle_applied["metrics"]["all_scenes"]["f1"] - picked["metrics"]["all_scenes"]["f1"]
        )
    if all(np.isin(grid, thresholds).all() for grid in AUPRC_GRIDS.values()):
        summary["auprc"] = auprc_by_grid(counts, thresholds, buckets)
    return summary


def _flatten_applied(prefix: str, applied: dict, flat: dict) -> None:
    for group in _BUCKET_GROUPS:
        for name, value in applied["metrics"][group].items():
            flat[f"{prefix}_{group}_{name}"] = float(value)
    if applied["tile_fpr"] is not None:
        flat[f"{prefix}_tile_fpr"] = float(applied["tile_fpr"])
    for group in ("strong", "weak", "plume_scenes"):
        rate = applied["captured"][group]["rate"]
        if rate is not None:
            flat[f"{prefix}_captured_{group}"] = float(rate)


def flatten_metrics(summary: dict) -> dict[str, float]:
    """The scalars of `summarize_test`'s output, under MLflow-safe metric names."""
    flat = {"n_scenes": float(summary["n_scenes"]), "pixel_total": float(summary["pixel_total"])}
    _flatten_applied("at_0p5", summary["at_0p5"], flat)
    choice = summary["validation_choice"]
    if choice is not None:
        flat["val_pick_threshold"] = float(choice["threshold"])
        flat["val_pick_score"] = float(choice["score"])
        flat["val_pick_widenings"] = float(choice["n_widenings"])
        flat["val_pick_at_edge"] = float(choice["at_edge"])
        _flatten_applied("at_val_pick", summary["at_validation_pick"], flat)
        flat["oracle_threshold"] = float(summary["oracle_choice"]["threshold"])
        flat["oracle_pooled_f1"] = float(summary["oracle_choice"]["score"])
        flat["optimism_gap_pooled_f1"] = float(summary["optimism_gap_pooled_f1"])
    if summary["auprc"] is not None:
        for grid, populations in summary["auprc"].items():
            for population, value in populations.items():
                flat[f"auprc_{grid}_{population}"] = float(value)
    return flat


def _parse_kv_args(argv: list[str]) -> dict[str, str]:
    """`key=value` CLI args, as in `evaluate.py`."""
    return dict(arg.split("=", 1) for arg in argv if "=" in arg)


def _sync(device: str) -> None:
    """Wait for queued CUDA work, so a wall-clock timer measures work done, not work launched."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _timed_score(model, manifest, dataset, grid, device, batch_size, num_workers, **kwargs):
    """`(SceneCounts, wall-clock seconds, EnergyReading)` for one `score_scenes` call."""
    _sync(device)
    meter = PowerMeter()
    meter.start()
    start = time.perf_counter()
    result = score_scenes(
        model,
        manifest,
        thresholds=grid,
        dataset=dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs,
    )
    _sync(device)
    seconds = time.perf_counter() - start
    meter.stop()
    return result, seconds, meter.reading


def main() -> None:
    """CLI entry point -- see the module docstring."""
    args = _parse_kv_args(sys.argv[1:])
    architecture = args.get("architecture", "E2")
    checkpoint_tier = args.get("checkpoint_tier", "raw-full")
    eval_tier = args.get("eval_tier", "raw-full")
    mode = args.get("mode", "full_scene")
    threshold_mode = args.get("thresholds", "union")
    device_override = args.get("device")
    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(args.get("batch_size", 4 if mode == "full_scene" else 16))
    num_workers = int(args.get("num_workers", 4))

    os.chdir(_REPO_ROOT)
    source = EVAL_SOURCES[eval_tier]
    dataset = source["dataset"]
    run_name = build_run_name(architecture, checkpoint_tier, eval_tier, mode, device_override)
    grid = score_grid(threshold_mode)
    model = load_checkpoint(
        architecture, _CHECKPOINT_DIR / f"{architecture}-{checkpoint_tier}.pt", device
    )

    test_manifest = load_scene_manifest(source["test"]["labels"], source["test"]["patches"])
    kwargs = {"mode": mode}
    if mode == "patches":
        patches_df = pd.read_csv(source["test"]["patches"], low_memory=False)
        kwargs["patches_df"] = patches_df
        kwargs["cache_dir"] = patch_cache.resolve_cache_dir(
            _CACHE_ROOT, dataset, "test", patches_df
        )

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        test_result, test_seconds, test_energy = _timed_score(
            model, test_manifest, dataset, grid, device, batch_size, num_workers, **kwargs
        )
        val_result, val_seconds, chosen = None, None, None
        pick_on_validation = (
            mode == "full_scene" and threshold_mode == "union" and source["val"] is not None
        )
        if pick_on_validation:
            val_manifest = load_scene_manifest(source["val"]["labels"], source["val"]["patches"])
            val_result, val_seconds, _ = _timed_score(
                model, val_manifest, dataset, grid, device, batch_size, num_workers, mode=mode
            )

            def rescore(wider_grid):
                if np.isin(wider_grid, grid).all():
                    positions = [int(np.flatnonzero(grid == value)[0]) for value in wider_grid]
                    return val_result.counts[:, positions, :]
                return _timed_score(
                    model,
                    val_manifest,
                    dataset,
                    wider_grid,
                    device,
                    batch_size,
                    num_workers,
                    mode=mode,
                )[0].counts

            chosen = select_with_widening(rescore, val_manifest["bucket"].tolist())
            if not np.isin(np.float32(chosen.threshold), test_result.thresholds):
                extra = np.unique(np.append(test_result.thresholds, np.float32(chosen.threshold)))
                test_result, _, _ = _timed_score(
                    model, test_manifest, dataset, extra, device, batch_size, num_workers, **kwargs
                )

        summary = summarize_test(
            test_result.counts,
            test_result.thresholds,
            test_manifest["bucket"].tolist(),
            chosen=chosen,
        )
        summary.update(
            {
                "protocol": "paper",
                "mode": mode,
                "architecture": architecture,
                "checkpoint_tier": checkpoint_tier,
                "eval_tier": eval_tier,
                "device": device,
                "threshold_source": "validation_pooled_f1" if chosen else "fixed_0.5",
                "n_thresholds_scored": int(len(test_result.thresholds)),
                "test_wall_clock_seconds": test_seconds,
                "test_scenes_per_second": len(test_manifest) / test_seconds,
                "val_wall_clock_seconds": val_seconds,
                "test_energy": None if test_energy is None else test_energy._asdict(),
            }
        )
        mlflow.log_params(
            {
                "protocol": "paper",
                "architecture": architecture,
                "checkpoint_tier": checkpoint_tier,
                "eval_tier": eval_tier,
                "mode": mode,
                "device": device,
                "threshold_source": summary["threshold_source"],
                "n_thresholds_scored": summary["n_thresholds_scored"],
                "batch_size": batch_size,
                "num_workers": num_workers,
                "torch_num_threads": torch.get_num_threads(),
                "cpu_count": os.cpu_count(),
                "torch_version": torch.__version__,
            }
        )
        mlflow.log_metrics(
            {
                **flatten_metrics(summary),
                "test_wall_clock_seconds": test_seconds,
                "test_scenes_per_second": summary["test_scenes_per_second"],
                **({"val_wall_clock_seconds": val_seconds} if val_seconds is not None else {}),
                **reading_to_metrics("test", test_energy, len(test_manifest), "scene"),
            }
        )
        for split, result in (("test", test_result), ("val", val_result)):
            if result is None:
                continue
            path = os.path.join(_SCORES_DIR, f"{run_name}__{split}.npz")
            save_scene_counts(path, result)
            mlflow.log_artifact(path)
        mlflow.log_dict(json.loads(json.dumps(summary)), "summary.json")
        print(
            f"{run_name} [{device}]: {len(test_manifest)} scenes in {test_seconds:.1f}s "
            f"({summary['test_scenes_per_second']:.2f} scenes/s); "
            f"F1@0.5 all={summary['at_0p5']['metrics']['all_scenes']['f1']:.4f}"
            + (
                f"; val-picked t={chosen.threshold:.6f} "
                f"F1={summary['at_validation_pick']['metrics']['all_scenes']['f1']:.4f} "
                f"(oracle gap {summary['optimism_gap_pooled_f1']:+.4f})"
                if chosen
                else ""
            )
        )


if __name__ == "__main__":
    main()
