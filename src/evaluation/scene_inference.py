"""Per-scene confusion counts for the paper-protocol evaluation.

The GPU runs **once per (checkpoint, split, mode)** and produces, for every scene, the
`[tp, fp, fn, tn]` pixel counts at every threshold of a grid. Everything downstream (the papers'
strong/weak F1, tile FPR, AUPRC on any grid, the validation-picked threshold) is then offline
arithmetic over these integer counts.

Two modes:
- `"full_scene"` runs the model directly on each whole scene (512x512 in this project), which is
  how the STARCOP paper scores its test set. This is the reported result.
- `"patches"` scores the scene's existing 128x128 patches one by one and sums their counts per
  scene. **A diagnostic only**: it reproduces this project's older patch-pooled protocol broken
  down by scene, so it can be checked against `evaluate.py`. The patches overlap (stride 64), so a
  pixel is counted once per patch that contains it and a scene's pixel total is
  `n_patches * 128 * 128`, not 512 * 512.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import PatchDataset
from evaluation.metrics import sweep_confusion_counts

SCENE_MODES = ("full_scene", "patches")


class SceneCounts(NamedTuple):
    """Per-scene confusion counts for a grid of thresholds.

    `counts` has shape `(n_scenes, n_thresholds, 4)` and dtype int64, the last axis in the
    order `[tp, fp, fn, tn]`; `scene_ids` and `thresholds` label its first two axes.
    """

    counts: np.ndarray
    scene_ids: list[str]
    thresholds: np.ndarray
    mode: str


def _threshold_tensor(thresholds) -> torch.Tensor:
    """1-D float32 tensor from a list, array or tensor of thresholds (at least one)."""
    tensor = torch.as_tensor(thresholds, dtype=torch.float32).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("at least one threshold is required")
    return tensor


def _loader(windows_df, dataset, device, batch_size, num_workers, cache_dir=None) -> DataLoader:
    """Ordered, non-augmenting loader over the windows of `windows_df` (scenes or patches).

    Order matters: callers pair batches back to `windows_df` rows by position, so nothing is
    shuffled. Every window is read exactly once, so the RAM cache is switched off (it would only
    hold memory).

    Worker tensors are shared through named shared-memory files (`file_system`), not the default
    file-descriptor passing: with 512x512 scenes and `pin_memory`, the fd handshake between a worker
    and the pin-memory thread deadlocked intermittently (about 1 run in 8, both sides parked in a
    futex with nothing logged).
    """
    torch.multiprocessing.set_sharing_strategy("file_system")

    windows = PatchDataset(
        windows_df,
        dataset=dataset,
        augment=False,
        max_cache_bytes=0,
        cache_dir=cache_dir,
    )
    return DataLoader(
        windows,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )


def _full_scene_counts(model, scene_df, dataset, thresholds, device, batch_size, num_workers):
    """One `(n_thresholds, 4)` float64 tensor per scene, running the model on whole scenes."""
    loader = _loader(scene_df, dataset, device, batch_size, num_workers)
    per_scene = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["input"].to(device)))
            targets = batch["output"].to(device)

            for index in range(probs.shape[0]):
                sweep = sweep_confusion_counts(probs[index], targets[index], thresholds)
                per_scene.append(sweep.double().cpu())
    return per_scene


def _patch_counts_by_scene(
    model, scene_ids, patches_df, dataset, thresholds, device, batch_size, num_workers, cache_dir
):
    """Per-scene sums of per-patch sweeps, in the order of `scene_ids`."""
    subset = patches_df[patches_df["id_original"].isin(scene_ids)]
    missing = [scene_id for scene_id in scene_ids if scene_id not in set(subset["id_original"])]
    if missing:
        raise ValueError(f"no patches for scene(s): {missing[:5]}")
    loader = _loader(subset, dataset, device, batch_size, num_workers, cache_dir)
    parent_of_row = subset["id_original"].tolist()

    totals = {
        scene_id: torch.zeros(len(thresholds), 4, dtype=torch.float64) for scene_id in scene_ids
    }

    row = 0
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["input"].to(device)))
            targets = batch["output"].to(device)
            for index in range(probs.shape[0]):
                sweep = sweep_confusion_counts(probs[index], targets[index], thresholds)
                totals[parent_of_row[row]] += sweep.double().cpu()
                row += 1
    return [totals[scene_id] for scene_id in scene_ids]


def score_scenes(
    model: torch.nn.Module,
    scene_df: pd.DataFrame,
    *,
    mode: str,
    thresholds,
    dataset: str,
    device: str = "cpu",
    batch_size: int = 4,
    patches_df: pd.DataFrame | None = None,
    cache_dir: Path | None = None,
    num_workers: int = 0,
) -> SceneCounts:
    """Score every scene of `scene_df` at every threshold; see the module docstring for the modes.

    `scene_df` is a scene manifest (`scene_manifest.build_scene_manifest`): one row per scene with
    `scene_id`, `folder` and the window columns `PatchDataset` reads. `mode="patches"` also needs
    `patches_df`, the patch manifest whose `id_original` names each patch's parent scene, and
    accepts an optional `cache_dir` -- which must match the patch manifest restricted to the
    requested scenes (`PatchDataset` rejects a mismatch). `dataset` selects the band contract
    (`starcop_raw` / `starcop_mini`). A pixel is predicted positive when its probability is
    strictly greater than the threshold, as in `evaluate.py`. `folder` paths are used as given:
    the real manifests hold repo-root-relative paths, so run from the repo root.

    The model is switched to eval mode and run without gradients. The counts are exact integers.
    """
    if mode not in SCENE_MODES:
        raise ValueError(f"mode must be one of {SCENE_MODES}, got {mode!r}")
    if mode == "patches" and patches_df is None:
        raise ValueError("mode='patches' needs patches_df, the patch manifest")
    threshold_tensor = _threshold_tensor(thresholds)
    scene_ids = scene_df["scene_id"].tolist()
    device_thresholds = threshold_tensor.to(device)

    model.eval()
    if mode == "full_scene":
        per_scene = _full_scene_counts(
            model, scene_df, dataset, device_thresholds, device, batch_size, num_workers
        )
    else:
        per_scene = _patch_counts_by_scene(
            model,
            scene_ids,
            patches_df,
            dataset,
            device_thresholds,
            device,
            batch_size,
            num_workers,
            cache_dir,
        )
    counts = np.rint(torch.stack(per_scene).numpy()).astype(np.int64)
    return SceneCounts(
        counts=counts,
        scene_ids=scene_ids,
        thresholds=threshold_tensor.numpy(),
        mode=mode,
    )


def save_scene_counts(path: Path, result: SceneCounts) -> None:
    """Write `result` to a compressed-free `.npz` at `path` (parent directories are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        counts=result.counts,
        scene_ids=np.array(result.scene_ids),
        thresholds=result.thresholds,
        mode=np.array(result.mode),
    )


def load_scene_counts(path: Path) -> SceneCounts:
    """Inverse of `save_scene_counts`."""

    with np.load(Path(path), allow_pickle=False) as data:
        return SceneCounts(
            counts=data["counts"],
            scene_ids=data["scene_ids"].tolist(),
            thresholds=data["thresholds"],
            mode=str(data["mode"]),
        )
