"""Train the segmentation models and record runs in the local MLflow store.

The SQLite backend is explicit so the experiment history remains isolated in
this project and can be read without starting a new run.
"""

import hashlib
import os
import sys
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Sampler

from data import patch_cache
from data.dataset import PatchDataset
from models.architectures import build_e1, build_e2, build_e3
from models.losses import build_loss, compute_pos_weight
from training.early_stopping import EarlyStopper

_DEFAULT_SEED = 42


_MONITOR_METRIC_KEYS = {"val_loss": ("loss", "min"), "val_f1": ("f1", "max")}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_CACHE_ROOT = _PROJECT_ROOT / "patch_cache"
_STATE_DIR = _PROJECT_ROOT / "run_state"
_MLFLOW_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"
_MLFLOW_EXPERIMENT = "dl-final-project"

_ARCHITECTURE_BUILDERS = {"E1": build_e1, "E2": build_e2, "E3": build_e3}


def build_model(architecture: str) -> torch.nn.Module:
    """Build E1/E2/E3 by name -- E2/E3 pretrained, E1 (no `pretrained` param) is not."""
    build = _ARCHITECTURE_BUILDERS[architecture]
    return build() if architecture == "E1" else build(pretrained=True)


def augment_for(architecture: str) -> bool:
    """Whether `architecture` trains with augmentation -- E1 is the deliberately plain baseline
    (no pretraining, no augmentation, minimal regularization); E2/E3 use it."""
    return architecture != "E1"


def set_seed(seed: int) -> None:
    """Seed every RNG `fit()` draws from -- torch (CPU + CUDA) and numpy.

    `train.py` seeded nothing before this, so
    two runs of the same configuration (weight init via the caller's own
    pre-`fit()` seeding, kornia augmentation, `DataLoader` shuffling --
    all of which read from these same global generators) could not be
    reproduced. Call this before building the model too, not only inside
    `fit()`, so weight initialization is also covered.

    Also forces `cudnn.deterministic=True` / `cudnn.benchmark=False` --
    seeding the RNGs alone is NOT sufficient for reproducible results on
    CUDA. A reproducibility check with a full 50-epoch E1/mini
    rerun diverged from its recorded result, best_epoch 47->50 and
    val_loss 0.0142->0.0120, a ~16% difference, not noise. Without this,
    cuDNN can pick a different convolution algorithm run to run even with
    identical seeds, and that compounds over many steps into materially
    different trajectories. Per torch's own reproducibility notes
    (the PyTorch reproducibility guidance).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Return the total element count across all of `model`'s parameter tensors.

    Logged as the `param_count` MLflow param so parameter-count column doesn't
    have to be recomputed or retyped from `architectures.py`'s own docstrings later.
    """
    return sum(p.numel() for p in model.parameters())


def library_versions() -> dict[str, str]:
    """Return `{torch_version, smp_version, cuda_version}` for MLflow params.

    "record exact library versions, not just 'torch' without a version pin"
    read directly from the imported modules so this can't drift out of sync with what actually ran.
    `cuda_version` is `"cpu"` on a build with no CUDA support.
    """
    return {
        "torch_version": torch.__version__,
        "smp_version": smp.__version__,
        "cuda_version": torch.version.cuda or "cpu",
    }


def _seed_worker(worker_id: int, base_seed: int) -> None:
    """`DataLoader` `worker_init_fn`: reseed numpy/torch per worker off `base_seed`.

    Each worker process needs its own deterministic-but-distinct seed --
    without this, workers otherwise fall back to arbitrary per-process RNG
    state, and augmentation draws would no longer be reproducible under
    `num_workers>0` even with the main process's RNGs seeded.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _loader_kwargs(device: str, num_workers: int, seed: int, *, persistent_workers: bool) -> dict:
    """Build one DataLoader's kwargs: `pin_memory` plus, when `num_workers>0`, the worker knobs.

    `pin_memory` is set whenever `device` is CUDA, independent of
    `num_workers` -- pinning host memory speeds up the host->device copy
    itself and costs nothing when there's no multiprocessing worker
    involved at all.

    `persistent_workers` is the caller's choice per loader, never turned on
    when `num_workers=0` (torch raises a `ValueError` if it is) -- see
    `fit()`'s own docstring for why its train and val loaders pass
    different values here.
    """
    kwargs: dict = {"pin_memory": device.startswith("cuda")}
    if num_workers > 0:
        kwargs["num_workers"] = num_workers
        kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=seed)
        if persistent_workers:
            kwargs["persistent_workers"] = True
    return kwargs


class EpochShuffleSampler(Sampler[int]):
    """Shuffled order that is a pure function of `(seed, epoch)`.

    A `DataLoader(shuffle=True, generator=...)` draws its permutation from a
    generator whose consumption depends on how many iterators the loader has
    already created, so a *fresh* loader after a crash could not reproduce the
    order an uninterrupted run would have used. `set_epoch()` (called by
    `fit()` before each epoch) is all the state this needs, and it never touches
    the global RNG.
    """

    def __init__(self, num_samples: int, seed: int):
        """Shuffle `range(num_samples)` with permutations derived from `seed`."""
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select which epoch's permutation the next iteration yields."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        """Yield every index once, in this `(seed, epoch)`'s order."""
        entropy = np.random.SeedSequence([self.seed, self.epoch]).generate_state(1, dtype=np.uint64)
        generator = torch.Generator()
        generator.manual_seed(int(entropy[0]))
        yield from torch.randperm(self.num_samples, generator=generator).tolist()

    def __len__(self) -> int:
        """Number of indices per epoch."""
        return self.num_samples


_FINGERPRINT_COLUMNS = [
    "folder",
    "window_col_off",
    "window_row_off",
    "window_width",
    "window_height",
]


def train_set_fingerprint(train_df: pd.DataFrame) -> str:
    """Hash of which patches (folder + window, in order) make up `train_df`."""
    hashed = pd.util.hash_pandas_object(train_df[_FINGERPRINT_COLUMNS], index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def run_signature(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    *,
    dataset: str,
    lr: float,
    batch_size: int,
    augment: bool,
    monitor: str,
    seed: int,
    pos_weight: float,
) -> dict:
    """Everything that must be identical for a resumed run to continue the same trajectory.

    `patience`, `max_epochs`, `max_steps`, `num_workers` and `device` are deliberately
    absent: none of them changes what an epoch computes (a resumed run may extend
    `max_epochs` or raise `patience`).
    """
    return {
        "dataset": dataset,
        "lr": lr,
        "batch_size": batch_size,
        "augment": augment,
        "monitor": monitor,
        "seed": seed,
        "train_fingerprint": train_set_fingerprint(train_df),
        "pos_weight": float(pos_weight),
        "param_count": count_parameters(model),
    }


def _atomic_save(obj, path: Path) -> None:
    """`torch.save` to `path` so that a kill mid-write never leaves a truncated file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".tmp")
    torch.save(obj, partial_path)
    os.replace(partial_path, path)


def _capture_rng_state() -> dict:
    """Snapshot the main process's torch (CPU + CUDA) and numpy RNG states."""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
    }


def _restore_rng_state(state: dict) -> None:
    """Inverse of `_capture_rng_state()`."""
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])


def _signature_mismatch(saved: dict, current: dict) -> str:
    """Human-readable list of the signature entries that differ ('' when identical)."""
    return ", ".join(
        f"{key}: saved {saved.get(key)!r} != current {current.get(key)!r}"
        for key in current
        if saved.get(key) != current[key]
    )


def _epoch_marker(is_best: bool) -> str:
    """One-glyph marker for an epoch's outcome, for `fit(verbose=True)`'s printed line.

    Purely cosmetic terminal readability -- a `raw-full` epoch takes
    ~15 minutes and previously printed nothing until it finished, so at a
    glance it wasn't obvious whether a given epoch had actually improved
    on the run's best result. Carries no information not already in
    `EarlyStopper.is_best`, and plays no role in checkpointing or
    early-stopping decisions themselves.
    """
    return "✅" if is_best else "❌"


def evaluate(
    model: torch.nn.Module,
    loader,
    loss_fn: torch.nn.Module,
    device: str,
    max_batches: int | None = None,
) -> dict:
    """Run `model` in eval mode over `loader`; return loss/F1/degeneracy metrics.

    Accumulates true/false positive/negative counts (and an all-zero/
    all-one flag) incrementally per batch rather than concatenating every
    batch's predictions -- `starcop_raw`'s full val split (26,607 patches)
    OOM'd a first version of this function that did exactly that. Same
    incremental-accumulation reasoning as `stats.py`'s own
    `compute_band_stats`/`compute_class_distribution`: O(1)
    memory per batch, not O(dataset size).

    `max_batches` (default `None` = every batch) stops after that many batches
    for dry runs, where `max_steps=2` on `tier=raw-full` would otherwise still
    walk all 26,607 val patches every "epoch" before the loop even reaches the cheap
    part. Breaking out of the loop, not slicing `loader` beforehand, means
    the un-visited patches are never read off disk at all.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    true_positive = false_positive = false_negative = 0.0
    positive_predictions = 0.0
    total_pixels = 0
    all_zero = True
    all_one = True
    epsilon = 1e-6

    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break
            x, y = batch["input"].to(device), batch["output"].to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            binary = (probs > 0.5).float()
            true_positive += (binary * y).sum().item()
            false_positive += (binary * (1 - y)).sum().item()
            false_negative += ((1 - binary) * y).sum().item()
            positive_predictions += binary.sum().item()
            total_pixels += binary.numel()
            all_zero = all_zero and bool((probs < epsilon).all())
            all_one = all_one and bool((probs > 1 - epsilon).all())

    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "loss": total_loss / n_batches,
        "f1": (2 * true_positive / f1_denominator) if f1_denominator > 0 else 0.0,
        "degenerate": all_zero or all_one,
        "positive_fraction": positive_predictions / total_pixels,
    }


def fit(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset: str,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    max_steps: int | None = None,
    monitor: str = "val_loss",
    resume_from: Path | None = None,
    augment: bool = True,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
    log_to_mlflow: bool = True,
    num_workers: int = 0,
    verbose: bool = False,
    seed: int = _DEFAULT_SEED,
    max_val_batches: int | None = None,
    progress_every: int | None = None,
    state_path: Path | None = None,
    resume_state: bool = False,
) -> dict:
    """Train `model` on `train_df`, validate on `val_df`; return final metrics + run length.

    `state_path` (optional) makes the run **exactly resumable**: after every epoch
    the full training state (model, optimizer, epoch and step counters, both early
    stoppers, best metrics, main-process RNG states, elapsed time and a run
    signature) is written atomically to that file. `resume_state=True` reloads it
    and continues bit-identically to an uninterrupted run, because the two other
    sources of randomness are pure functions of `(seed, epoch[, index])`:
    `EpochShuffleSampler` for the shuffle order and `PatchDataset(augment_seed=...)`
    for augmentation. A resume is refused when `state_path` is missing, when it is
    combined with the weights-only `resume_from`, or when the saved signature
    (`run_signature()`) differs from the current run; a fresh run (`resume_state=
    False`) refuses to overwrite an existing state file. `max_epochs` and `patience`
    may change on resume (extending a run that hit its cap is the intended use).
    The best-checkpoint file at `checkpoint_path` is only ever rewritten by an epoch
    that is a new best, exactly as in an uninterrupted run.

    `progress_every` (optional, default `None` = off) prints a mid-epoch
    heartbeat line every that many (cumulative) steps when `verbose=True`
    -- a `raw-full` epoch takes ~15 minutes and otherwise prints nothing
    until it finishes. Harmless to leave off for `mini`/`r2`, whose
    epochs are already short (25-380 steps) and finish before a
    reasonably-sized interval would ever fire.

    `max_val_batches` ( default `None` = full
    val split) is forwarded to every `evaluate()` call this makes -- for
    dry-running a cell at `tier=raw-full` with `max_steps=2`, so the val
    pass doesn't still walk all 26,607 patches every "epoch" (`raw-smoke`
    already had its own escape hatch for this via a pre-sliced `val_df` in
    `_load_split`; this is the general version for any tier).

    `seed` is applied via `set_seed()` before
    the loaders are built and also seeds `EpochShuffleSampler` (shuffle order)
    and `PatchDataset(augment_seed=...)` (augmentation), so calling `fit()`
    twice with the same seed and the same starting model weights reproduces the
    same shuffle order, augmentation draws, and therefore the same metrics. It does **not**
    seed the model's own weight initialization -- that already happened
    before `model` was passed in; call `set_seed()` yourself before
    building the model if that also needs to be reproducible (`main()`
    does this).

    `wall_clock_seconds` (total) and `seconds_per_epoch` (average; also
    logged per-epoch to MLflow individually) are returned so R3 epoch-cap sizing
    can be based on a number this function actually measured, not the `stats.py`
    I/O-timing estimate.

    `num_workers=0` (the default) is right for tests -- worker-process
    startup is pure overhead against a handful of synthetic patches. Real
    training runs should pass `num_workers>0`: with `PatchDataset`'s
    per-item rasterio reads, a single-process loader re-reads every patch
    from disk every epoch serially -- measured at ~8s/epoch on
    `starcop_mini`'s 392 patches before `PatchDataset` cached decoded
    patches per index; `num_workers=4` with `persistent_workers=True` (on
    the train loader; see below) cuts that further to ~1.5-2s/epoch (real
    GeoTIFFs, not a synthetic fixture) by parallelizing that I/O across
    worker processes that stay alive between epochs instead of respawning
    every time.

    **`num_workers` applies to both the train and val loader, but only the
    train loader's pool is `persistent_workers=True`.** The train loader
    iterates every epoch, so keeping its workers alive avoids re-spawning
    them each time; the val loader runs once per epoch right after it, so a
    *second* full persistent pool held alive for the whole call would just
    double concurrent worker processes for no benefit. That doubling is
    exactly what measurably thrashed this machine's 12-thread CPU at
    `num_workers=8` (16 workers total, both persistent at the time) -- a
    300-step + one-eval smoke run went from ~18s isolated to hanging past a
    200s timeout; `num_workers=4` (8 total, both persistent at the time)
    was verified fast for the same real workload. Keep `num_workers * 2`
    comfortably under the machine's thread count regardless.

    Both loaders also auto-detect a matching `patch_cache`-built on-disk
    cache under `_CACHE_ROOT/<dataset>/{train,val}` (`patch_cache.
    resolve_cache_dir`) and use it transparently when present -- this is
    what actually fixes `starcop_raw`'s disk-I/O bottleneck (benchmarked
    ~95x faster than the live GeoTIFF read it replaces), since the RAM
    cache above barely helps at that scale (see `patch_cache.py`'s own
    docstring). Falls back to live reads with no error when no matching
    cache exists (a tier whose data doesn't match what was cached, e.g.
    `r2`'s subsample or `raw-smoke`'s sliced val, or simply nothing
    precomputed yet) -- build one with `precompute_patch_cache.py`.

    `monitor` (default `"val_loss"`, or `"val_f1"`) is the `evaluate()` key
    early stopping and checkpoint selection track -- `"val_loss"` picks the
    epoch with the lowest val loss (`EarlyStopper(mode="min")`), `"val_f1"`
    the epoch with the highest val F1 (`mode="max"`). Found for real
    reviewing R2/R3 follow-up results: under `starcop_raw`'s much larger
    `pos_weight` (~270-314, vs. ~87 on `mini`) and its huge, overwhelmingly-
    negative val split, `BCEWithLogitsLoss`'s mean-reduced value stops
    tracking segmentation quality -- every R2/R3 run had an epoch a few
    steps from the loss-selected one with 21-39% better F1. `mini` keeps
    the default: its own divergence was small, and its 49-patch val split
    (9 positive) makes raw F1 too noisy per-epoch to select on directly
    (confirmed: an untrained epoch-1 checkpoint scored `val_f1=0.78` on
    9 positive patches purely by chance, alongside a `val_loss=0.44` that
    correctly showed it was still untrained).

    When `monitor="val_f1"`, stopping *also* requires `val_loss`'s own
    patience to be independently exhausted -- checkpoint selection stays
    on raw, unsmoothed `val_f1` (`stopper.is_best` below is untouched),
    only the stop decision gets this second condition. Replaces an
    earlier attempt at fixing the same problem via a smoothed/windowed
    `val_f1` average: that fixed `E2-raw-full` (see above) but broke
    `E1-r2` -- smoothing pulled the *selected* checkpoint away from a
    genuinely good, isolated F1 spike (confirmed by scoring worse on the
    real test split than even plain `val_loss` monitoring), because
    averaging conflates "should we stop" with "which epoch is best" into
    one number. Tracking `val_loss` as a second, independent stop
    condition can only extend training (never change *which* epoch's raw
    F1 ends up selected), so it can't repeat that mistake: `E2-raw-full`
    stopped at epoch 14 because F1 hadn't beaten its epoch-4 spike in
    `patience` epochs, even though `val_loss` was still finding new minima
    through epoch 12 -- proof the model hadn't actually plateaued yet.

    `resume_from` (optional path, default `None`) warm-starts `model`'s
    weights from a previously-saved checkpoint before training begins --
    added after a real GPU driver crash (`cudaErrorLaunchTimeout`) lost an
    in-progress `raw-full` (R3) run partway through (72 epochs in, best
    `val_f1` already far ahead of any prior attempt). Not a full resume:
    only weights are restored, not the optimizer/epoch-counter/early-
    stopping state -- Adam's own momentum re-adapts within a handful of
    steps, and a true resume would mean changing the checkpoint file
    format (currently a bare `state_dict`, loaded the same way by every
    other caller in this project, e.g. `evaluate.py`'s `load_checkpoint`).
    The new run's own `EarlyStopper`(s) start fresh and epoch numbering
    restarts at 1, but before epoch 1 the resumed weights are scored once
    on the val split ("epoch 0", also logged to MLflow at `step=0`) and that
    score seeds the stoppers' starting "best", with the weights written to
    `checkpoint_path`. An epoch therefore only replaces `checkpoint_path`
    if it beats the resumed weights on `monitor`; if none does, the result
    is the baseline's metrics with `best_epoch=0`. (The first version of
    this skipped the baseline, so epoch 1 always overwrote the checkpoint
    -- even when one epoch of fresh-Adam training left the model worse.)
    """
    monitor_key, monitor_mode = _MONITOR_METRIC_KEYS[monitor]
    if resume_state:
        if state_path is None:
            raise ValueError("resume_state=True requires a state_path")
        if resume_from is not None:
            raise ValueError(
                "resume_state=True cannot be combined with the weights-only resume_from"
            )
        if not state_path.exists():
            raise FileNotFoundError(f"no run state to resume at {state_path}")
    elif state_path is not None and state_path.exists():
        raise FileExistsError(
            f"{state_path} already exists: pass resume_state=True to continue that run, "
            "or move the file aside to start a new one"
        )
    set_seed(seed)
    model.to(device)
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from, map_location=device))
    pos_weight = compute_pos_weight(train_df)
    loss_fn = build_loss(pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    signature = run_signature(
        model,
        train_df,
        dataset=dataset,
        lr=lr,
        batch_size=batch_size,
        augment=augment,
        monitor=monitor,
        seed=seed,
        pos_weight=pos_weight,
    )
    saved_state = None
    if resume_state:
        saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
        mismatch = _signature_mismatch(saved_state["signature"], signature)
        if mismatch:
            raise ValueError(f"run signature mismatch, refusing to resume: {mismatch}")

    train_cache_dir = patch_cache.resolve_cache_dir(_CACHE_ROOT, dataset, "train", train_df)
    val_cache_dir = patch_cache.resolve_cache_dir(_CACHE_ROOT, dataset, "val", val_df)
    train_dataset = PatchDataset(
        train_df, dataset=dataset, augment=augment, cache_dir=train_cache_dir, augment_seed=seed
    )
    sampler = EpochShuffleSampler(len(train_dataset), seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        **_loader_kwargs(device, num_workers, seed, persistent_workers=True),
    )
    val_loader = DataLoader(
        PatchDataset(val_df, dataset=dataset, augment=False, cache_dir=val_cache_dir),
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(device, num_workers, seed, persistent_workers=False),
    )

    stopper = EarlyStopper(patience=patience, mode=monitor_mode)

    loss_plateau_stopper = (
        EarlyStopper(patience=patience, mode="min") if monitor == "val_f1" else None
    )
    step_count = 0
    epoch = 0
    val_metrics: dict = {}
    best_metrics: dict = {}
    best_epoch = 0
    stop_early = False
    elapsed_before_resume = 0.0

    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        optimizer.load_state_dict(saved_state["optimizer"])
        stopper.load_state_dict(saved_state["stopper"])
        if loss_plateau_stopper is not None:
            loss_plateau_stopper.load_state_dict(saved_state["loss_plateau_stopper"])
        step_count = saved_state["step_count"]
        epoch = saved_state["epoch"]
        val_metrics = saved_state["val_metrics"]
        best_metrics = saved_state["best_metrics"]
        best_epoch = saved_state["best_epoch"]
        stop_early = saved_state["stopped"]
        elapsed_before_resume = saved_state["elapsed_seconds"]
        _restore_rng_state(saved_state["rng"])
        if verbose:
            print(f"  ⏯️  resumed exact state after epoch {epoch} (step {step_count})")
    fit_start = time.perf_counter() - elapsed_before_resume

    if resume_from is not None:
        baseline_metrics = evaluate(model, val_loader, loss_fn, device, max_batches=max_val_batches)
        stopper.step(baseline_metrics[monitor_key])
        if loss_plateau_stopper is not None:
            loss_plateau_stopper.step(baseline_metrics["loss"])
        best_metrics = baseline_metrics
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        if log_to_mlflow:
            mlflow.log_metrics(
                {"val_loss": baseline_metrics["loss"], "val_f1": baseline_metrics["f1"]}, step=0
            )
        if verbose:
            print(
                f"  🏁 resume baseline (epoch 0): val_loss={baseline_metrics['loss']:.4f} "
                f"val_f1={baseline_metrics['f1']:.4f} -- epochs below must beat this to be saved"
            )

    remaining_epochs = range(epoch + 1, max_epochs + 1) if not stop_early else range(0)
    for epoch in remaining_epochs:
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        model.train()
        for batch in train_loader:
            x, y = batch["input"].to(device), batch["output"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            step_count += 1
            if verbose and progress_every and step_count % progress_every == 0:
                print(
                    f"    ... epoch {epoch}/{max_epochs} step {step_count} "
                    f"({time.perf_counter() - epoch_start:.0f}s elapsed this epoch)"
                )
            if max_steps is not None and step_count >= max_steps:
                break

        val_metrics = evaluate(model, val_loader, loss_fn, device, max_batches=max_val_batches)
        epoch_seconds = time.perf_counter() - epoch_start
        if log_to_mlflow:
            mlflow.log_metrics(
                {
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "seconds_per_epoch": epoch_seconds,
                },
                step=epoch,
            )

        should_stop = stopper.step(val_metrics[monitor_key])
        if loss_plateau_stopper is not None:
            loss_plateaued = loss_plateau_stopper.step(val_metrics["loss"])
            should_stop = should_stop and loss_plateaued
        if verbose:
            print(
                f"  {_epoch_marker(stopper.is_best)} epoch {epoch}/{max_epochs} "
                f"step {step_count}: val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_metrics['f1']:.4f} ({epoch_seconds:.1f}s)"
            )

        if stopper.is_best:
            best_metrics = val_metrics
            best_epoch = epoch
            if checkpoint_path is not None:
                _atomic_save(model.state_dict(), checkpoint_path)

        stopping = (max_steps is not None and step_count >= max_steps) or should_stop
        if state_path is not None:
            _atomic_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "step_count": step_count,
                    "stopper": stopper.state_dict(),
                    "loss_plateau_stopper": (
                        loss_plateau_stopper.state_dict()
                        if loss_plateau_stopper is not None
                        else None
                    ),
                    "best_metrics": best_metrics,
                    "best_epoch": best_epoch,
                    "val_metrics": val_metrics,
                    "stopped": stopping,
                    "elapsed_seconds": time.perf_counter() - fit_start,
                    "rng": _capture_rng_state(),
                    "signature": signature,
                },
                state_path,
            )
        if stopping:
            stop_early = True
            break

    wall_clock_seconds = time.perf_counter() - fit_start
    return {
        **best_metrics,
        "best_epoch": best_epoch,
        "last_epoch_metrics": val_metrics,
        "epochs_run": epoch,
        "steps_run": step_count,
        "pos_weight": pos_weight,
        "stopped_early": stop_early,
        "wall_clock_seconds": wall_clock_seconds,
        "seconds_per_epoch": wall_clock_seconds / epoch,
    }


def _parse_kv_args(argv: list[str]) -> dict[str, str]:
    """Parse command-line `key=value` arguments."""
    parsed = {}
    for arg in argv:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed[key] = value
    return parsed


def _load_split(dataset: str, tier: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (train_df, val_df) for `tier` -- 'mini'/'raw-full' use the dataset's own splits,
    'r2' uses the persisted subsample manifests (built once by `build_r2_manifest.py`, never
    regenerated per run), 'raw-smoke' uses the full raw train split (so `max_steps` samples real,
    varied patches) but a small seeded slice of val -- the smoke run only needs to prove the
    loop works, not a real val number, and evaluating the full 26,607-patch raw val split every
    epoch is pure waste here."""
    if tier == "r2":
        return (
            pd.read_csv(
                _PROJECT_ROOT / "data" / "manifests" / "r2_manifest_train.csv", low_memory=False
            ),
            pd.read_csv(
                _PROJECT_ROOT / "data" / "manifests" / "r2_manifest_val.csv", low_memory=False
            ),
        )
    patches_root = _REPO_ROOT / "data" / "processed" / dataset / "patches"
    train_df = pd.read_csv(patches_root / "train_tiled_128_128.csv", low_memory=False)
    val_df = pd.read_csv(patches_root / "val_tiled_128_128.csv", low_memory=False)
    if tier == "raw-smoke":
        val_df = val_df.sample(n=min(200, len(val_df)), random_state=42).reset_index(drop=True)
    return train_df, val_df


def main() -> None:
    """Run a configured training experiment."""
    args = _parse_kv_args(sys.argv[1:])
    architecture = args.get("architecture", "E1")
    dataset = args.get("dataset", "starcop_mini")
    tier = args.get("tier", "mini")
    max_steps = int(args["max_steps"]) if "max_steps" in args else None
    seed = int(args.get("seed", _DEFAULT_SEED))
    max_val_batches = int(args["max_val_batches"]) if "max_val_batches" in args else None
    progress_every = int(args["progress_every"]) if "progress_every" in args else None
    monitor = args.get("monitor", "val_loss")
    resume_from = Path(args["resume_from"]) if "resume_from" in args else None
    resume_state = args.get("resume_state", "false").lower() == "true"
    state_path = _STATE_DIR / f"{architecture}-{tier}.state.pt"
    run_id_path = state_path.with_suffix(".mlflow_run_id")
    if not resume_state and state_path.exists():
        raise FileExistsError(
            f"{state_path} exists: pass resume_state=true to continue it, or move it aside"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 1e-3 if architecture == "E1" else 1e-4
    batch_size = 16
    default_max_epochs = 50 if max_steps is None else 1
    max_epochs = int(args.get("max_epochs", default_max_epochs))
    patience = int(args.get("patience", 10))
    num_workers = 4
    augment = augment_for(architecture)
    loss_name = "BCEWithLogitsLoss+pos_weight"
    optimizer_name = "Adam"

    train_df, val_df = _load_split(dataset, tier)

    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    set_seed(seed)
    model = build_model(architecture)
    resumed_run_id = (
        run_id_path.read_text().strip() if resume_state and run_id_path.exists() else None
    )
    with mlflow.start_run(run_id=resumed_run_id, run_name=f"{architecture}-{tier}") as active_run:
        if resumed_run_id is None:
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id_path.write_text(active_run.info.run_id)
        if resumed_run_id is None:
            mlflow.log_params(
                {
                    "architecture": architecture,
                    "dataset": dataset,
                    "tier": tier,
                    "device": device,
                    "seed": seed,
                    "train_patches": len(train_df),
                    "val_patches": len(val_df),
                    "lr": lr,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "monitor": monitor,
                    "num_workers": num_workers,
                    "augment": augment,
                    "loss": loss_name,
                    "optimizer": optimizer_name,
                    "param_count": count_parameters(model),
                    **library_versions(),
                    **({"resume_from": str(resume_from)} if resume_from is not None else {}),
                }
            )
        else:
            mlflow.set_tag("resumed_from_state", "true")
        result = fit(
            model,
            train_df,
            val_df,
            dataset=dataset,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            max_steps=max_steps,
            monitor=monitor,
            resume_from=resume_from,
            augment=augment,
            device=device,
            checkpoint_path=_CHECKPOINT_DIR / f"{architecture}-{tier}.pt",
            state_path=state_path,
            resume_state=resume_state,
            num_workers=num_workers,
            verbose=True,
            seed=seed,
            max_val_batches=max_val_batches,
            progress_every=progress_every,
        )
        mlflow.log_metrics({k: v for k, v in result.items() if isinstance(v, int | float)})
        print(f"architecture={architecture} tier={tier} dataset={dataset}")
        print(result)


if __name__ == "__main__":
    main()
