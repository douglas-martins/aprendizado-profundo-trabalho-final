"""Load normalized image patches and their binary segmentation labels."""

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kornia.augmentation import (
    AugmentationSequential,
    RandomHorizontalFlip,
    RandomRotation90,
    RandomVerticalFlip,
)
from omegaconf import OmegaConf
from rasterio.windows import Window
from torch.utils.data import Dataset

from data import patch_cache
from data.preprocessing import BAND_NORMALIZATION, normalize_band
from visualization.eda import read_patch_bands

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _dataset_config_path(dataset: str) -> Path:
    """Return the configuration path for `dataset`."""
    return _REPO_ROOT / "configs" / "dataset" / f"{dataset}.yaml"


def _load_dataset_config(dataset: str) -> tuple[list[str], list[str]]:
    """Return input and output products from the dataset YAML configuration."""
    config = OmegaConf.load(_dataset_config_path(dataset))
    return list(config.dataset_cfg.input_products), list(config.dataset_cfg.output_products)


def _build_augmenter() -> AugmentationSequential:
    """Build the train-only augmenter: flips + 90-degree rotations, image and mask synced.

    No color-jitter-style transform is included -- those would distort the
    physically meaningful mag1c/reflectance values.
    """

    rotation = RandomRotation90(times=(0, 3), p=0.5)

    data_keys = ["image", "mask"]

    return AugmentationSequential(
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        rotation,
        data_keys=data_keys,
    )


@contextmanager
def _seeded_cpu_rng(seed: int) -> Iterator[None]:
    """Seed the CPU torch RNG with `seed` inside the block, restoring its previous state after."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


_DEFAULT_MAX_CACHE_BYTES = 2 * 1024**3


class PatchDataset(Dataset):
    """One 128x128 (or fixture-sized) patch per item: normalized input stack + raw label."""

    def __init__(
        self,
        patches_df: pd.DataFrame,
        dataset: str,
        augment: bool,
        max_cache_bytes: int = _DEFAULT_MAX_CACHE_BYTES,
        cache_dir: Path | None = None,
        augment_seed: int | None = None,
    ):
        """Store `patches_df`, resolve `dataset`'s band contract, and build the augmenter.

        `augment_seed` (optional) makes augmentation a pure function of
        `(augment_seed, epoch, index)`: each read seeds a forked CPU RNG from that
        triple, so the draw does not depend on how many samples any worker process
        has already produced (which is what made a crash + resume diverge from an
        uninterrupted run) and never touches the global torch RNG. `set_epoch()`
        advances the epoch. `None` keeps the legacy behaviour: draws come from the
        global torch RNG.

        `max_cache_bytes` bounds the decoded-patch RAM cache (see
        `__getitem__`) with LRU eviction, so caching stays safe at any
        dataset size instead of growing without limit.

        `cache_dir` (optional) points at a `patch_cache`-built on-disk
        cache (see that module's own docstring for why it exists) -- when
        given, it **must** match `patches_df` exactly (`patch_cache.
        is_valid`); this is an explicit opt-in contract, not a hint, so a
        stale or mismatched cache fails loudly here rather than silently
        serving wrong patches or silently falling back to live reads. When
        active, every access reads from the on-disk cache instead of doing
        a live GeoTIFF windowed read, and the RAM cache above is unused --
        the on-disk cache is already far faster than the RAM cache could
        make live reads.
        """
        self.patches_df = patches_df.reset_index(drop=True)
        self.input_products, self.output_products = _load_dataset_config(dataset)
        self.augmenter = _build_augmenter() if augment else None
        self.augment_seed = augment_seed

        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()
        self._cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._max_cache_bytes = max_cache_bytes
        self._cache_bytes_used = 0
        self._disk_cache: tuple[np.ndarray, np.ndarray] | None = None
        if cache_dir is not None:
            if not patch_cache.is_valid(cache_dir, self.patches_df):
                detail = "rebuild it with precompute_patch_cache.py before using it here."

                raise ValueError(
                    f"cache_dir {cache_dir!r} does not match the given patches_df -- {detail}"
                )
            self._disk_cache = patch_cache.load(cache_dir)

    @staticmethod
    def _entry_bytes(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> int:
        """Byte size of one cached (input, output) pair."""
        return (
            input_tensor.numel() * input_tensor.element_size()
            + output_tensor.numel() * output_tensor.element_size()
        )

    def __len__(self) -> int:
        """Number of patches."""
        return len(self.patches_df)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch that seeded augmentation draws use (see `augment_seed`).

        Visible to DataLoader worker processes, including persistent ones.
        """
        self._epoch[0] = epoch

    def _augmentation_rng(self, index: int) -> AbstractContextManager:
        """Context seeding augmentation from `(augment_seed, epoch, index)`; RNG restored after.

        No-op when `augment_seed` is `None` (legacy: global RNG).
        """
        if self.augment_seed is None:
            return nullcontext()
        entropy = [self.augment_seed, int(self._epoch[0]), index]
        seed = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
        return _seeded_cpu_rng(seed)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return {"input": (C,H,W) normalized float32, "output": (1,H,W) raw 0/1 float32}.

        The decoded, normalized (pre-augmentation) pair is cached per index
        after its first read -- without this, the same patch is re-read and
        re-decoded from disk on every access, every epoch, which measured as
        the dominant per-epoch cost (`fit()`'s own docstring: ~8s/epoch on
        `starcop_mini`'s 392 patches with no worker parallelism). Augmentation
        still runs fresh on every call from a clone of the cached pair, since
        it must vary per epoch and must never let one caller's in-place
        mutation of a returned tensor corrupt a later read of the same index.

        The cache is LRU-bounded by `max_cache_bytes` (see `__init__`): an
        entry that would push total cached bytes over budget evicts the
        least-recently-used entries first, and an entry larger than the
        whole budget is served but never cached.

        When `cache_dir` was given at construction, this reads from that
        on-disk cache instead -- see `__init__`'s own docstring.
        """
        if self._disk_cache is not None:
            cached_inputs, cached_outputs = self._disk_cache
            input_tensor = torch.from_numpy(cached_inputs[index].astype(np.float32))
            output_tensor = torch.from_numpy(cached_outputs[index].astype(np.float32))
        elif index in self._cache:
            self._cache.move_to_end(index)
            cached_input, cached_output = self._cache[index]
            input_tensor, output_tensor = cached_input.clone(), cached_output.clone()
        else:
            row = self.patches_df.iloc[index]
            window = Window(
                col_off=row["window_col_off"],
                row_off=row["window_row_off"],
                width=row["window_width"],
                height=row["window_height"],
            )
            output_band = self.output_products[0]
            bands = read_patch_bands(
                Path(row["folder"]), window, [*self.input_products, output_band]
            )

            normalized = [
                normalize_band(bands[product], **BAND_NORMALIZATION[product])
                for product in self.input_products
            ]

            input_tensor = torch.from_numpy(np.stack(normalized, axis=0))
            output_tensor = torch.from_numpy(bands[output_band]).unsqueeze(0).float()

            entry_bytes = self._entry_bytes(input_tensor, output_tensor)
            while self._cache and self._cache_bytes_used + entry_bytes > self._max_cache_bytes:
                oldest_index = next(iter(self._cache))
                evicted = self._cache.pop(oldest_index)
                self._cache_bytes_used -= self._entry_bytes(*evicted)
            if entry_bytes <= self._max_cache_bytes:
                self._cache[index] = (input_tensor.clone(), output_tensor.clone())
                self._cache_bytes_used += entry_bytes

        if self.augmenter is not None:
            input_batch = input_tensor.unsqueeze(0)

            output_batch = output_tensor.unsqueeze(0)
            with self._augmentation_rng(index):
                augmented_input, augmented_output = self.augmenter(input_batch, output_batch)
            input_tensor, output_tensor = augmented_input[0], augmented_output[0]

        return {"input": input_tensor, "output": output_tensor}
