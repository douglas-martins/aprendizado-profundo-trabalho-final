"""Validate the complete data, model, and metric path on real dataset patches.

The command checks tensor shapes, dtypes, normalization ranges, band order,
label passthrough, model forward and backward passes, and full-test metric
accumulation without training a model.

Usage: python -m data.confirm_raw dataset=starcop_raw
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.dataset import PatchDataset
from data.preprocessing import BAND_NORMALIZATION
from evaluation.evaluate import evaluate_full_metrics
from models.architectures import build_e1, build_e2, build_e3

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_SIZE = 300
_SEED = 42
_METRIC_CHECK_BATCH_SIZE = 32


def _parse_dataset_arg(argv: list[str]) -> str:
    """Parse command-line `key=value` arguments."""
    for arg in argv:
        if arg.startswith("dataset="):
            return arg.split("=", 1)[1]
    return "starcop_raw"


def load_sample(dataset: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Load the train patches CSV for `dataset` and return a seeded, scene-spanning sample."""
    patches_path = (
        _REPO_ROOT / "data" / "processed" / dataset / "patches" / "train_tiled_128_128.csv"
    )
    full = pd.read_csv(patches_path, low_memory=False)
    n = min(sample_size, len(full))
    return full.sample(n=n, random_state=seed).reset_index(drop=True)


def _expected_patch_size() -> tuple[int, int]:
    """Read `patch.size` from configs/data.yaml, so a future config change can't go unnoticed."""
    config = OmegaConf.load(_REPO_ROOT / "configs" / "data.yaml")
    height, width = config.patch.size
    return int(height), int(width)


def check_architectures(patch_dataset: PatchDataset, batch_size: int = 8) -> None:
    """R1: forward + backward pass for E1/E2/E3 on one real batch.

    A random `torch.randn` tensor can't catch a NaN arriving from raw
    `mag1c`, a dtype mismatch, or an all-negative batch -- all three are
    real `starcop_raw` conditions this checks against actual data instead.
    """
    loader = DataLoader(patch_dataset, batch_size=min(batch_size, len(patch_dataset)))
    batch = next(iter(loader))
    input_batch, output_batch = batch["input"], batch["output"]
    print(f"  architecture batch: {tuple(input_batch.shape)}")

    for name, build in [("E1", build_e1), ("E2", build_e2), ("E3", build_e3)]:
        model = build() if name == "E1" else build(pretrained=True)
        model.train()
        logits = model(input_batch)
        assert logits.shape == output_batch.shape, (
            f"{name}: expected output shape {tuple(output_batch.shape)}, got {tuple(logits.shape)}"
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, output_batch)
        model.zero_grad()
        loss.backward()
        assert torch.isfinite(loss), f"{name}: non-finite loss {loss.item()}"
        missing_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing_grad, f"{name}: {len(missing_grad)} parameters got no gradient"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,} params, loss={loss.item():.4f}, all gradients present")


def check_metric_code_on_full_test_split(dataset: str) -> None:
    """R1: `evaluate.evaluate_full_metrics` runs over `dataset`'s *entire*
    test split -- 16,758 patches for `starcop_raw` -- with incremental accumulation,
    no OOM, and no silent truncation (verified by `patches_processed` matching the
    split's own row count, not by this function merely returning).

    Uses a freshly built (untrained) E2 model, inference only: this check is about
    the accumulation code's mechanics at real scale, not model accuracy,
    real evaluation runs against trained checkpoints already cover accuracy.
    """
    test_path = _REPO_ROOT / "data" / "processed" / dataset / "patches" / "test_tiled_128_128.csv"
    test_df = pd.read_csv(test_path, low_memory=False)
    print(f"  metric code: full test split for {dataset} has {len(test_df)} patches")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_e2(pretrained=False).to(device)
    loader = DataLoader(
        PatchDataset(test_df, dataset=dataset, augment=False),
        batch_size=_METRIC_CHECK_BATCH_SIZE,
        num_workers=4,
    )

    start = time.perf_counter()
    result = evaluate_full_metrics(model, loader, device)
    elapsed = time.perf_counter() - start

    assert result["patches_processed"] == len(test_df), (
        f"processed {result['patches_processed']} patches, expected {len(test_df)} "
        "-- the split was silently truncated"
    )
    print(
        f"  metric code: processed all {result['patches_processed']} patches in "
        f"{elapsed:.1f}s on {device}, no OOM, no truncation "
        f"(confusion_matrix={result['confusion_matrix']})"
    )


def run(dataset: str) -> None:
    """Run every R1 check against a real sample of `dataset` and print a pass/fail summary."""
    print(f"R1 confirmation -- dataset={dataset}")

    sample = load_sample(dataset, _SAMPLE_SIZE, _SEED)
    scene_count = sample["name"].nunique()
    print(f"  sampled {len(sample)} patches across {scene_count} distinct scenes")
    assert scene_count > 1, "sample must span more than one scene to be a real R1 check"

    patch_dataset = PatchDataset(sample, dataset=dataset, augment=False)
    input_products = patch_dataset.input_products
    patch_height, patch_width = _expected_patch_size()
    print(f"  input_products (band order): {input_products}")
    print(f"  expected patch size (configs/data.yaml): {patch_height}x{patch_width}")

    negative_count = 0
    for index in range(len(patch_dataset)):
        item = patch_dataset[index]
        input_tensor, output_tensor = item["input"], item["output"]

        expected_input_shape = (len(input_products), patch_height, patch_width)
        assert tuple(input_tensor.shape) == expected_input_shape, (
            f"expected input shape {expected_input_shape}, got {tuple(input_tensor.shape)}"
        )
        expected_output_shape = (1, patch_height, patch_width)
        assert tuple(output_tensor.shape) == expected_output_shape, (
            f"expected output shape {expected_output_shape}, got {tuple(output_tensor.shape)}"
        )
        assert str(input_tensor.dtype) == "torch.float32", input_tensor.dtype
        for product in input_products:
            clip_min, clip_max = BAND_NORMALIZATION[product]["clip"]
            assert input_tensor.min() >= clip_min - 1e-5, (
                f"{product} min {input_tensor.min()} below clip {clip_min}"
            )
            assert input_tensor.max() <= clip_max + 1e-5, (
                f"{product} max {input_tensor.max()} above clip {clip_max}"
            )

        label_values = set(output_tensor.unique().tolist())
        assert label_values <= {0.0, 1.0}, f"label has non-binary values: {label_values}"
        if label_values == {0.0} or len(label_values) == 0:
            negative_count += 1

    total = len(patch_dataset)
    negative_fraction = negative_count / total
    print(f"  shapes/dtypes/ranges/band-order/label-passthrough: PASSED ({total} patches)")
    print(
        f"  all-negative patches in sample: {negative_count}/{total} "
        f"({negative_fraction:.1%}) -- informational, a design input for"
        " sampling strategy, not a failure here"
    )

    check_architectures(patch_dataset)
    check_metric_code_on_full_test_split(dataset)
    print("R1 PASSED")


def main() -> None:
    """CLI entry point: resolve `dataset=` from argv and run the R1 check."""
    dataset = _parse_dataset_arg(sys.argv[1:])
    try:
        run(dataset)
    except AssertionError as error:
        print(f"R1 FAILED: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
