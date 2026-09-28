"""Scene-level manifests for the paper-protocol evaluation.

The papers score whole 512x512 scenes; this project trains and (until now) evaluated on
128x128 patches cut from them. The patch manifests carry a *patch-level* `has_plume`, so the
scene-level truth -- `has_plume`, `qplume` and the strong/weak/plume-free bucket -- has to be
re-joined from the papers' own scene CSVs (`test.csv`; `train.csv` for the validation scenes).
One manifest row per parent scene, in the same column layout `PatchDataset` reads, so a
scene manifest can be fed to it directly (window = the whole scene).

Bucket definitions follow the STARCOP paper's evaluation: strong plumes are emissions of at
least 1000 kg/h, weak plumes are the remaining positive scenes, and plume-free scenes are
those without an emission.
"""

import pandas as pd

SCENE_SIZE = 512
PATCHES_PER_SCENE = 49
STRONG_QPLUME_THRESHOLD = 1000.0


def assign_bucket(has_plume: bool, qplume: float) -> str:
    """Return `"strong"`, `"weak"` or `"plume_free"` for one scene's label.

    A scene has a plume exactly when its emission rate is positive; a label that says
    otherwise (a plume with no positive emission, an emission without a plume, a missing
    rate) is rejected rather than silently bucketed.
    """
    if has_plume:
        if not qplume > 0:
            raise ValueError(f"inconsistent scene label: has_plume=True with qplume={qplume}")
        return "strong" if qplume >= STRONG_QPLUME_THRESHOLD else "weak"
    if qplume != 0:
        raise ValueError(f"inconsistent scene label: has_plume=False with qplume={qplume}")
    return "plume_free"


def build_scene_manifest(
    labels_df: pd.DataFrame,
    patch_manifest_df: pd.DataFrame,
    *,
    patches_per_scene: int = PATCHES_PER_SCENE,
    scene_size: int = SCENE_SIZE,
) -> pd.DataFrame:
    """One row per parent scene of `patch_manifest_df`, labelled from `labels_df`.

    `labels_df` needs `id`, `qplume`, `has_plume` (scene level, e.g. `test.csv`); rows for
    scenes with no patches are ignored, so `train.csv` serves for the validation scenes.
    `patch_manifest_df` needs `id_original` (the parent scene) and `folder` (its processed
    directory). Raises `ValueError` if a parent scene has no label, does not have exactly
    `patches_per_scene` patches, has patches in more than one folder, or carries a
    contradictory label. Result columns: `scene_id, folder, window_col_off, window_row_off,
    window_width, window_height, has_plume, qplume, bucket, n_patches`, sorted by `scene_id`.
    """
    duplicated = labels_df.loc[labels_df["id"].duplicated(), "id"].tolist()
    if duplicated:
        raise ValueError(f"duplicate scene labels for: {duplicated[:5]}")
    labels = labels_df.set_index("id")

    grouped = patch_manifest_df.groupby("id_original")
    patch_counts = grouped.size()
    wrong_count = patch_counts[patch_counts != patches_per_scene]
    if len(wrong_count):
        first = wrong_count.index[0]
        raise ValueError(
            f"scene {first} has {wrong_count.iloc[0]} patches, expected {patches_per_scene} "
            f"({len(wrong_count)} scene(s) affected)"
        )
    folder_counts = grouped["folder"].nunique()
    split_scenes = folder_counts[folder_counts > 1].index.tolist()
    if split_scenes:
        raise ValueError(f"patches in more than one folder for scene(s): {split_scenes[:5]}")
    unlabelled = [scene_id for scene_id in patch_counts.index if scene_id not in labels.index]
    if unlabelled:
        raise ValueError(f"no scene label for {len(unlabelled)} scene(s), e.g. {unlabelled[:5]}")

    folders = grouped["folder"].first()
    rows = []
    for scene_id in patch_counts.index:
        has_plume = bool(labels.at[scene_id, "has_plume"])
        qplume = float(labels.at[scene_id, "qplume"])
        try:
            bucket = assign_bucket(has_plume, qplume)
        except ValueError as error:
            raise ValueError(f"scene {scene_id}: {error}") from error
        rows.append(
            {
                "scene_id": scene_id,
                "folder": folders[scene_id],
                "window_col_off": 0,
                "window_row_off": 0,
                "window_width": scene_size,
                "window_height": scene_size,
                "has_plume": has_plume,
                "qplume": qplume,
                "bucket": bucket,
                "n_patches": int(patch_counts[scene_id]),
            }
        )
    return pd.DataFrame(rows)


def load_scene_manifest(labels_csv, patch_manifest_csv, **kwargs) -> pd.DataFrame:
    """`build_scene_manifest` over two CSV files (`kwargs` are passed through)."""

    labels_df = pd.read_csv(labels_csv, usecols=["id", "qplume", "has_plume"])
    patch_manifest_df = pd.read_csv(patch_manifest_csv, usecols=["id_original", "folder"])

    return build_scene_manifest(labels_df, patch_manifest_df, **kwargs)
