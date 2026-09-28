"""Reference metric records used by the comparison calculations."""

from typing import NamedTuple

Value = tuple[float | None, float | None]


class Reference(NamedTuple):
    """One published result row."""

    label: str
    source: str
    citation: str
    metrics: dict[str, Value]
    caveats: tuple[str, ...] = ()


_STARCOP_TABLE_2 = "Table 2, page 10 (AVIRIS-NG)"
_HEREC_CAVEATS = (
    "Herec's weak/strong split is by label size, not by STARCOP's emission "
    "rate of 1000 kg/h, so 'F1 strong' is not strictly the same population as STARCOP's.",
    "The paper calls this column F1 'for all plumes' (p. 6, 8) without stating whether pixels are "
    "pooled or averaged per scene; treating it as pooled pixels over plume scenes only is an "
    "assumption.",
    "Herec's models take the Mag1c-SAS product as input; ours take standard mag1c, so the input "
    "differs.",
    "Verified against the 2026 publication (Table I caption, p. 8; Methodology, p. 6): mean and "
    "std are over 5 repeated training runs, and the decision threshold is 0.5, the same fixed "
    "point ours uses.",
)

REFERENCES: dict[str, Reference] = {
    "starcop_baseline": Reference(
        label="STARCOP baseline: mag1c + morphology",
        source="starcop",
        citation=f"STARCOP paper, {_STARCOP_TABLE_2}, row 'Baseline, mag1c + morpho.'",
        metrics={
            "strong_f1": (67.45, None),
            "weak_f1": (39.95, None),
            "tile_fpr": (75.43, None),
            "auprc": (None, None),
        },
    ),
    "starcop_mag1c_only": Reference(
        label="HyperSTARCOP, only mag1c",
        source="starcop",
        citation=f"STARCOP paper, {_STARCOP_TABLE_2}, row 'HyperSTARCOP, only mag1c'",
        metrics={
            "strong_f1": (74.15, 6.10),
            "weak_f1": (47.57, 4.17),
            "tile_fpr": (52.11, 10.98),
            "auprc": (49.41, 5.49),
        },
    ),
    "starcop_mag1c_rgb": Reference(
        label="HyperSTARCOP, mag1c + rgb",
        source="starcop",
        citation=f"STARCOP paper, {_STARCOP_TABLE_2}, row 'HyperSTARCOP, mag1c + rgb'",
        metrics={
            "strong_f1": (81.96, 3.71),
            "weak_f1": (43.42, 5.72),
            "tile_fpr": (43.66, 7.36),
            "auprc": (51.99, 2.76),
        },
        caveats=(
            "The paper's prose (Results, p. 10) gives this FPR as 43.79 while its Table 2 gives "
            "43.66; the table value is used.",
        ),
    ),
    "herec_mag1c_column_wise": Reference(
        label="Herec: Mag1c original, column-wise",
        source="herec",
        citation=(
            "Herec et al. 2026 (arXiv 2606.03675), Table I, p. 8, row "
            "'Mag1c original, column-wise' (identical values in the 2025 preprint, "
            "arXiv 2507.01472)"
        ),
        metrics={
            "recall": (58.42, None),
            "precision": (30.57, None),
            "f1": (40.14, None),
            "f1_strong": (67.50, None),
        },
        caveats=_HEREC_CAVEATS[:2],
    ),
    "herec_unet_mag1c_sas": Reference(
        label="Herec: U-Net + Mag1c-SAS",
        source="herec",
        citation=(
            "Herec et al. 2026 (arXiv 2606.03675), Table I, p. 8, row 'U-Net + Mag1c-SAS' "
            "(identical values in the 2025 preprint, arXiv 2507.01472)"
        ),
        metrics={
            "recall": (56.41, 7.0),
            "precision": (34.62, 7.4),
            "f1": (42.54, 6.7),
            "f1_strong": (61.38, 7.7),
        },
        caveats=_HEREC_CAVEATS,
    ),
    "herec_linknet_mag1c_sas": Reference(
        label="Herec: LinkNet + Mag1c-SAS",
        source="herec",
        citation=(
            "Herec et al. 2026 (arXiv 2606.03675), Table I, p. 8, row 'LinkNet + Mag1c-SAS' "
            "(identical values in the 2025 preprint, arXiv 2507.01472)"
        ),
        metrics={
            "recall": (51.11, 7.2),
            "precision": (40.43, 6.4),
            "f1": (44.44, 3.9),
            "f1_strong": (60.37, 5.1),
        },
        caveats=_HEREC_CAVEATS,
    ),
}
