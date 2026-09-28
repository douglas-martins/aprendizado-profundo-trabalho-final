import re

from evaluation.paper_reference import REFERENCES, Reference

STARCOP_KEYS = {"strong_f1", "weak_f1", "tile_fpr", "auprc"}
HEREC_KEYS = {"recall", "precision", "f1", "f1_strong"}


class TestStarcopTable2:
    def test_the_hyperstarcop_mag1c_plus_rgb_row(self):
        row = REFERENCES["starcop_mag1c_rgb"]

        assert row.metrics == {
            "strong_f1": (81.96, 3.71),
            "weak_f1": (43.42, 5.72),
            "tile_fpr": (43.66, 7.36),
            "auprc": (51.99, 2.76),
        }

    def test_the_hyperstarcop_mag1c_only_row(self):
        assert REFERENCES["starcop_mag1c_only"].metrics == {
            "strong_f1": (74.15, 6.10),
            "weak_f1": (47.57, 4.17),
            "tile_fpr": (52.11, 10.98),
            "auprc": (49.41, 5.49),
        }

    def test_the_matched_filter_baseline_has_no_spread_and_no_auprc(self):

        assert REFERENCES["starcop_baseline"].metrics == {
            "strong_f1": (67.45, None),
            "weak_f1": (39.95, None),
            "tile_fpr": (75.43, None),
            "auprc": (None, None),
        }

    def test_the_text_and_the_table_disagree_on_the_fpr_and_that_is_recorded(self):

        caveats = " ".join(REFERENCES["starcop_mag1c_rgb"].caveats)

        assert "43.79" in caveats
        assert "43.66" in caveats


class TestHerecTableI:
    def test_the_linknet_mag1c_sas_row(self):
        assert REFERENCES["herec_linknet_mag1c_sas"].metrics == {
            "recall": (51.11, 7.2),
            "precision": (40.43, 6.4),
            "f1": (44.44, 3.9),
            "f1_strong": (60.37, 5.1),
        }

    def test_the_unet_mag1c_sas_row(self):
        assert REFERENCES["herec_unet_mag1c_sas"].metrics == {
            "recall": (56.41, 7.0),
            "precision": (34.62, 7.4),
            "f1": (42.54, 6.7),
            "f1_strong": (61.38, 7.7),
        }

    def test_the_original_mag1c_baseline_has_no_spread(self):
        assert REFERENCES["herec_mag1c_column_wise"].metrics == {
            "recall": (58.42, None),
            "precision": (30.57, None),
            "f1": (40.14, None),
            "f1_strong": (67.50, None),
        }

    def test_the_comparability_caveats_are_recorded_with_every_row(self):
        for name in ("herec_linknet_mag1c_sas", "herec_unet_mag1c_sas"):
            caveats = " ".join(REFERENCES[name].caveats)
            assert "label size" in caveats
            assert "assumption" in caveats
            assert "Mag1c-SAS" in caveats

    def test_herec_rows_cite_page_8_of_the_verified_2026_publication(self):

        for name in ("herec_mag1c_column_wise", "herec_unet_mag1c_sas", "herec_linknet_mag1c_sas"):
            assert "p. 8" in REFERENCES[name].citation, name

    def test_the_run_count_and_threshold_are_now_verified_not_assumed(self):

        for name in ("herec_unet_mag1c_sas", "herec_linknet_mag1c_sas"):
            caveats = " ".join(REFERENCES[name].caveats)
            assert "5 repeated training runs" in caveats
            assert "threshold is 0.5" in caveats
            assert "not stated" not in caveats.lower()

    def test_the_f1_aggregation_caveat_still_flags_an_assumption(self):

        for name in ("herec_unet_mag1c_sas", "herec_linknet_mag1c_sas"):
            caveats = " ".join(REFERENCES[name].caveats)
            assert "for all plumes" in caveats
            assert "pooled" in caveats


class TestEveryRow:
    def test_rows_are_reference_records_with_the_expected_metric_keys(self):
        for name, row in REFERENCES.items():
            assert isinstance(row, Reference)
            expected = STARCOP_KEYS if name.startswith("starcop") else HEREC_KEYS
            assert set(row.metrics) == expected, name

    def test_every_row_cites_its_table_and_source(self):
        for name, row in REFERENCES.items():
            assert re.search(r"Table (2|I)\b", row.citation), name
            assert row.source in ("starcop", "herec"), name
            assert row.label, name

    def test_starcop_rows_cite_a_page(self):
        for name, row in REFERENCES.items():
            if name.startswith("starcop"):
                assert "page 10" in row.citation, name
