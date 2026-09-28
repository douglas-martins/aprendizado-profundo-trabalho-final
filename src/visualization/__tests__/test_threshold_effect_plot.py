import pytest

from visualization.figure_common import EXPERIMENT_NAME, TIER_LABELS, open_client
from visualization.threshold_effect_plot import (
    SCENE_RUN_NAMES,
    ThresholdEffect,
    load_scene_metrics,
    threshold_effect_rows,
)


def _metrics(f1_fixed, f1_validation, fpr_fixed, fpr_validation):
    return {
        "f1_fixed": f1_fixed,
        "f1_validation": f1_validation,
        "fpr_fixed": fpr_fixed,
        "fpr_validation": fpr_validation,
    }


class TestSceneRunNames:
    def test_covers_the_eight_models_scored_on_the_real_test_scenes(self):
        assert len(SCENE_RUN_NAMES) == 8
        assert len({label for label, _ in SCENE_RUN_NAMES}) == 8

    def test_r3_models_come_first_then_r2_then_mini(self):
        tiers = [label.split("(")[1].rstrip(")") for label, _ in SCENE_RUN_NAMES]

        assert tiers == ["R3", "R3", "R2", "R2", "R2", "mini", "mini", "mini"]

    def test_every_label_matches_the_architecture_and_tier_in_its_run_name(self):
        tier_by_label = {label: tier for tier, label in TIER_LABELS.items()}
        for label, run_name in SCENE_RUN_NAMES:
            architecture, tier_label = label.split(" (")
            tier = tier_by_label[tier_label.rstrip(")")]
            assert run_name == f"{architecture}-{tier}-on-raw-full-scene-full_scene"


class TestThresholdEffectRows:
    def test_computes_the_f1_gain_of_the_validation_threshold(self):
        rows = threshold_effect_rows({"E2 (R3)": _metrics(0.35, 0.62, 0.89, 0.27)})

        assert rows[0].f1_gain == pytest.approx(0.27)

    def test_computes_the_change_in_tile_false_positive_rate(self):
        rows = threshold_effect_rows({"E2 (R3)": _metrics(0.35, 0.62, 0.89, 0.27)})

        assert rows[0].fpr_change == pytest.approx(-0.62)

    def test_keeps_the_order_of_the_models_it_is_given(self):
        rows = threshold_effect_rows(
            {
                "E3 (R2)": _metrics(0.3, 0.4, 0.9, 0.5),
                "E1 (mini)": _metrics(0.5, 0.6, 0.3, 0.4),
            }
        )

        assert [row.label for row in rows] == ["E3 (R2)", "E1 (mini)"]

    def test_carries_the_four_raw_numbers_through_unchanged(self):
        rows = threshold_effect_rows({"E1 (R2)": _metrics(0.1652, 0.3929, 0.9148, 0.5057)})

        assert rows == [ThresholdEffect("E1 (R2)", 0.1652, 0.3929, 0.9148, 0.5057)]

    def test_names_the_model_and_metric_that_is_missing(self):
        incomplete = {"f1_fixed": 0.3, "f1_validation": 0.4, "fpr_fixed": 0.9}

        with pytest.raises(KeyError, match=r"E2 \(R3\).*fpr_validation"):
            threshold_effect_rows({"E2 (R3)": incomplete})


class TestLoadSceneMetrics:
    def test_reads_the_four_scene_metrics_from_a_real_run(self, tmp_path):
        client = open_client(f"sqlite:///{tmp_path / 'mlflow.db'}")
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        run_id = client.create_run(experiment_id).info.run_id
        for key, value in {
            "at_0p5_all_scenes_f1": 0.35,
            "at_val_pick_all_scenes_f1": 0.62,
            "at_0p5_tile_fpr": 0.89,
            "at_val_pick_tile_fpr": 0.27,
        }.items():
            client.log_metric(run_id, key, value)

        assert load_scene_metrics(client, run_id) == _metrics(0.35, 0.62, 0.89, 0.27)
