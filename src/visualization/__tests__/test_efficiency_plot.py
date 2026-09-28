import pytest

from visualization.efficiency_plot import (
    ENERGY_RUN_NAMES,
    PR_AUC_RUN_NAMES,
    EnergyRow,
    PrAucPoint,
    energy_ratio,
    energy_rows,
    load_param_count,
    pr_auc_points,
)
from visualization.figure_common import EXPERIMENT_NAME, open_client

_PARAMETERS = {"E1": 487_361, "E2": 6_629_233, "E3": 856_635}


class TestRunNames:
    def test_r3_has_no_e1_pr_auc_run(self):
        assert set(PR_AUC_RUN_NAMES["raw-full"]) == {"E2", "E3"}

    def test_mini_pr_auc_comes_from_the_cross_tier_evaluation_on_the_raw_test_split(self):

        assert PR_AUC_RUN_NAMES["mini"]["E1"] == "E1-mini-on-raw-full-eval"
        assert PR_AUC_RUN_NAMES["r2"]["E1"] == "E1-r2-eval"
        assert PR_AUC_RUN_NAMES["raw-full"]["E2"] == "E2-raw-full-eval"

    def test_energy_is_compared_for_the_selected_models(self):
        assert {a: run["tier"] for a, run in ENERGY_RUN_NAMES.items()} == {
            "E1": "mini",
            "E2": "raw-full",
            "E3": "raw-full",
        }

    def test_each_energy_model_has_a_cuda_and_a_cpu_scene_run(self):
        for run in ENERGY_RUN_NAMES.values():
            assert run["cuda"].endswith("-scene-full_scene-cuda")
            assert run["cpu"].endswith("-scene-full_scene-cpu")


class TestPrAucPoints:
    def test_orders_points_within_a_tier_by_parameter_count(self):

        points = pr_auc_points({"r2": {"E1": 0.3153, "E2": 0.3786, "E3": 0.3869}}, _PARAMETERS)

        assert [p.architecture for p in points] == ["E1", "E3", "E2"]

    def test_keeps_the_tiers_in_the_order_they_are_given(self):
        points = pr_auc_points({"raw-full": {"E2": 0.4933}, "mini": {"E1": 0.5523}}, _PARAMETERS)

        assert [p.tier for p in points] == ["raw-full", "mini"]

    def test_does_not_invent_a_point_for_an_architecture_a_tier_lacks(self):
        points = pr_auc_points({"raw-full": {"E2": 0.4933, "E3": 0.5218}}, _PARAMETERS)

        assert [p.architecture for p in points] == ["E3", "E2"]

    def test_carries_the_parameter_count_and_the_pr_auc_of_each_point(self):
        points = pr_auc_points({"mini": {"E1": 0.5523}}, _PARAMETERS)

        assert points == [PrAucPoint("mini", "E1", 487_361, 0.5523)]

    def test_names_the_architecture_whose_parameter_count_is_missing(self):
        with pytest.raises(KeyError, match="E9"):
            pr_auc_points({"mini": {"E9": 0.5}}, _PARAMETERS)


class TestEnergyRatio:
    def test_divides_cpu_energy_by_gpu_energy(self):
        assert energy_ratio(cpu_joules=18.11, gpu_joules=1.25) == pytest.approx(14.488)

    def test_refuses_a_non_positive_gpu_energy(self):
        with pytest.raises(ValueError, match="GPU energy"):
            energy_ratio(cpu_joules=5.0, gpu_joules=0.0)


class TestEnergyRows:
    def test_builds_a_row_per_model_with_its_ratio_in_the_order_given(self):
        rows = energy_rows(
            {
                "E1": {"tier": "mini", "gpu_joules": 1.087, "cpu_joules": 6.871},
                "E3": {"tier": "raw-full", "gpu_joules": 0.936, "cpu_joules": 5.185},
            }
        )

        assert [row.architecture for row in rows] == ["E1", "E3"]
        assert rows[0] == EnergyRow("E1", "mini", gpu_joules=1.087, cpu_joules=6.871)
        assert rows[1].ratio == pytest.approx(5.5395, abs=1e-3)


class TestLoadParamCount:
    def test_reads_the_logged_parameter_count_as_an_integer(self, tmp_path):
        client = open_client(f"sqlite:///{tmp_path / 'mlflow.db'}")
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        run_id = client.create_run(experiment_id).info.run_id
        client.log_param(run_id, "param_count", 487361)

        assert load_param_count(client, run_id) == 487_361
