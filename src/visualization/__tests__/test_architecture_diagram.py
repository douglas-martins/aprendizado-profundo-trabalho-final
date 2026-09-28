import pytest
import torch

from models.architectures import build_e1, build_e2, build_e3
from visualization.architecture_diagram import (
    Stage,
    _box_height,
    _last_conv_channels,
    network_plan,
    parameter_count,
    plan_labels,
)


@pytest.fixture(scope="module")
def models():
    return {
        "E1": build_e1().eval(),
        "E2": build_e2(pretrained=False).eval(),
        "E3": build_e3(pretrained=False).eval(),
    }


def _plan(models, name):

    return network_plan(name, models[name])


def _channels(stages):
    return [stage.channels for stage in stages]


def _sizes(stages):
    return [stage.size for stage in stages]


class TestParameterCount:
    def test_counts_every_parameter_of_each_configuration(self, models):
        assert parameter_count(models["E1"]) == 487_361
        assert parameter_count(models["E2"]) == 6_629_233
        assert parameter_count(models["E3"]) == 856_635


class TestE1Plan:
    def test_encoder_widens_while_halving_the_resolution(self, models):
        assert _channels(_plan(models, "E1").down) == [8, 16, 32, 64, 128]
        assert _sizes(_plan(models, "E1").down) == [128, 64, 32, 16, 8]

    def test_decoder_narrows_while_doubling_the_resolution(self, models):
        assert _channels(_plan(models, "E1").up) == [64, 32, 16, 8]
        assert _sizes(_plan(models, "E1").up) == [16, 32, 64, 128]

    def test_every_encoder_stage_but_the_bottleneck_feeds_a_decoder_stage(self, models):
        assert _plan(models, "E1").skips == [(3, 0), (2, 1), (1, 2), (0, 3)]

    def test_joins_skips_by_concatenation(self, models):
        assert _plan(models, "E1").skip_kind == "concat"

    def test_reports_name_parameters_and_channels_in_and_out(self, models):
        plan = _plan(models, "E1")

        assert plan.architecture == "E1"
        assert plan.parameters == 487_361
        assert (plan.in_channels, plan.out_channels) == (4, 1)


class TestE2Plan:
    def test_encoder_is_the_mobilenetv2_feature_stages(self, models):
        assert _channels(_plan(models, "E2").down) == [16, 24, 32, 96, 1280]
        assert _sizes(_plan(models, "E2").down) == [64, 32, 16, 8, 4]

    def test_decoder_is_the_unet_decoder_up_to_full_resolution(self, models):
        assert _channels(_plan(models, "E2").up) == [256, 128, 64, 32, 16]
        assert _sizes(_plan(models, "E2").up) == [8, 16, 32, 64, 128]

    def test_last_decoder_stage_has_no_skip_because_no_encoder_stage_is_that_large(self, models):
        assert _plan(models, "E2").skips == [(3, 0), (2, 1), (1, 2), (0, 3)]

    def test_joins_skips_by_concatenation(self, models):
        assert _plan(models, "E2").skip_kind == "concat"

    def test_reports_name_parameters_and_channels_in_and_out(self, models):
        plan = _plan(models, "E2")

        assert plan.architecture == "E2"
        assert plan.parameters == 6_629_233
        assert (plan.in_channels, plan.out_channels) == (4, 1)


class TestE3Plan:
    def test_encoder_is_the_mobilenetv3_small_minimal_feature_stages(self, models):
        assert _channels(_plan(models, "E3").down) == [16, 16, 24, 48, 576]
        assert _sizes(_plan(models, "E3").down) == [64, 32, 16, 8, 4]

    def test_decoder_is_the_linknet_decoder_up_to_full_resolution(self, models):
        assert _channels(_plan(models, "E3").up) == [48, 24, 16, 16, 32]
        assert _sizes(_plan(models, "E3").up) == [8, 16, 32, 64, 128]

    def test_joins_skips_by_addition_not_concatenation(self, models):
        assert _plan(models, "E3").skip_kind == "soma"

    def test_reports_name_parameters_and_channels_in_and_out(self, models):
        plan = _plan(models, "E3")

        assert plan.architecture == "E3"
        assert plan.parameters == 856_635
        assert (plan.in_channels, plan.out_channels) == (4, 1)


class TestPlansAreConsistent:
    @pytest.mark.parametrize("name", ["E1", "E2", "E3"])
    def test_a_skip_joins_stages_of_the_same_resolution(self, models, name):
        plan = _plan(models, name)

        for down_index, up_index in plan.skips:
            assert plan.down[down_index].size == plan.up[up_index].size

    @pytest.mark.parametrize("name", ["E1", "E2", "E3"])
    def test_decoder_ends_at_the_full_image_resolution(self, models, name):
        assert _plan(models, name).up[-1].size == 128

    def test_resolution_follows_the_image_size_it_is_given(self, models):
        plan = network_plan("E1", models["E1"], image_size=256)

        assert _sizes(plan.down) == [256, 128, 64, 32, 16]

    def test_pretrained_encoder_resolution_follows_the_image_size_it_is_given(self, models):
        plan = network_plan("E2", models["E2"], image_size=256)

        assert _sizes(plan.down) == [128, 64, 32, 16, 8]
        assert _sizes(plan.up) == [16, 32, 64, 128, 256]

    def test_refuses_a_model_that_is_not_one_of_the_three_configurations(self):
        with pytest.raises(TypeError, match="Linear"):
            network_plan("E9", torch.nn.Linear(1, 1))

    @pytest.mark.parametrize("name", ["E1", "E2", "E3"])
    def test_every_stage_size_is_a_whole_number_of_pixels(self, models, name):

        plan = _plan(models, name)

        assert all(type(stage.size) is int for stage in plan.down + plan.up)

    def test_stage_is_a_channels_and_size_pair(self):
        assert Stage(channels=8, size=128) == Stage(8, 128)


class TestLastConvChannels:
    def test_reads_the_width_of_the_last_convolution_of_a_three_convolution_block(self):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 1),
            torch.nn.Conv2d(2, 3, 1),
            torch.nn.Conv2d(3, 4, 1),
            torch.nn.ReLU(),
        )

        assert _last_conv_channels(block) == 4

    def test_ignores_a_transposed_convolution_that_comes_after_the_last_convolution(self):
        block = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.ConvTranspose2d(2, 9, 2))

        assert _last_conv_channels(block) == 2


class TestBoxHeight:
    def test_a_full_resolution_box_gets_the_maximum_height(self):
        assert _box_height(128, 128) == pytest.approx(0.9)

    def test_a_quarter_resolution_box_is_halfway_between_the_floor_and_the_maximum(self):
        assert _box_height(32, 128) == pytest.approx(0.53)

    def test_smaller_feature_maps_get_shorter_boxes(self):
        heights = [_box_height(size, 128) for size in (4, 8, 16, 32, 64, 128)]

        assert heights == sorted(heights)
        assert len(set(heights)) == len(heights)

    def test_height_depends_on_the_size_relative_to_the_image_not_on_its_absolute_value(self):
        assert _box_height(64, 128) == pytest.approx(_box_height(128, 256))


def _observed(model, modules):
    """(channels, size) of each module's output during one real forward pass."""
    seen = []
    handles = [
        module.register_forward_hook(lambda _m, _i, out: seen.append((out.shape[1], out.shape[-1])))
        for module in modules
    ]
    with torch.no_grad():
        model(torch.zeros(1, 4, 128, 128))
    for handle in handles:
        handle.remove()
    return seen


class TestPlanMatchesARealForwardPass:
    def test_e1_stages_have_the_feature_map_shapes_the_plan_claims(self, models):
        model, plan = models["E1"], _plan(models, "E1")
        down = [model.enc1, model.enc2, model.enc3, model.enc4, model.bottleneck]
        up = [model.dec4, model.dec3, model.dec2, model.dec1]

        seen = _observed(model, down + up)

        expected = [(s.channels, s.size) for s in plan.down + plan.up]
        assert seen == expected

    @pytest.mark.parametrize("name", ["E2", "E3"])
    def test_pretrained_family_stages_have_the_feature_map_shapes_the_plan_claims(
        self, models, name
    ):
        model, plan = models[name], _plan(models, name)

        seen_up = _observed(model, list(model.decoder.blocks))
        with torch.no_grad():
            features = model.encoder(torch.zeros(1, 4, 128, 128))[1:]

        assert [(f.shape[1], f.shape[-1]) for f in features] == [
            (s.channels, s.size) for s in plan.down
        ]
        assert seen_up == [(s.channels, s.size) for s in plan.up]


class TestPlanLabels:
    def test_e1_title_and_subtitle_say_it_has_no_pretraining_and_concatenates_skips(self, models):
        title, subtitle = plan_labels(_plan(models, "E1"))

        assert title == "E1: U-Net pequena, blocos convolucionais simples"
        assert subtitle == "487.361 parâmetros · sem pré-treino · saltos por concatenação"

    def test_e2_title_and_subtitle_call_the_pretrained_part_an_encoder(self, models):
        title, subtitle = plan_labels(_plan(models, "E2"))

        assert title == "E2: U-Net com encoder MobileNetV2"
        assert subtitle == (
            "6.629.233 parâmetros · encoder pré-treinado no ImageNet, "
            "1.ª convolução de 3 para 4 canais · saltos por concatenação"
        )

    def test_e3_title_and_subtitle_say_its_skips_are_sums(self, models):
        title, subtitle = plan_labels(_plan(models, "E3"))

        assert title == "E3: LinkNet com encoder MobileNetV3-small"
        assert subtitle == (
            "856.635 parâmetros · encoder pré-treinado no ImageNet, "
            "1.ª convolução de 3 para 4 canais · saltos por soma"
        )

    @pytest.mark.parametrize("name", ["E1", "E2", "E3"])
    def test_no_label_uses_the_portuguese_word_for_encoder(self, models, name):
        title, subtitle = plan_labels(_plan(models, name))

        assert "codificador" not in title + subtitle
