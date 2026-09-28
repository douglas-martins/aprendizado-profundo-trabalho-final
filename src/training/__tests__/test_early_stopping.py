from training.early_stopping import EarlyStopper


class TestEarlyStopper:
    def test_does_not_stop_while_improving(self):
        stopper = EarlyStopper(patience=2, mode="min")
        assert stopper.step(1.0) is False
        assert stopper.step(0.9) is False
        assert stopper.step(0.8) is False

    def test_stops_after_patience_epochs_without_improvement(self):
        stopper = EarlyStopper(patience=2, mode="min")
        stopper.step(1.0)
        assert stopper.step(1.1) is False
        assert stopper.step(1.2) is True

    def test_a_new_best_resets_the_counter(self):
        stopper = EarlyStopper(patience=2, mode="min")
        stopper.step(1.0)
        stopper.step(1.1)
        stopper.step(0.5)
        assert stopper.step(0.6) is False

    def test_mode_max_treats_higher_as_better(self):
        stopper = EarlyStopper(patience=1, mode="max")
        stopper.step(0.5)
        assert stopper.step(0.4) is True

    def test_tracks_whether_the_current_step_is_a_new_best(self):
        stopper = EarlyStopper(patience=3, mode="min")
        stopper.step(1.0)
        assert stopper.is_best is True
        stopper.step(1.5)
        assert stopper.is_best is False

    def test_mode_defaults_to_min(self):
        stopper = EarlyStopper(patience=1)
        stopper.step(0.5)
        assert stopper.step(0.6) is True

    def test_initial_state_before_any_step(self):
        stopper = EarlyStopper(patience=2, mode="min")
        assert stopper.counter == 0
        assert stopper.is_best is False

    def test_equal_value_does_not_count_as_improvement_in_min_mode(self):
        stopper = EarlyStopper(patience=1, mode="min")
        stopper.step(1.0)
        assert stopper.step(1.0) is True

    def test_equal_value_does_not_count_as_improvement_in_max_mode(self):
        stopper = EarlyStopper(patience=1, mode="max")
        stopper.step(1.0)
        assert stopper.step(1.0) is True


class TestStateDict:
    def test_a_restored_stopper_continues_exactly_like_the_original(self):
        original = EarlyStopper(patience=3, mode="max")
        for value in (0.2, 0.5, 0.4):
            original.step(value)

        restored = EarlyStopper(patience=3, mode="max")
        restored.load_state_dict(original.state_dict())

        for value in (0.45, 0.3, 0.6):
            assert restored.step(value) == original.step(value)
            assert restored.best == original.best
            assert restored.counter == original.counter
            assert restored.is_best == original.is_best

    def test_the_state_records_best_counter_and_is_best(self):
        stopper = EarlyStopper(patience=3, mode="min")
        stopper.step(1.0)
        stopper.step(2.0)

        assert stopper.state_dict() == {"best": 1.0, "counter": 1, "is_best": False}

    def test_loading_does_not_change_the_configured_patience_or_mode(self):
        source = EarlyStopper(patience=3, mode="min")
        source.step(1.0)
        target = EarlyStopper(patience=9, mode="max")

        target.load_state_dict(source.state_dict())

        assert (target.patience, target.mode) == (9, "max")
