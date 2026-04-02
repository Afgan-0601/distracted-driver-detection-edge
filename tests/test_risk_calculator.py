"""Tests for RiskCalculator and related utilities."""

import pytest

from src.utils.risk_calculator import (
    RiskCalculator,
    RiskResult,
    class_base_risks,
    score_to_level,
    trip_summary,
)


class TestScoreToLevel:
    def test_safe_zone(self):
        assert score_to_level(0.0) == "SAFE"
        assert score_to_level(19.9) == "SAFE"

    def test_low_zone(self):
        assert score_to_level(20.0) == "LOW"
        assert score_to_level(44.9) == "LOW"

    def test_medium_zone(self):
        assert score_to_level(45.0) == "MEDIUM"
        assert score_to_level(69.9) == "MEDIUM"

    def test_high_zone(self):
        assert score_to_level(70.0) == "HIGH"
        assert score_to_level(84.9) == "HIGH"

    def test_critical_zone(self):
        assert score_to_level(85.0) == "CRITICAL"
        assert score_to_level(100.0) == "CRITICAL"


class TestSustainedMultiplier:
    def test_zero_seconds_returns_one(self):
        m = RiskCalculator._sustained_multiplier(0.0)
        assert m == pytest.approx(1.0)

    def test_interpolates_correctly(self):
        # At 3.5s, between (2s, 1.2) and (5s, 1.5): frac=0.5, expected=1.35
        m = RiskCalculator._sustained_multiplier(3.5)
        assert m == pytest.approx(1.35, rel=1e-4)

    def test_beyond_schedule_returns_max(self):
        m = RiskCalculator._sustained_multiplier(999.0)
        assert m == pytest.approx(2.5)

    def test_at_boundary(self):
        m = RiskCalculator._sustained_multiplier(10.0)
        assert m == pytest.approx(2.0)


class TestRiskCalculator:
    def setup_method(self):
        self.calc = RiskCalculator(smoothing_alpha=1.0)  # alpha=1 → no smoothing

    def test_safe_class_produces_zero_risk(self):
        result = self.calc.evaluate(0, 0.99, override_sustained_seconds=0.0)
        assert result.class_key == "c0"
        assert result.composite_risk == pytest.approx(0.0)
        assert result.risk_level == "SAFE"
        assert result.alert is False
        assert result.is_distracted is False

    def test_texting_produces_critical_risk(self):
        result = self.calc.evaluate(1, 0.95, override_sustained_seconds=15.0)
        assert result.risk_level == "CRITICAL"
        assert result.alert is True
        assert result.is_distracted is True

    def test_low_confidence_scales_risk_down(self):
        high_conf = self.calc.evaluate(1, 1.0, override_sustained_seconds=0.0)
        self.calc.reset()
        low_conf = self.calc.evaluate(1, 0.1, override_sustained_seconds=0.0)
        assert low_conf.weighted_risk < high_conf.weighted_risk

    def test_reset_clears_state(self):
        self.calc.evaluate(1, 0.9, override_sustained_seconds=12.0)
        self.calc.reset()
        result = self.calc.evaluate(0, 0.9, override_sustained_seconds=0.0)
        assert result.composite_risk == pytest.approx(0.0)

    def test_invalid_class_id_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            self.calc.evaluate(99, 0.5)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            self.calc.evaluate(0, 1.5)

    def test_evaluate_batch(self):
        predictions = [(0, 0.9)] * 10 + [(1, 0.95)] * 10
        results = self.calc.evaluate_batch(predictions, frame_interval_seconds=0.5)
        assert len(results) == 20
        # First frame is safe
        assert results[0].class_key == "c0"
        # Last frame is distracted
        assert results[-1].is_distracted is True

    def test_to_dict_has_required_keys(self):
        result = self.calc.evaluate(3, 0.8, override_sustained_seconds=1.0)
        d = result.to_dict()
        for key in [
            "class_key", "label", "confidence", "base_risk",
            "sustained_seconds", "composite_risk", "risk_level",
            "alert", "is_distracted",
        ]:
            assert key in d


class TestClassBaseRisks:
    def test_returns_all_ten_classes(self):
        data = class_base_risks()
        assert len(data) == 10

    def test_safe_class_is_low_risk(self):
        data = class_base_risks()
        assert data["c0"]["default_level"] == "SAFE"

    def test_texting_is_critical(self):
        data = class_base_risks()
        assert data["c1"]["default_level"] == "CRITICAL"


class TestTripSummary:
    def test_empty_list(self):
        assert trip_summary([]) == {}

    def test_all_safe(self):
        calc = RiskCalculator(smoothing_alpha=1.0)
        results = calc.evaluate_batch([(0, 0.99)] * 50, frame_interval_seconds=0.1)
        summary = trip_summary(results)
        assert summary["distracted_pct"] == 0.0
        assert summary["max_composite_risk"] == pytest.approx(0.0)

    def test_distracted_percentage(self):
        calc = RiskCalculator(smoothing_alpha=1.0)
        predictions = [(0, 0.9)] * 50 + [(1, 0.9)] * 50
        results = calc.evaluate_batch(predictions, frame_interval_seconds=0.1)
        summary = trip_summary(results)
        assert summary["distracted_frames"] == 50
        assert summary["distracted_pct"] == pytest.approx(50.0)
