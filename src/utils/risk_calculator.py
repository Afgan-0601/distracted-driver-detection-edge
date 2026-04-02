"""
risk_calculator.py
==================
Utility for computing driver distraction risk from model predictions.

Key concepts
------------
* **Base risk score** – a static, class-level danger value (0-100) that
  captures how dangerous a particular behaviour is independent of time.
* **Confidence weight** – scales the base risk by the model's prediction
  confidence so that uncertain detections produce lower risk.
* **Sustained distraction multiplier** – a piecewise-linear function that
  amplifies risk the longer the driver stays distracted.
* **Composite risk score** – the final 0-100 score used for alerting.
* **Risk level** – categorical label (SAFE / LOW / MEDIUM / HIGH / CRITICAL).

Usage example
-------------
>>> from src.utils.risk_calculator import RiskCalculator
>>> calc = RiskCalculator()
>>> result = calc.evaluate(class_id=1, confidence=0.92, sustained_seconds=6.0)
>>> print(result)
RiskResult(class_key='c1', label='texting_right', base_risk=95.0,
           composite_risk=100.0, risk_level='CRITICAL',
           alert=True, sustained_seconds=6.0)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import (
    CLASS_BASE_RISK,
    CLASS_NAMES,
    INDEX_TO_CLASS,
    RISK_THRESHOLDS,
    SUSTAINED_RISK_SCHEDULE,
)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RiskResult:
    """Immutable snapshot of a single risk evaluation."""

    class_key: str          # e.g. "c1"
    label: str              # e.g. "texting_right"
    confidence: float       # model confidence [0, 1]
    base_risk: float        # class static risk score [0, 100]
    sustained_seconds: float  # how long the driver has been in this state
    sustained_multiplier: float
    weighted_risk: float    # base_risk × confidence × multiplier, capped at 100
    composite_risk: float   # final score after smoothing (0-100)
    risk_level: str         # SAFE | LOW | MEDIUM | HIGH | CRITICAL
    alert: bool             # True when level ≥ HIGH

    @property
    def is_distracted(self) -> bool:
        return self.class_key != "c0"

    def to_dict(self) -> dict:
        return {
            "class_key": self.class_key,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "base_risk": self.base_risk,
            "sustained_seconds": round(self.sustained_seconds, 2),
            "sustained_multiplier": round(self.sustained_multiplier, 3),
            "weighted_risk": round(self.weighted_risk, 2),
            "composite_risk": round(self.composite_risk, 2),
            "risk_level": self.risk_level,
            "alert": self.alert,
            "is_distracted": self.is_distracted,
        }


@dataclass
class _DistactionState:
    """Internal mutable state tracked per driver session."""
    class_key: str = "c0"
    start_ts: float = field(default_factory=time.monotonic)
    smoothed_risk: float = 0.0  # exponential moving average


# ─── Core Calculator ──────────────────────────────────────────────────────────

class RiskCalculator:
    """
    Stateful risk calculator for a single driver camera feed.

    The calculator tracks *sustained distraction time* automatically.
    Call :meth:`evaluate` once per inference frame and it will:

    1. Detect whether the behaviour class changed (resets the timer).
    2. Compute the sustained multiplier from the elapsed time.
    3. Compute the weighted risk score.
    4. Apply exponential smoothing to reduce jitter between frames.
    5. Resolve the risk level and alert flag.

    Parameters
    ----------
    smoothing_alpha : float
        EMA smoothing factor (0 < α ≤ 1).  Higher = less smoothing.
    alert_level : str
        Minimum risk level that triggers ``alert=True``.  One of
        ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"CRITICAL"``.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.4,
        alert_level: str = "HIGH",
    ) -> None:
        if not (0 < smoothing_alpha <= 1):
            raise ValueError("smoothing_alpha must be in (0, 1]")
        if alert_level not in RISK_THRESHOLDS:
            raise ValueError(
                f"alert_level must be one of {list(RISK_THRESHOLDS)}"
            )
        self._alpha = smoothing_alpha
        self._alert_threshold = RISK_THRESHOLDS[alert_level]
        self._state = _DistactionState()

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        class_id: int,
        confidence: float,
        *,
        override_sustained_seconds: Optional[float] = None,
    ) -> RiskResult:
        """
        Compute the risk for a single prediction frame.

        Parameters
        ----------
        class_id : int
            Integer class index returned by the model (0-9).
        confidence : float
            Model softmax confidence for the predicted class [0, 1].
        override_sustained_seconds : float, optional
            If provided, bypasses the internal timer — useful for batch /
            offline evaluation where inference timestamps are known.

        Returns
        -------
        RiskResult
        """
        class_key = INDEX_TO_CLASS.get(class_id)
        if class_key is None:
            raise ValueError(
                f"class_id {class_id} is out of range (0-{len(INDEX_TO_CLASS)-1})"
            )
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")

        # Update sustained timer
        if class_key != self._state.class_key:
            self._state.class_key = class_key
            self._state.start_ts = time.monotonic()

        if override_sustained_seconds is not None:
            sustained = float(override_sustained_seconds)
        else:
            sustained = time.monotonic() - self._state.start_ts

        base_risk = CLASS_BASE_RISK[class_key]
        multiplier = self._sustained_multiplier(sustained)
        weighted = min(base_risk * confidence * multiplier, 100.0)

        # Exponential moving average for temporal smoothing
        self._state.smoothed_risk = (
            self._alpha * weighted
            + (1.0 - self._alpha) * self._state.smoothed_risk
        )
        composite = self._state.smoothed_risk
        level = self._risk_level(composite)

        return RiskResult(
            class_key=class_key,
            label=CLASS_NAMES[class_key],
            confidence=confidence,
            base_risk=base_risk,
            sustained_seconds=sustained,
            sustained_multiplier=multiplier,
            weighted_risk=weighted,
            composite_risk=composite,
            risk_level=level,
            alert=composite >= self._alert_threshold,
        )

    def evaluate_batch(
        self,
        predictions: list[tuple[int, float]],
        *,
        frame_interval_seconds: float = 0.1,
    ) -> list[RiskResult]:
        """
        Evaluate a list of (class_id, confidence) tuples sequentially,
        simulating real-time playback at *frame_interval_seconds*.

        Useful for offline post-processing of recorded trips.
        """
        results: list[RiskResult] = []
        elapsed = 0.0
        for class_id, confidence in predictions:
            results.append(
                self.evaluate(
                    class_id,
                    confidence,
                    override_sustained_seconds=elapsed,
                )
            )
            elapsed += frame_interval_seconds
        return results

    def reset(self) -> None:
        """Reset internal state (call between driver sessions)."""
        self._state = _DistactionState()

    @property
    def current_composite_risk(self) -> float:
        return round(self._state.smoothed_risk, 2)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _sustained_multiplier(seconds: float) -> float:
        """
        Piecewise-linear interpolation over SUSTAINED_RISK_SCHEDULE.

        Example schedule: [(0s, ×1.0), (2s, ×1.2), (5s, ×1.5), (10s, ×2.0)]
        At 3.5 s → interpolates between (2s,1.2) and (5s,1.5).
        """
        schedule = SUSTAINED_RISK_SCHEDULE
        if seconds <= schedule[0][0]:
            return schedule[0][1]
        for i in range(1, len(schedule)):
            t0, m0 = schedule[i - 1]
            t1, m1 = schedule[i]
            if seconds <= t1:
                frac = (seconds - t0) / (t1 - t0)
                return m0 + frac * (m1 - m0)
        return schedule[-1][1]

    @staticmethod
    def _risk_level(score: float) -> str:
        """Map a composite score to a categorical risk level."""
        thresholds = RISK_THRESHOLDS
        if score < thresholds["LOW"]:
            return "SAFE"
        if score < thresholds["MEDIUM"]:
            return "LOW"
        if score < thresholds["HIGH"]:
            return "MEDIUM"
        if score < thresholds["CRITICAL"]:
            return "HIGH"
        return "CRITICAL"


# ─── Convenience functions ────────────────────────────────────────────────────

def score_to_level(score: float) -> str:
    """Stateless helper — convert a raw 0-100 score to a risk level string."""
    return RiskCalculator._risk_level(score)


def class_base_risks() -> dict[str, dict]:
    """
    Return a summary table of all classes with their base risk scores and
    default risk levels (at full confidence, no sustained time).
    """
    rows = {}
    for idx, key in INDEX_TO_CLASS.items():
        base = CLASS_BASE_RISK[key]
        rows[key] = {
            "index": idx,
            "label": CLASS_NAMES[key],
            "base_risk": base,
            "default_level": score_to_level(base),
        }
    return rows


def trip_summary(results: list[RiskResult]) -> dict:
    """
    Aggregate statistics for a complete trip / video clip.

    Returns
    -------
    dict with keys: total_frames, distracted_frames, distracted_pct,
    max_composite_risk, mean_composite_risk, time_per_level (dict),
    alert_frames.
    """
    if not results:
        return {}

    level_counts: dict[str, int] = {
        "SAFE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0
    }
    total = len(results)
    distracted = sum(1 for r in results if r.is_distracted)
    alerts = sum(1 for r in results if r.alert)
    scores = [r.composite_risk for r in results]
    for r in results:
        level_counts[r.risk_level] += 1

    return {
        "total_frames": total,
        "distracted_frames": distracted,
        "distracted_pct": round(distracted / total * 100, 1),
        "alert_frames": alerts,
        "max_composite_risk": round(max(scores), 2),
        "mean_composite_risk": round(sum(scores) / total, 2),
        "time_per_level": {
            lvl: round(cnt / total * 100, 1)
            for lvl, cnt in level_counts.items()
        },
    }
