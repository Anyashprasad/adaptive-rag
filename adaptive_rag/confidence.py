from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from .models import QueryAnalysis, SearchHit


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class PlattCalibrator:
    """Tiny dependency-free probability calibrator: sigmoid(a * raw + b)."""

    a: float = 1.0
    b: float = 0.0

    def predict(self, raw: float) -> float:
        return _sigmoid(self.a * raw + self.b)

    def fit(self, raw_scores: list[float], labels: list[int], lr: float = 0.08, epochs: int = 1200) -> "PlattCalibrator":
        if not raw_scores or len(raw_scores) != len(labels) or len(set(labels)) < 2:
            return self
        a, b = self.a, self.b
        n = float(len(raw_scores))
        for _ in range(epochs):
            da = db = 0.0
            for x, y in zip(raw_scores, labels):
                p = _sigmoid(a * x + b)
                err = p - float(y)
                da += err * x
                db += err
            a -= lr * da / n
            b -= lr * db / n
        self.a, self.b = float(a), float(b)
        return self


class ConfidencePolicy:
    def __init__(self, calibration_path: str | Path | None = None):
        self.calibrators = {"lexical": PlattCalibrator(), "semantic": PlattCalibrator()}
        if calibration_path:
            self.load(calibration_path)

    def threshold(self, channel: str, analysis: QueryAnalysis) -> float:
        # Threshold is a target calibrated correctness probability and remains query-dependent.
        if channel == "lexical":
            return round(max(0.50, 0.64 - 0.13 * analysis.exactness + 0.05 * analysis.conceptuality), 4)
        return round(max(0.52, 0.64 - 0.10 * analysis.conceptuality + 0.08 * analysis.exactness), 4)

    def raw_score(self, channel: str, hits: list[SearchHit], analysis: QueryAnalysis) -> float:
        if not hits:
            return 0.0
        top = hits[0].score
        second = hits[1].score if len(hits) > 1 else 0.0
        if channel == "semantic":
            top_quality = max(0.0, min(1.0, (top + 1.0) / 2.0))
            margin = max(0.0, min(1.0, top - second))
            confidence = 0.68 * top_quality + 0.20 * margin + 0.12 * analysis.conceptuality
            # Dense similarity alone is less trustworthy when the query contains brittle exact constraints.
            confidence *= 1.0 - 0.22 * analysis.exactness
        else:
            top_quality = top / (top + 5.0) if top > 0 else 0.0
            margin = max(0.0, min(1.0, (top - second) / max(abs(top), 1e-6)))
            exact_coverage = float(hits[0].features.get("exact_coverage", 0.0))
            token_coverage = float(hits[0].features.get("token_coverage", 0.0))
            confidence = 0.32 * top_quality + 0.16 * margin + 0.42 * exact_coverage + 0.10 * token_coverage
        return max(0.0, min(1.0, confidence))

    def score(self, channel: str, hits: list[SearchHit], analysis: QueryAnalysis) -> float:
        raw = self.raw_score(channel, hits, analysis)
        # Calibrators are identity-ish until fitted; calibration is optional but persistable.
        cal = self.calibrators[channel]
        if cal.a == 1.0 and cal.b == 0.0:
            return round(raw, 4)
        return round(cal.predict(raw), 4)

    def fit_channel(self, channel: str, raw_scores: list[float], labels: list[int]) -> None:
        self.calibrators[channel] = PlattCalibrator().fit(raw_scores, labels)

    def save(self, path: str | Path) -> None:
        data = {k: {"a": v.a, "b": v.b} for k, v in self.calibrators.items()}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for channel in ("lexical", "semantic"):
            if channel in data:
                self.calibrators[channel] = PlattCalibrator(float(data[channel]["a"]), float(data[channel]["b"]))
