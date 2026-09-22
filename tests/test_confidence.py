from adaptive_rag.confidence import PlattCalibrator, ConfidencePolicy


def test_platt_calibrator_learns_probability_ordering():
    cal = PlattCalibrator().fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert cal.predict(0.9) > cal.predict(0.1)
    assert cal.a > 0


def test_confidence_roundtrip(tmp_path):
    policy = ConfidencePolicy()
    policy.fit_channel("lexical", [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    out = tmp_path / "cal.json"
    policy.save(out)
    loaded = ConfidencePolicy(out)
    assert abs(loaded.calibrators["lexical"].a - policy.calibrators["lexical"].a) < 1e-9
