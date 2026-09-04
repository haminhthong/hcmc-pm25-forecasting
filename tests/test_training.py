from src.train import build_quality_gate


def test_quality_gate_evaluates_multi_criteria():
    config = {
        "quality_gate": {
            "minimum_mae_improvement": 0.05,
            "minimum_high_pm25_recall": 0.75,
            "maximum_rolling_mae_std": 1.0,
        }
    }
    passed_metrics = {"mae": 1.0, "high_pm25_recall": 0.8}
    persistence_metrics = {"mae": 2.0, "high_pm25_recall": 0.5}

    passed_gate = build_quality_gate(passed_metrics, persistence_metrics, 0.5, config)
    assert passed_gate["passes_baseline"] is True

    failed_metrics = {"mae": 1.98, "high_pm25_recall": 0.6}  # improvement < 5%
    failed_gate = build_quality_gate(failed_metrics, persistence_metrics, 0.5, config)
    assert failed_gate["passes_baseline"] is False
