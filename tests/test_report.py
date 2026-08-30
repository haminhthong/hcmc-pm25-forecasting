from src.report import build_markdown


def test_report_contains_model_and_quality_gate():
    evaluation = {
        "backtest": {
            "random_forest": {"mae_mean": 1.2, "mae_std": 0.2, "rmse_mean": 1.5}
        },
        "champion_test": {"mae": 1.1, "rmse": 1.4, "macro_f1": 0.7, "high_pm25_recall": 0.8},
        "quality_gate": {"status": "đạt"},
    }
    report = build_markdown(evaluation)
    assert "random_forest" in report
    assert "Quality gate: **đạt**" in report
