import numpy as np
import pandas as pd

from src.train import build_quality_gate, expanding_folds, split_by_time


def test_expanding_folds_never_train_on_future():
    folds = list(expanding_folds(size=100, folds=4, minimum_train_rows=40))
    assert len(folds) == 4
    for train_indices, validation_indices in folds:
        assert np.max(train_indices) < np.min(validation_indices)
        assert len(set(train_indices) & set(validation_indices)) == 0


def test_time_split_preserves_order():
    frame = pd.DataFrame({"value": range(10)})
    train_frame, test_frame = split_by_time(frame, test_fraction=0.2)
    assert train_frame["value"].max() < test_frame["value"].min()
    assert len(test_frame) == 2


def test_quality_gate_requires_model_to_beat_persistence():
    passed = build_quality_gate(champion_mae=1.0, persistence_mae=2.0)
    failed = build_quality_gate(champion_mae=3.0, persistence_mae=2.0)
    assert passed["passes_baseline"] is True
    assert failed["passes_baseline"] is False
