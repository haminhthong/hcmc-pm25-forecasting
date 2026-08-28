import numpy as np

from src.train import expanding_folds


def test_expanding_folds_never_train_on_future():
    folds = list(expanding_folds(size=100, folds=4, minimum_train_rows=40))
    assert len(folds) == 4
    for train_indices, validation_indices in folds:
        assert np.max(train_indices) < np.min(validation_indices)
        assert len(set(train_indices) & set(validation_indices)) == 0
