from __future__ import annotations

import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def tmp_mlflow_tracking_uri():
    """Point MLflow at a fresh temp directory for the duration of the test."""
    with tempfile.TemporaryDirectory() as tmp:
        uri = f"file://{os.path.abspath(tmp)}"
        prev = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(uri)
        # File-store backend doesn't auto-create the Default experiment;
        # do it explicitly so start_run() works.
        mlflow.set_experiment("test")
        try:
            yield uri
        finally:
            mlflow.set_tracking_uri(prev)


@pytest.fixture()
def toy_dataset():
    rng = np.random.RandomState(0)
    n, d = 40, 6
    X = pd.DataFrame(rng.randn(n, d), columns=[f"b{i}" for i in range(d)])
    y = (X.sum(axis=1) > 0).astype(int).values
    return X, y
