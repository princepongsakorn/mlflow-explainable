from __future__ import annotations

import os
import tempfile

import mlflow
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def tmp_mlflow_tracking_uri():
    """Point MLflow at a fresh SQLite store in a temp directory for the test.

    Recent MLflow releases refuse new runs on the plain file store, so the
    tracking data goes to SQLite and the artifacts to a folder beside it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.abspath(tmp)
        uri = f"sqlite:///{os.path.join(root, 'mlflow.db')}"
        prev = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.create_experiment(
            "test", artifact_location=f"file://{os.path.join(root, 'artifacts')}"
        )
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
