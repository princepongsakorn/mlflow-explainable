"""Smoke tests for the ``mlflow_explainable`` contract.

These tests do not rely on any network access — they use a temp-directory
MLflow tracking URI and the toy classifier fixture.
"""

from __future__ import annotations

import io

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from mlflow_explainable import (
    ExplainableModel,
    ExplainableModelError,
    NoActiveRunError,
    log_explainable_model,
)
from mlflow_explainable.contract import _autodetect_code_paths


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------
def test_base_contract_is_abstract():
    """``ExplainableModel`` must reject direct use — subclasses must implement."""
    m = ExplainableModel()
    with pytest.raises(NotImplementedError):
        m.predict(None, pd.DataFrame())
    with pytest.raises(NotImplementedError):
        m.shap_explain(pd.DataFrame())


# ---------------------------------------------------------------------------
# log_explainable_model — sklearn path
# ---------------------------------------------------------------------------
def test_log_without_active_run_raises(toy_dataset):
    X, y = toy_dataset
    model = RandomForestClassifier(n_estimators=10).fit(X, y)
    with pytest.raises(NoActiveRunError):
        log_explainable_model(model, X)


def test_log_rf_and_load(tmp_mlflow_tracking_uri, toy_dataset):
    X, y = toy_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    with mlflow.start_run() as run:
        info = log_explainable_model(model, X, registered_name="test-rf")
        run_id = run.info.run_id

    assert info is not None  # ModelInfo (MLflow 2.x/3.x have different shapes)

    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    out = loaded.predict(X.head(5))
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["Y_proba", "Y_class"]
    assert len(out) == 5
    assert ((0.0 <= out["Y_proba"]) & (out["Y_proba"] <= 1.0)).all()
    assert set(out["Y_class"].unique()).issubset({0, 1})


def test_shap_explain_contract(tmp_mlflow_tracking_uri, toy_dataset):
    X, y = toy_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    with mlflow.start_run() as run:
        log_explainable_model(model, X, registered_name="test-rf-shap")
        run_id = run.info.run_id

    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    impl = loaded._model_impl.python_model

    out = impl.shap_explain(X.head(3))
    assert set(out.keys()) == {"values", "base_values", "data"}
    assert out["values"].shape[0] == 3
    assert out["values"].shape[1] == X.shape[1]
    assert out["data"].shape == (3, X.shape[1])


# ---------------------------------------------------------------------------
# Auto code paths
# ---------------------------------------------------------------------------
def test_autodetect_skips_third_party():
    """sklearn estimators should not pull sklearn source files."""
    rf = RandomForestClassifier(n_estimators=2)
    paths = _autodetect_code_paths(rf)
    assert not any("sklearn" in p for p in paths), f"sklearn leaked: {paths}"


def test_autodetect_detects_user_class(tmp_path):
    # Define a class inside this test module — its source file is this file.
    class Holder:
        def __init__(self, n):
            self.n = n

    obj = Holder(5)
    paths = _autodetect_code_paths(obj)
    # Should pick up THIS test file (a user module, not __main__ during pytest).
    assert any(p.endswith("test_contract.py") for p in paths), paths


# ---------------------------------------------------------------------------
# Pickle roundtrip
# ---------------------------------------------------------------------------
def test_pickle_roundtrip_self_test_passes_for_rf(toy_dataset):
    """Sklearn RF + TreeExplainer must survive a cloudpickle roundtrip."""
    from mlflow_explainable.contract import _AutoExplainableModel
    import shap

    X, y = toy_dataset
    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    explainer = shap.Explainer(rf, X)
    pm = _AutoExplainableModel(rf, explainer, list(X.columns))
    buf = io.BytesIO()
    cloudpickle.dump(pm, buf)
    buf.seek(0)
    pm2 = cloudpickle.load(buf)
    out = pm2.predict(None, X.head(2))
    assert list(out.columns) == ["Y_proba", "Y_class"]


# ---------------------------------------------------------------------------
# Generic tree-model coverage — GradientBoosting + XGBoost
# ---------------------------------------------------------------------------
def test_log_gb_and_load(tmp_mlflow_tracking_uri, toy_dataset):
    """GradientBoostingClassifier — same sklearn surface as RF, should work
    through the generic contract without any code changes."""
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = toy_dataset
    gb = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y)

    with mlflow.start_run() as run:
        log_explainable_model(gb, X, registered_name="test-model")
        run_id = run.info.run_id

    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    out = loaded.predict(X.head(4))
    assert list(out.columns) == ["Y_proba", "Y_class"]
    assert len(out) == 4

    impl = loaded._model_impl.python_model
    exp = impl.shap_explain(X.head(3))
    assert exp["values"].shape[0] == 3
    assert exp["values"].shape[1] == X.shape[1]


def test_log_xgboost_and_load(tmp_mlflow_tracking_uri, toy_dataset):
    """xgboost.XGBClassifier — different package, same sklearn-compatible
    interface. Verifies the contract is genuinely generic."""
    xgb = pytest.importorskip("xgboost")

    X, y = toy_dataset
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=0,
    ).fit(X, y)

    with mlflow.start_run() as run:
        log_explainable_model(
            model,
            X,
            registered_name="test-model",
            # Pin xgboost so the artifact records its true dependency surface.
            extra_pip_requirements=["xgboost"],
        )
        run_id = run.info.run_id

    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    out = loaded.predict(X.head(4))
    assert list(out.columns) == ["Y_proba", "Y_class"]
    assert len(out) == 4
    assert ((0.0 <= out["Y_proba"]) & (out["Y_proba"] <= 1.0)).all()

    impl = loaded._model_impl.python_model
    exp = impl.shap_explain(X.head(3))
    assert exp["values"].shape[0] == 3
    assert exp["values"].shape[1] == X.shape[1]
    # base_values should exist regardless of explainer type
    assert exp["base_values"] is not None


def test_autodetect_skips_xgboost():
    """An XGBoost booster should not leak xgboost source files into code_paths."""
    xgb = pytest.importorskip("xgboost")
    model = xgb.XGBClassifier(n_estimators=2, verbosity=0)
    # Fit on a tiny array so the booster is materialised.
    X = np.random.RandomState(0).randn(20, 4)
    y = (X.sum(axis=1) > 0).astype(int)
    model.fit(X, y)
    paths = _autodetect_code_paths(model)
    assert not any("xgboost" in p for p in paths), f"xgboost leaked: {paths}"


# ---------------------------------------------------------------------------
# registered_name safety net
# ---------------------------------------------------------------------------
def test_registered_name_auto_derived_from_experiment(
    tmp_mlflow_tracking_uri, toy_dataset
):
    """If registered_name is omitted, the library should auto-derive it from
    the active experiment name and emit a UserWarning."""
    X, y = toy_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    mlflow.set_experiment("my-derived-exp")
    with mlflow.start_run():
        with pytest.warns(UserWarning, match="experiment name 'my-derived-exp'"):
            log_explainable_model(model, X)


def test_registered_name_auto_random_when_default_experiment(
    tmp_mlflow_tracking_uri, toy_dataset, monkeypatch
):
    """When the experiment is 'Default' (i.e. no meaningful name to derive
    from), the library falls back to a random UUID name and warns loudly."""
    X, y = toy_dataset
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    # Point at Default experiment by setting it explicitly.
    mlflow.set_experiment("Default")
    with mlflow.start_run():
        with pytest.warns(UserWarning, match="Auto-registered as 'explainable-model-"):
            log_explainable_model(model, X)


# ---------------------------------------------------------------------------
# feature_names.json artifact
# ---------------------------------------------------------------------------
def test_feature_names_artifact_written(tmp_mlflow_tracking_uri, toy_dataset):
    X, y = toy_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    with mlflow.start_run() as run:
        log_explainable_model(model, X, registered_name="test-feat-names")
        run_id = run.info.run_id

    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model/artifacts/feature_names.json"
    )
    import json
    with open(local) as f:
        names = json.load(f)
    assert names == list(X.columns)
