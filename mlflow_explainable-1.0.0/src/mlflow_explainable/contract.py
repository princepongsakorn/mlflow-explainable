"""Contract for explainable models served by the platform's runtime.

This module defines a single user-facing API:

    log_explainable_model(model, background, *, registered_name=..., ...) -> ModelInfo

It logs the trained predictor *and* a SHAP explainer together as a single
self-contained ``mlflow.pyfunc`` artifact. The serving runtime can then load
*any* model logged through this API with a uniform interface — it does not
need to know whether the underlying model is a sklearn estimator, a torch
GCN, or anything else.

Contract surface exposed to the runtime
---------------------------------------
After ``loaded = mlflow.pyfunc.load_model(model_uri)``::

    # 1) Standard pyfunc predict — returns a DataFrame with two columns.
    df = loaded.predict(X)                           # columns: ["Y_proba", "Y_class"]

    # 2) Custom SHAP explanation — reach into the python_model implementation.
    impl = loaded._model_impl.python_model           # ExplainableModel instance
    out  = impl.shap_explain(X)                      # dict: values / base_values / data

The auto-wrapped predictor expects the underlying model to expose either
``predict_proba`` (preferred) or ``__call__`` returning class probabilities
of shape ``(n_samples, 2)``. ``predict`` is used to produce the discrete
class label.

User-facing usage examples
--------------------------
::

    # 1) sklearn-style model (RF, GB, XGB, ...)
    from sklearn.ensemble import RandomForestClassifier
    from mlflow_explainable import log_explainable_model

    model = RandomForestClassifier(...).fit(X_train, y_train)
    with mlflow.start_run():
        log_explainable_model(model, X_train, registered_name="sample-rf-crc")

    # 2) Custom torch model behind a tabular wrapper
    wrapper = GCNTabularWrapper(gcn, edge_index, device, feature_names)
    with mlflow.start_run():
        log_explainable_model(
            wrapper,
            X_train,
            registered_name="sample-gcn-crc",
            explainer_kwargs={"algorithm": "permutation"},
        )
"""

from __future__ import annotations

import inspect
import io
import json
import os
import sys
import tempfile
import uuid
import warnings
from typing import Any

import cloudpickle
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature

from .exceptions import ExplainableModelError, NoActiveRunError

__all__ = [
    "ExplainableModel",
    "log_explainable_model",
]


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------
class ExplainableModel(mlflow.pyfunc.PythonModel):
    """Base contract for explainable models served by the platform.

    Concrete subclasses must implement :py:meth:`predict` and
    :py:meth:`shap_explain`. Most callers should use :py:func:`log_explainable_model`,
    which auto-generates a subclass internally and does not require users to
    write their own.

    The runtime contract is::

        predict(self, context, model_input, params=None) -> pd.DataFrame
            # columns: ["Y_proba", "Y_class"]
            # index : preserved from model_input

        shap_explain(self, X) -> dict
            # keys:   "values"      shape (n, n_features)  or (n, n_features, n_classes)
            #         "base_values" shape ()                or (n_classes,) or (n, n_classes)
            #         "data"        shape (n, n_features)
    """

    def predict(self, context, model_input, params=None):  # noqa: D401, ARG002
        raise NotImplementedError("Subclasses must implement predict().")

    def shap_explain(self, X):  # noqa: D401, ARG002
        raise NotImplementedError("Subclasses must implement shap_explain().")


# ---------------------------------------------------------------------------
# Internal auto-wrapper
# ---------------------------------------------------------------------------
class _AutoExplainableModel(ExplainableModel):
    """Wraps a ``(predictor, explainer, feature_names)`` triple into the contract.

    The predictor must expose at least ``predict_proba`` returning probabilities
    of shape ``(n, 2)``. ``predict`` is used for the discrete class label; if
    the predictor does not expose ``predict``, ``argmax`` over the probabilities
    is used as a fallback.
    """

    def __init__(
        self,
        predictor: Any,
        explainer: shap.Explainer,
        feature_names: list[str],
    ):
        self._predictor = predictor
        self._explainer = explainer
        self._feature_names = list(feature_names)

    # ---- helpers -----------------------------------------------------------
    def _as_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return pd.DataFrame(arr, columns=self._feature_names)

    def _proba(self, X_df: pd.DataFrame) -> np.ndarray:
        p = self._predictor.predict_proba(X_df) if hasattr(self._predictor, "predict_proba") \
            else self._predictor(X_df)
        return np.asarray(p)

    def _class(self, X_df: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
        if hasattr(self._predictor, "predict"):
            return np.asarray(self._predictor.predict(X_df))
        return proba.argmax(axis=1)

    # ---- contract methods --------------------------------------------------
    def predict(self, context, model_input, params=None):  # noqa: ARG002
        X_df = self._as_dataframe(model_input)
        proba = self._proba(X_df)
        klass = self._class(X_df, proba)
        # binary task assumption — column 1 is the positive class
        y_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        return pd.DataFrame(
            {"Y_proba": y_proba, "Y_class": klass},
            index=X_df.index,
        )

    def shap_explain(self, X) -> dict:
        X_df = self._as_dataframe(X)
        # Prefer the new SHAP API which returns an Explanation object —
        # it works uniformly across TreeExplainer / PermutationExplainer /
        # KernelExplainer and exposes both .values and .base_values.
        try:
            explanation = self._explainer(X_df)
            return {
                "values": np.asarray(explanation.values),
                "base_values": np.asarray(explanation.base_values),
                "data": np.asarray(explanation.data),
            }
        except Exception:
            # Legacy fallback: explainers that only support .shap_values().
            values = self._explainer.shap_values(X_df)
            base = getattr(self._explainer, "expected_value", 0.0)
            return {
                "values": np.asarray(values),
                "base_values": np.asarray(base),
                "data": np.asarray(X_df.values),
            }


# ---------------------------------------------------------------------------
# Auto code_paths detection
# ---------------------------------------------------------------------------
_SKIP_MODULE_PREFIXES = (
    "builtins",
    "abc",
    "collections",
    "typing",
    "pickle",
    "cloudpickle",
    "joblib",
    "_thread",
    "mlflow",
    "shap",
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "sklearn",
    "xgboost",
    "lightgbm",
    "torch",
    "torch_geometric",
    "torchvision",
    "hyperopt",
    "mlflow_explainable",
    "numba",
    "llvmlite",
    "cython",
    "tqdm",
    "threading",
    "logging",
    "json",
    "io",
    "os",
    "sys",
    "re",
)


def _autodetect_code_paths(obj: Any, _seen: set | None = None, depth: int = 0) -> set[str]:
    """Walk ``obj`` recursively and collect source files of user-defined classes.

    The walk skips stdlib and well-known third-party modules to avoid bundling
    half of PyPI. Anything else with a resolvable ``__file__`` is collected.
    """
    MAX_DEPTH = 8
    if _seen is None:
        _seen = set()
    paths: set[str] = set()
    if depth > MAX_DEPTH:
        return paths

    oid = id(obj)
    if oid in _seen:
        return paths
    _seen.add(oid)

    # Cheap rejects.
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return paths

    cls = type(obj)
    mod_name = getattr(cls, "__module__", "") or ""

    is_third_party = any(
        mod_name == p or mod_name.startswith(p + ".") for p in _SKIP_MODULE_PREFIXES
    )
    if mod_name and mod_name != "__main__" and not is_third_party:
        mod = sys.modules.get(mod_name)
        file = getattr(mod, "__file__", None) if mod else None
        if file:
            paths.add(os.path.abspath(file))

    # Recurse — stop walking into third-party objects to bound the graph size.
    if is_third_party:
        return paths

    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                paths |= _autodetect_code_paths(k, _seen, depth + 1)
                paths |= _autodetect_code_paths(v, _seen, depth + 1)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                paths |= _autodetect_code_paths(item, _seen, depth + 1)
        elif hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                paths |= _autodetect_code_paths(v, _seen, depth + 1)
    except Exception:
        # Best-effort walk — any object that doesn't behave like a normal
        # python container is skipped silently.
        pass

    return paths


# ---------------------------------------------------------------------------
# Pickle roundtrip self-test
# ---------------------------------------------------------------------------
def _pickle_roundtrip_or_raise(obj: Any) -> None:
    try:
        buf = io.BytesIO()
        cloudpickle.dump(obj, buf)
        buf.seek(0)
        cloudpickle.load(buf)
    except Exception as exc:
        raise ExplainableModelError(
            "Pickle roundtrip self-test failed before logging the model.\n"
            f"Underlying error: {type(exc).__name__}: {exc}\n\n"
            "This usually means a class definition could not be imported from a file. "
            "If your class lives in a module that the autodetector skipped (or in a "
            "function body), pass extra_code_paths=['path/to/module.py'] to "
            "log_explainable_model()."
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def log_explainable_model(
    model: Any,
    background: pd.DataFrame | np.ndarray,
    *,
    registered_name: str | None = None,
    explainer: shap.Explainer | None = None,
    explainer_kwargs: dict | None = None,
    extra_code_paths: list[str] | None = None,
    extra_artifacts: dict[str, str] | None = None,
    artifact_path: str = "model",
    run_id: str | None = None,
    extra_pip_requirements: list[str] | None = None,
):
    """Log an explainable model (predictor + SHAP explainer) as a pyfunc artifact.

    Parameters
    ----------
    model
        The trained predictor. Must expose ``predict_proba`` (preferred) or
        ``__call__`` returning class probabilities of shape ``(n, 2)``. The
        ``predict`` method, if present, is used for the discrete class label.
    background
        Background data passed to ``shap.Explainer`` as the masker. For tabular
        data this is typically the training set (or a representative sample).
        Also used to (a) extract ``feature_names``, (b) build a signature, and
        (c) construct an ``input_example`` for the run.
    registered_name
        Name under which the logged model is registered in the MLflow
        registry. **Effectively required** — the serving runtime pulls models
        from the registry by name, so an unregistered artifact is
        non-discoverable. If you omit this argument, the library auto-derives
        one (from the active experiment name, or a random UUID as a last
        resort) and emits a ``UserWarning``.
    explainer
        Pre-built ``shap.Explainer``. If omitted, one is constructed via
        ``shap.Explainer(model, background, **explainer_kwargs)``.
    explainer_kwargs
        Extra kwargs forwarded to ``shap.Explainer(...)``. Common usage::

            explainer_kwargs={"algorithm": "permutation"}
    extra_code_paths
        Escape hatch — additional source files to include alongside the
        auto-detected ones. Normally not needed; only required if your class
        is defined inside a function body, an interactive shell, or a module
        whose prefix is on the autodetector's skip list.
    extra_artifacts
        Additional files to attach to the artifact. Standard MLflow
        ``artifacts={"key": "/local/path"}`` mapping.
    artifact_path
        Artifact subdirectory inside the run. Defaults to ``"model"``.
    run_id
        Target run ID. If omitted, the currently active run is used.
    extra_pip_requirements
        Additional pip requirements to record in the MLmodel file. Common
        usage::

            extra_pip_requirements=["torch", "torch_geometric"]

    Returns
    -------
    mlflow.models.model.ModelInfo
        The model info returned by ``mlflow.pyfunc.log_model``.

    Raises
    ------
    NoActiveRunError
        No active run and no ``run_id`` was provided.
    ExplainableModelError
        Explainer construction failed, or the pickle roundtrip self-test
        failed before logging.
    """
    # 1) Active run check.
    active = mlflow.active_run()
    if run_id is None and active is None:
        raise NoActiveRunError(
            "No active MLflow run. Wrap the call in `with mlflow.start_run():` "
            "or pass an explicit `run_id`."
        )

    # 1b) Resolve registered_name. We treat it as effectively required (the
    # serving runtime pulls by name) but provide a safety net rather than
    # hard-failing — a forgotten name in a 50-trial sweep would otherwise
    # waste a lot of compute.
    if not registered_name:
        derived: str | None = None
        if active is not None:
            try:
                exp = mlflow.get_experiment(active.info.experiment_id)
                if exp and exp.name and exp.name != "Default":
                    derived = exp.name
            except Exception:
                derived = None
        if derived:
            warnings.warn(
                f"`registered_name` not provided. Defaulting to the active "
                f"experiment name '{derived}'. Pass registered_name=... "
                f"explicitly to control this.",
                UserWarning,
                stacklevel=2,
            )
            registered_name = derived
        else:
            generated = f"explainable-model-{uuid.uuid4().hex[:8]}"
            warnings.warn(
                f"`registered_name` not provided and no active experiment name "
                f"to derive from. Auto-registered as '{generated}'. The serving "
                f"runtime pulls models by name — pass registered_name=... or "
                f"this artifact will be hard to discover.",
                UserWarning,
                stacklevel=2,
            )
            registered_name = generated

    # 2) Coerce background to DataFrame and pull feature names.
    if isinstance(background, pd.DataFrame):
        bg_df = background
    else:
        arr = np.asarray(background)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        bg_df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])
    feature_names = list(bg_df.columns)

    # 3) Build explainer if not supplied.
    #
    # shap.Explainer auto-detects sklearn tree models (RF, GB) and falls back
    # to a callable-based explainer for everything else. As of shap 0.49,
    # XGBoost's sklearn-style ``XGBClassifier`` is no longer auto-detected and
    # the meta-explainer rejects it with "not callable". To keep the user-facing
    # API uniform, we transparently retry with ``model.predict_proba`` (or
    # ``model.__call__``) when the first attempt fails.
    if explainer is None:
        kwargs = dict(explainer_kwargs or {})
        first_err: Exception | None = None
        try:
            explainer = shap.Explainer(model, bg_df, **kwargs)
        except Exception as exc:
            first_err = exc

        if explainer is None:
            callable_for_shap = None
            if hasattr(model, "predict_proba"):
                callable_for_shap = model.predict_proba
            elif callable(model):
                callable_for_shap = model

            if callable_for_shap is not None:
                try:
                    explainer = shap.Explainer(callable_for_shap, bg_df, **kwargs)
                except Exception as exc2:
                    raise ExplainableModelError(
                        f"Failed to construct shap.Explainer for model="
                        f"{type(model).__name__}.\n"
                        f"  First attempt (model object):  {first_err}\n"
                        f"  Fallback (predict_proba/callable): {exc2}"
                    ) from exc2
            else:
                raise ExplainableModelError(
                    f"Failed to construct shap.Explainer for model="
                    f"{type(model).__name__}: {first_err}"
                ) from first_err

    # 4) Auto-detect code paths from the predictor and the explainer object graph.
    detected = _autodetect_code_paths(model) | _autodetect_code_paths(explainer)
    if extra_code_paths:
        detected |= {os.path.abspath(p) for p in extra_code_paths}
    code_paths = sorted(detected)

    # 5) Wrap into the contract.
    python_model = _AutoExplainableModel(
        predictor=model,
        explainer=explainer,
        feature_names=feature_names,
    )

    # 6) Pickle self-test — fail loud and early.
    _pickle_roundtrip_or_raise(python_model)

    # 7) Build input_example + signature internally (user never sees this).
    sample = bg_df.head(3).copy()
    sample_out = python_model.predict(None, sample)
    signature = infer_signature(sample, sample_out)

    # 8) Always emit ``feature_names.json`` as a first-class artifact so that
    #    downstream services (kserve transformer, frontend) have a stable
    #    source of truth for the model's expected schema.
    with tempfile.TemporaryDirectory() as tmp:
        fn_path = os.path.join(tmp, "feature_names.json")
        with open(fn_path, "w") as fh:
            json.dump(feature_names, fh)

        artifacts: dict[str, str] = {"feature_names": fn_path}
        if extra_artifacts:
            artifacts.update(extra_artifacts)

        # 9) Log via pyfunc. ``code_path`` is the key piece — it makes the
        #    artifact self-contained so the runtime never needs to import the
        #    user's classes from its own codebase.
        # ``code_path`` was renamed to ``code_paths`` in MLflow 3.x — try the
        # new spelling first, fall back to the legacy kw for MLflow 2.x.
        log_kwargs = dict(
            python_model=python_model,
            artifacts=artifacts,
            input_example=sample,
            signature=signature,
            registered_model_name=registered_name,
            extra_pip_requirements=extra_pip_requirements,
        )
        # MLflow 3.x prefers ``name``; 2.x prefers ``artifact_path``.
        try:
            return mlflow.pyfunc.log_model(
                name=artifact_path,
                code_paths=code_paths or None,
                **log_kwargs,
            )
        except TypeError:
            return mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                code_path=code_paths or None,
                **log_kwargs,
            )


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def _module_of(obj: Any) -> str | None:
    """Return the module name a class was defined in, or ``None``."""
    cls = type(obj)
    try:
        return inspect.getmodule(cls).__name__  # type: ignore[union-attr]
    except Exception:
        return None
