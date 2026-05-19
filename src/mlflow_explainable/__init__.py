"""``mlflow-explainable`` — log explainable ML models to MLflow with a single call.

A logged model is a self-contained ``mlflow.pyfunc`` artifact that bundles the
predictor *and* a SHAP explainer. The serving runtime can then load any model
through a uniform contract (``predict`` + ``shap_explain``) regardless of the
underlying framework (sklearn, xgboost, torch, ...).

Public API
----------
- :class:`ExplainableModel` — base contract for python_models served by the runtime.
- :func:`log_explainable_model` — one-liner to log a predictor + explainer pair.

Exceptions
----------
- :class:`MLflowExplainableError` — base class for all errors raised here.
- :class:`NoActiveRunError` — no active run and no ``run_id`` provided.
- :class:`ExplainableModelError` — model could not be constructed, pickled, or logged.

Example
-------
::

    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from mlflow_explainable import log_explainable_model

    model = RandomForestClassifier().fit(X_train, y_train)
    with mlflow.start_run():
        log_explainable_model(model, X_train, registered_name="sample-rf-crc")
"""

from __future__ import annotations

from .contract import ExplainableModel, log_explainable_model
from .exceptions import (
    ExplainableModelError,
    MLflowExplainableError,
    NoActiveRunError,
)

__version__ = "1.0.0"

__all__ = [
    "ExplainableModel",
    "log_explainable_model",
    "MLflowExplainableError",
    "NoActiveRunError",
    "ExplainableModelError",
    "__version__",
]
