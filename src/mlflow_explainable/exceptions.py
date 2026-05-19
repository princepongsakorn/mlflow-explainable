"""Custom exceptions raised by ``mlflow_explainable``."""

from __future__ import annotations

__all__ = [
    "MLflowExplainableError",
    "NoActiveRunError",
    "ExplainableModelError",
]


class MLflowExplainableError(Exception):
    """Base class for all errors raised by ``mlflow_explainable``."""


class NoActiveRunError(MLflowExplainableError):
    """Raised when ``log_explainable_model`` is called without an active MLflow run
    and no explicit ``run_id`` was provided.
    """


class ExplainableModelError(MLflowExplainableError):
    """Raised when an explainable model cannot be constructed, pickled, or logged.

    Common causes:
        * ``shap.Explainer(model, background)`` fails to construct.
        * The pickle roundtrip self-test fails — usually because a class
          definition is not importable from a file (e.g. defined in an
          interactive session, or inside a function body). Pass
          ``extra_code_paths=[...]`` to include those source files.
    """
