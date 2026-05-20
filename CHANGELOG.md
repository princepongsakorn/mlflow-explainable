# Changelog

## 1.0.0

Major redesign. Successor to ``mlflow-shap``.

- **NEW** ``log_explainable_model(model, background, *, registered_name=...)`` —
  logs a predictor + SHAP explainer as a single self-contained ``mlflow.pyfunc``
  artifact.
- **NEW** ``ExplainableModel`` base class defining a two-method runtime contract
  (``predict`` + ``shap_explain``) decoupled from any framework.
- **NEW** Auto-detection of user-defined class source files — packed into the
  artifact via ``code_path`` so the serving runtime never needs to import the
  user's classes.
- **NEW** Pickle roundtrip self-test runs before logging — fails loud if a
  class definition is unreachable.
- **REMOVED** ``log_explainer`` / ``load_explainer`` / ``ExplainerCreationError``
  from the previous ``mlflow-shap`` API. The explainer is now bundled inside
  the pyfunc artifact and accessed via ``impl.shap_explain(...)``.

## 0.1.x (mlflow-shap)

Previous package, not maintained. See the ``mlflow-shap`` repository for
historical notes.
