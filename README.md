# mlflow-explainable

Log explainable ML models (predictor + SHAP explainer) to MLflow as a single
self-contained `mlflow.pyfunc` artifact. Designed for a multi-model serving
runtime that should not need to know whether the underlying model is a sklearn
estimator, an XGBoost booster, or a custom torch network.

## Installation

```bash
pip install mlflow-explainable
```

## Quick start

### sklearn-style model

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from mlflow_explainable import log_explainable_model

model = RandomForestClassifier().fit(X_train, y_train)

with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    log_explainable_model(
        model,
        X_train,
        registered_name="sample-rf-crc",
    )
```

### Gradient Boosting (sklearn)

```python
from sklearn.ensemble import GradientBoostingClassifier
from mlflow_explainable import log_explainable_model

model = GradientBoostingClassifier().fit(X_train, y_train)
with mlflow.start_run():
    log_explainable_model(model, X_train, registered_name="sample-gb-crc")
```

### XGBoost

```python
import xgboost as xgb
from mlflow_explainable import log_explainable_model

model = xgb.XGBClassifier(eval_metric="logloss").fit(X_train, y_train)
with mlflow.start_run():
    log_explainable_model(
        model,
        X_train,
        registered_name="sample-xgboost-crc",
        extra_pip_requirements=["xgboost"],
    )
```

Note: as of `shap` 0.49 the meta-`Explainer` no longer auto-detects
`XGBClassifier`. `log_explainable_model` transparently retries with
`model.predict_proba` when this happens, so the user-facing API stays the same.

### Custom torch model (or any callable wrapper)

```python
from mlflow_explainable import log_explainable_model

wrapper = GCNTabularWrapper(gcn, edge_index, device, feature_names)

with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    log_explainable_model(
        wrapper,
        X_train,
        registered_name="sample-gcn-crc",
        explainer_kwargs={"algorithm": "permutation"},
        extra_pip_requirements=["torch", "torch_geometric"],
    )
```

The library walks the predictor's object graph, collects source files of any
user-defined classes (skipping stdlib and well-known third-party prefixes),
and packs them into the artifact via MLflow's `code_path`. The serving runtime
never needs to import those classes from its own codebase.

## Loading at serving time

```python
loaded = mlflow.pyfunc.load_model(model_uri)
impl   = loaded._model_impl.python_model    # ExplainableModel instance

# 1) standard pyfunc predict — returns DataFrame[Y_proba, Y_class]
result = loaded.predict(X)

# 2) SHAP explanation — uniform shape across all explainer types
explanation = impl.shap_explain(X)
# explanation["values"]      shape (n, n_features) or (n, n_features, n_classes)
# explanation["base_values"] shape ()              or (n_classes,) or (n, n_classes)
# explanation["data"]        shape (n, n_features)
```

## Why a contract?

`mlflow.sklearn.load_model` ties the runtime to the sklearn API. Adding a
torch model means another branch, another set of attribute assumptions
(`predict_proba`, `expected_value`, ...), and another way for `kserve` to
break when the SHAP version changes.

`mlflow-explainable` standardises the runtime-facing surface to two methods —
`predict` and `shap_explain` — and pushes all framework-specific glue into the
artifact itself.

## License

MIT
