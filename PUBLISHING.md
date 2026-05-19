# Publishing `mlflow-explainable` to PyPI

The repository ships two GitHub Actions workflows:

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | push / PR on `main` | lint + tests across Python 3.9 – 3.12; build sdist + wheel as a CI artifact |
| `publish.yml` | release published, or git tag `v*.*.*` | build sdist + wheel and upload to PyPI via Trusted Publishing |

## Local sanity check before tagging

```bash
cd extension/mlflow_explainable

# 1. Run the test matrix locally (sklearn + XGBoost paths included)
pip install -e ".[dev]" xgboost
pytest --cov=mlflow_explainable

# 2. Build distributions
python -m build       # → dist/mlflow_explainable-X.Y.Z-py3-none-any.whl
                      #   dist/mlflow_explainable-X.Y.Z.tar.gz

# 3. Validate metadata + README rendering
twine check dist/*

# 4. (Optional) install the built wheel into a clean env and import it
pip install dist/mlflow_explainable-X.Y.Z-py3-none-any.whl
python -c "import mlflow_explainable as me; print(me.__version__)"
```

## First-time PyPI setup (Trusted Publishing)

1. Create the project on PyPI: <https://pypi.org/manage/projects/>
2. Add a Trusted Publisher pointing at this GitHub repo + the
   `publish.yml` workflow. No long-lived API token is needed.
   <https://docs.pypi.org/trusted-publishers/>
3. Tag the release:

   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

   The `publish.yml` workflow runs automatically and uploads the artifacts.

## Subsequent releases

1. Bump `version` in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Commit, tag (`git tag vX.Y.Z`), push the tag.

## Release checklist

- [ ] `pyproject.toml` version bumped
- [ ] `src/mlflow_explainable/__init__.py` `__version__` bumped
- [ ] `CHANGELOG.md` entry added
- [ ] `pytest` passes locally
- [ ] `twine check dist/*` passes
- [ ] git tag pushed
- [ ] PyPI release confirmed at <https://pypi.org/project/mlflow-explainable/>
