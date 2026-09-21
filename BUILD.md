# Build and publish

Build and validate the distributions locally:

```shell
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install ".[dev]"
python -m build
python -m twine check dist/*
```

Publishing is automated by `.github/workflows/publish.yml`. Create a GitHub release after updating the version in `pyproject.toml`; the workflow builds the release artifacts and publishes them to PyPI with Trusted Publishing.

The `torch_waymo` project on PyPI must have a Trusted Publisher configured with:

- owner: `willGuimont`
- repository: `torch_waymo`
- workflow: `publish.yml`
- environment: `pypi`

The GitHub `pypi` environment can optionally require approval before deployment. No API token secret is needed.
