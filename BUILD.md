# Build this package and upload it to PyPI

```shell
# Build
rm -r dist/
python -m pip install -r requirements.txt
python -m pip install build
python -m build

# Upload
python -m pip install --upgrade twine

# Test PyPI
# Use username __token__, and copy your PyPI token as the password
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps torch_waymo

# Real PyPI
# Use username __token__, and copy your PyPI token as the password
python -m twine upload dist/*
```
