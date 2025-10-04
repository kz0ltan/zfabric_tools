This is a small collection of tools to extract/save text from/to services.

This is meant to be used by another project.

To run tests, add this to the parent project's pyproject.toml:

```
[tool.pytest.ini_options]
pythonpath = ["tools"]
```

To execute tests (from the parent project's root):

```
uv run pytest tools/tests/test_web_extractor.py
```
