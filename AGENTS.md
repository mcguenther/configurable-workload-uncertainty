# Repository Guidelines

This repository is written mostly in Python. Tests are located under `wluncert/tests` and datasets under `wluncert/dal/data` and `wluncert/training-data`.

## Coding Style

- Follow standard PEP 8 style.
- **Always format Python code with `black`** (version pinned in `requirements.txt`). Run `black .` from the repository root before committing.
- Add type hints and docstrings for new functions where practical.
- Avoid committing large generated data or binary files.

## Testing

- Install dependencies using `pip install -r requirements.txt` if needed.
- Run tests with `pytest` from the repository root.

## Example workflow

```bash
pip install -r requirements.txt
black .
pytest
```
