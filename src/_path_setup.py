"""
_path_setup.py
==============
Ensures `src/` and `configs/` are on sys.path so the project's modules can
import each other by short names (e.g. ``from spherical_kv_pipeline import ...``,
``from config import MODEL_NAME``) regardless of where Python was invoked from.

Entry-point scripts in ``scripts/`` should import this module first:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import _path_setup  # noqa: F401  -- side-effect import

Modules inside ``src/`` itself can also do ``import _path_setup`` to ensure the
configs/ directory is on the path. It's idempotent — calling it many times is a
no-op after the first.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))           # .../src
_ROOT = os.path.dirname(_HERE)                                # .../
_CONFIGS = os.path.join(_ROOT, "configs")

for path in (_HERE, _CONFIGS):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
