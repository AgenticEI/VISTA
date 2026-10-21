"""Resolve this run's RPent source before an unrelated editable installation.

Enabled only by the launcher's explicit RPENT_REPO_ROOT and PYTHONPATH. Other
packages retain the interpreter's normal import order; shared venvs are untouched.
"""

import importlib.machinery
import os
import sys


_ROOT = os.environ.get("RPENT_REPO_ROOT")


class _RPentSourceFinder:
    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname.split(".", 1)[0] not in {"rpent", "meta_harness", "robots"}:
            return None
        return importlib.machinery.PathFinder.find_spec(fullname, path or [_ROOT], target)


if _ROOT and os.path.isdir(os.path.join(_ROOT, "rpent")):
    sys.meta_path.insert(0, _RPentSourceFinder)
