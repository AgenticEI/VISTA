"""Recognize explicitly recorded thin SSD run aliases without broad path bypasses."""
import json
from pathlib import Path


def migration_receipt(root, repository=None):
    root = Path(root)
    path = root.resolve() / "ssd-migration.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    original = Path(value.get("original_root", "/"))
    if (value.get("kind") != "thin-ssd-run-20261013"
            or Path(value.get("ssd_root", "/")).resolve() != root.resolve()
            or not original.is_symlink() or original.resolve() != root.resolve()):
        return None
    if repository is not None and original.parent.resolve() != (Path(repository) / "runs").resolve():
        return None
    return value


def migrated_result_dir(requested, repository):
    requested = Path(requested)
    for root in requested.parents:
        receipt = migration_receipt(root, repository)
        if receipt is None:
            continue
        relative = requested.relative_to(root)
        if len(relative.parts) < 3 or relative.parts[0] != "episodes":
            raise ValueError("SSD result directory must name an episode and rollout")
        if relative.parts[1] in receipt.get("history_episode_ids", ()):
            raise ValueError("historical episode storage is read-only")
        cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError("--result-dir must not pass through an unapproved symlink")
        if not requested.resolve().is_relative_to(root.resolve()):
            raise ValueError("--result-dir resolves outside its approved SSD run")
        return requested.resolve()
    return None
