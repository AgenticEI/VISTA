"""Keep LIBERO's online geometry available without archiving it.

The planner's online back-projection and SAM tools may revisit an observation
from the same episode.  The dense depth/world arrays therefore live for the
whole episode, but they must never be written into its persistent rollout.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import imageio.v2 as imageio

from rpent.tools.state import EnvState, StepRecord, _json_default


GEOMETRY_NAMES = frozenset(
    f"{camera}_{kind}.npz"
    for camera in ("agentview", "wrist")
    for kind in ("depth", "world", "world_high")
)
_DEPTH_PREVIEW = re.compile(r"libero_rgbd_[0-9]+_(?:agentview|wrist)_depth\.png\Z")


class LiberoGeometryState(EnvState):
    """The usual EnvState trace, with dense geometry held in process memory."""

    GEOMETRY_NAMES = GEOMETRY_NAMES

    def _ephemeral(self, name: str) -> bool:
        return name in self.GEOMETRY_NAMES or bool(_DEPTH_PREVIEW.fullmatch(name))

    def reset(self) -> None:
        self._geometry: dict[tuple[str, int], bytes] = {}
        super().reset()

    def save(self, name: str, value: Any, *, step: int | None = -1, **options: Any) -> str | None:
        if not self._ephemeral(name):
            return super().save(name, value, step=step, **options)
        self._validate_name(name)
        index = self._resolve_read_step(step)
        if index is None:
            raise ValueError("online geometry requires an observation step")
        record = self._record_for(index)
        stream = io.BytesIO()
        if name.endswith(".png"):
            imageio.imwrite(stream, np.asarray(value, dtype=np.uint8), format="png")
        elif name.endswith(".npy"):
            np.save(stream, np.asarray(value))
        else:
            np.savez_compressed(stream, array=np.asarray(value))
        self._geometry[(name, index)] = stream.getvalue()
        record.artifacts.add(name)
        if not self._step_open:
            self._write_manifest()
        return name

    def load(self, name: str, *, step: int | None = -1) -> Any:
        if not self._ephemeral(name):
            return super().load(name, step=step)
        index = self._resolve_read_step(step)
        if index is None:
            raise ValueError("online geometry requires an observation step")
        raw = self._geometry[(name, index)]
        if name.endswith(".png"):
            return imageio.imread(io.BytesIO(raw))
        if name.endswith(".npy"):
            return np.load(io.BytesIO(raw))
        with np.load(io.BytesIO(raw)) as archive:
            return archive["array"]

    def load_bytes(self, name: str, *, step: int | None = -1) -> bytes:
        if not self._ephemeral(name):
            return super().load_bytes(name, step=step)
        index = self._resolve_read_step(step)
        if index is None:
            raise ValueError("online geometry requires an observation step")
        return self._geometry[(name, index)]

    def exists(self, name: str, *, step: int | None = -1) -> bool:
        if not self._ephemeral(name):
            return super().exists(name, step=step)
        try:
            index = self._resolve_read_step(step)
        except LookupError:
            return False
        return index is not None and (name, index) in self._geometry

    def artifact_path(self, name: str, *, step: int | None = -1) -> Path:
        if self._ephemeral(name):
            raise FileNotFoundError("online geometry has no persistent artifact path")
        return super().artifact_path(name, step=step)

    def release_geometry(self) -> None:
        self._geometry.clear()
        for record in self._steps:
            record.artifacts = {name for name in record.artifacts if not self._ephemeral(name)}
        self._write_manifest()

    def _write_manifest(self) -> None:
        """Report only files that were actually archived on disk."""
        destination = self._manifest_file()
        temporary = self._temporary_file(destination)
        manifest = {
            "run_artifacts": sorted(self._run_artifacts),
            "steps": [self._persisted_step(record) for record in self._steps],
            "online_geometry_archived": False,
        }
        try:
            with temporary.open("w") as stream:
                json.dump(manifest, stream, indent=2, default=_json_default)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _persisted_step(self, record: StepRecord) -> dict[str, Any]:
        blob = record.to_blob()
        blob["artifacts"] = sorted(name for name in record.artifacts if not self._ephemeral(name))
        return blob
