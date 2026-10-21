# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO env client that forwards calls over an RPC transport.

Lives in :mod:`robots.libero` because the methods exposed
here (``raw_obs`` / ``render_camera`` / ``get_camera_meta`` / …)
reference LIBERO-specific obs dict keys and camera names. The generic
transport layer lives in :mod:`rpent.utils.rpc.socket_rpc`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rpent.robots.components.env_client_base import BaseEnvClient
from rpent.trace import current_trace_context
from rpent.utils.rpc import RpcClient


class LiberoEnvClient(BaseEnvClient):
    """Remote implementation of the LIBERO env protocol."""

    _TIMEOUT_S = {
        **BaseEnvClient._TIMEOUT_S,
        "env.render_camera": 120.0,
    }

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict,
        return_all_frames: bool = False,
    ):
        self.return_all_frames = return_all_frames
        self.terminated = False
        self.truncated = False
        self._status = {}
        self._client = client
        server_meta = client.call(
            "env.get_env_meta", timeout_s=self._TIMEOUT_S["default"]
        )
        if any(server_meta.get(key) != value for key, value in expected_meta.items()):
            raise ValueError(
                f"env_meta mismatch: expected={expected_meta!r} actual={server_meta!r}"
            )
        self.server_meta = dict(server_meta)
        self.reset()

    def check_done(self, term, trunc) -> None:
        self.terminated |= bool(np.asarray(term).any())
        self.truncated |= bool(np.asarray(trunc).any())

    def reset(self) -> tuple[dict, Any]:
        ret = super().reset()
        self.last_obs = ret[0]
        self.terminated = False
        self.truncated = False
        self._status = dict(ret[1].get("rpent_status", {}))
        self.check_done(
            self._status.get("terminated", False), self._status.get("truncated", False)
        )
        return ret

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        assert not (self.terminated or self.truncated), (
            "env.step called after the episode signaled term/trunc"
        )
        ret = self._client.call(
            "env.step",
            args=(action,),
            kwargs={"trace_context": current_trace_context()},
            timeout_s=self._TIMEOUT_S["env.step"],
        )
        self.last_obs = ret[0]
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        self._status = dict(ret[4].get("rpent_status", {}))
        return ret

    def chunk_step(
        self, actions, *, return_all_frames: bool | None = None
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Run an action chunk in one RPC. Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``.

        ``obs`` is ``list[Obs]`` when ``return_all_frames`` is True
        (one entry per chunk step), otherwise the final ``Obs`` dict.
        Terminated / truncated have shape ``[executed_steps]``. A success or
        horizon boundary may shorten the requested chunk.
        """
        assert not (self.terminated or self.truncated), (
            "env.chunk_step called after the episode signaled term/trunc"
        )
        if return_all_frames is None:
            return_all_frames = self.return_all_frames
        ret = self._client.call(
            "env.chunk_step",
            args=(actions,),
            kwargs={
                "return_all_frames": return_all_frames,
                "trace_context": current_trace_context(),
            },
            timeout_s=self._TIMEOUT_S["env.chunk_step"],
        )
        self.last_obs = ret[0][-1] if isinstance(ret[0], list) else ret[0]
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        self._status = dict(ret[4][-1].get("rpent_status", {}))
        return ret

    def enable_full_trace(self, output_dir, *, mode: str = "full_rgb"):
        """Attach the server-owned writer before the next episode reset."""
        return self._client.call(
            "env.enable_full_trace",
            args=(str(output_dir),),
            kwargs={"mode": mode},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def finish_trajectory(self):
        return self._client.call(
            "env.finish_trajectory", timeout_s=self._TIMEOUT_S["env.chunk_step"]
        )

    def get_status(self) -> dict:
        self._status = self._client.call(
            "env.get_status", timeout_s=self._TIMEOUT_S["default"]
        )
        self.terminated = bool(self._status["terminated"])
        self.truncated = bool(self._status["truncated"])
        return dict(self._status)

    @property
    def env_steps(self) -> int:
        return int(self._status.get("episode_steps", 0))

    @property
    def official_success(self) -> bool:
        return bool(self._status.get("official_success", self.terminated))

    def public_robot_state(self) -> dict:
        return self._client.call(
            "env.public_robot_state", timeout_s=self._TIMEOUT_S["default"]
        )

    def close(self):
        try:
            self.finish_trajectory()
        finally:
            self._client.call("env.close", timeout_s=self._TIMEOUT_S["default"])

    def raw_obs(self) -> dict:
        return self._client.call("env.raw_obs", timeout_s=self._TIMEOUT_S["default"])

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return self._client.call(
            "env.render_camera",
            kwargs={
                "camera_name": camera_name,
                "height": height,
                "width": width,
                "depth": depth,
            },
            timeout_s=self._TIMEOUT_S["env.render_camera"],
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return self._client.call(
            "env.get_camera_meta",
            kwargs={"camera_name": camera_name, "height": height, "width": width},
            timeout_s=self._TIMEOUT_S["default"],
        )

    def get_task_language(self) -> str | None:
        return self._client.call(
            "env.get_task_language", timeout_s=self._TIMEOUT_S["default"]
        )
