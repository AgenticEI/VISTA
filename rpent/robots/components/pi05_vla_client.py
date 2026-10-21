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

"""Pi0.5 VLA client for LIBERO."""

from __future__ import annotations

import copy
import os
import threading
import uuid
from typing import Any

import numpy as np

from rpent.robots.components.vla_client_base import BaseVLAClient


class Pi05VLAClient(BaseVLAClient):
    """Adapt LIBERO observations to the common VLA RPC protocol."""

    def __init__(self, client, *, policy_seed=None, session_id=None):
        super().__init__(client)
        protocol = os.environ.get("RPENT_VLA_SESSION_PROTOCOL")
        if protocol not in (None, "", "pi05"):
            raise ValueError(f"Pi0.5 requires session protocol 'pi05', got {protocol!r}")
        value = policy_seed if policy_seed is not None else os.environ.get("RPENT_POLICY_RNG")
        self._policy_seed = None if value is None else int(value)
        if self._policy_seed is not None and not 0 <= self._policy_seed < 2**63:
            raise ValueError("RPENT_POLICY_RNG must be in [0, 2**63)")
        if protocol == "pi05" and self._policy_seed is None:
            raise ValueError("Pi0.5 pool sessions require RPENT_POLICY_RNG")
        self._policy_session_id = session_id or os.environ.get("RPENT_VLA_SESSION_ID") or f"pi05-{uuid.uuid4().hex}"
        self._request_id = 0
        self._closed = False
        self._pending_request = None
        self._predict_lock = threading.Lock()

    def predict(self, obs, options=None):
        with self._predict_lock:
            if self._closed:
                raise RuntimeError("Pi0.5 episode session is closed")
            options = dict(options or {})
            if self._policy_seed is not None:
                options.update(
                    session_id=self._policy_session_id,
                    policy_seed=self._policy_seed,
                    request_id=self._request_id,
                )
                current = (obs, options)
                if self._pending_request is not None and not _same_request(self._pending_request, current):
                    raise RuntimeError("Pi0.5 previous request has no response; retry identical input or restart the episode")
                self._pending_request = copy.deepcopy(current)
            # Advance only after a response: a transport retry retains its ID.
            result = super().predict(obs, options)
            self._request_id += 1
            self._pending_request = None
            return result

    def close(self):
        """Release small cached actions and private RNG state after the episode."""
        with self._predict_lock:
            if self._closed:
                return
            if self._policy_seed is not None:
                self._client.call(
                    "vla.release_session", args=(self._policy_session_id,),
                    timeout_s=self._TIMEOUT_S["default"],
                )
            self._closed = True
            self._pending_request = None

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode one LIBERO observation and return a Pi0.5 action chunk."""
        main_image = np.asarray(env_obs["main_images"])
        if main_image.ndim != 3:
            raise ValueError(
                f"main_images expected shape [H,W,3]; got {main_image.shape}"
            )
        if main_image.dtype != np.uint8:
            main_image = main_image.astype(np.uint8)
        observation: dict[str, Any] = {
            "main_images": main_image[None],
            "wrist_images": None,
            "extra_view_images": None,
        }
        for source_key in ("wrist_images", "extra_view_images"):
            view = env_obs.get(source_key)
            if view is None:
                continue
            array = np.asarray(view)
            if array.size > 0 and array.ndim == 3:
                if array.dtype != np.uint8:
                    array = array.astype(np.uint8)
                observation[source_key] = array[None]

        states = np.asarray(env_obs["states"], dtype=np.float32)
        if states.ndim != 1:
            raise ValueError(
                f"states must be single-env shape [state_dim]; got {states.shape}"
            )

        observation["states"] = states[None]
        observation["task_descriptions"] = [str(env_obs.get("task_descriptions") or "")]
        response = self.predict(observation, options={"mode": mode})
        actions = np.asarray(response, dtype=np.float32)[0]
        return actions, {}


def _same_request(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_request(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_same_request(a, b) for a, b in zip(left, right))
    return left == right
