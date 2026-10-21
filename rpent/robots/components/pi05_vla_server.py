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

"""RPC server wrapping the Pi0.5 VLA."""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.config import (
    get_pi05_checkpoint_path,
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger

logger = get_logger("vla_server")


@dataclass
class _PolicySession:
    seed: int
    rng: dict[str, Any]
    next_request: int = 0
    results: dict[int, np.ndarray] = field(default_factory=dict)
    touched: float = field(default_factory=time.monotonic)


class Pi05PolicySessions:
    """Serialize inference while swapping private Python/NumPy/Torch RNG states.

    OpenPI draws from Torch's default generators at several denoising steps;
    seeding just the initial noise does not isolate its complete sampling path.
    A failed call does not advance the episode stream. Successful request IDs
    retain their original small action result until the episode is released.
    """

    def __init__(self, cuda_devices=(), *, timeout_s=3600, max_sessions=256):
        self.cuda_devices = tuple(cuda_devices)
        self.timeout_s = timeout_s
        self.max_sessions = max_sessions
        self.sessions: dict[str, _PolicySession] = {}
        self.lock = threading.RLock()

    def _capture(self):
        import torch

        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": {d: torch.cuda.get_rng_state(d) for d in self.cuda_devices},
        }

    @staticmethod
    def _restore(state):
        import torch

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        for device, value in state["cuda"].items():
            torch.cuda.set_rng_state(value, device)

    def _initial_state(self, seed):
        import torch

        return {
            "python": random.Random(seed).getstate(),
            "numpy": np.random.RandomState(seed % (2**32)).get_state(),
            "torch": torch.Generator(device="cpu").manual_seed(seed).get_state(),
            "cuda": {
                d: torch.Generator(device=f"cuda:{d}").manual_seed(seed).get_state()
                for d in self.cuda_devices
            },
        }

    def _expire(self):
        cutoff = time.monotonic() - self.timeout_s
        for sid in list(self.sessions):
            if self.sessions[sid].touched < cutoff:
                del self.sessions[sid]

    @contextmanager
    def _rng(self, session):
        outer = self._capture()
        self._restore(session.rng)
        try:
            yield
            session.rng = self._capture()
        finally:
            self._restore(outer)

    def run(self, session_id, policy_seed, request_id, predict):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("Pi0.5 session_id must be a nonempty string of at most 256 characters")
        if type(policy_seed) is not int or not 0 <= policy_seed < 2**63:
            raise ValueError("Pi0.5 policy_seed must be an integer in [0, 2**63)")
        if type(request_id) is not int or request_id < 0:
            raise ValueError("Pi0.5 request_id must be a nonnegative integer")
        with self.lock:
            self._expire()
            session = self.sessions.get(session_id)
            if session is None:
                if request_id != 0:
                    raise ValueError("Pi0.5 session missing or expired; restart the episode")
                if len(self.sessions) >= self.max_sessions:
                    raise RuntimeError("Pi0.5 session capacity reached; release completed episodes")
                session = _PolicySession(policy_seed, self._initial_state(policy_seed))
                self.sessions[session_id] = session
            if session.seed != policy_seed:
                raise ValueError("Pi0.5 policy_seed cannot change within an episode")
            session.touched = time.monotonic()
            if request_id in session.results:
                return session.results[request_id].copy()
            if request_id != session.next_request:
                raise ValueError("Pi0.5 requests must be consecutive within an episode")
            if request_id >= 10000:
                raise RuntimeError("Pi0.5 episode exceeded 10000 inference requests")
            with self._rng(session):
                result = np.asarray(predict(), dtype=np.float32).copy()
            session.results[request_id] = result
            session.next_request += 1
            session.touched = time.monotonic()
            return result.copy()

    def release(self, session_id):
        with self.lock:
            return {"released": self.sessions.pop(session_id, None) is not None}

RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_model_cfg(model_path: str) -> Any:
    """OmegaConf for ``rlinf.models.embodiment.openpi.get_model``."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 5,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": 5,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": None,
                "use_dsrl": False,
            },
        }
    )


# ---------------------------------------------------------------------------
# Facade implementing the Pi0.5 client protocol
# ---------------------------------------------------------------------------


class Pi05VLAFacade(BaseVLAFacade):
    """Implements :class:`rpent.robots.components.pi05_vla_client.Pi05VLAClient` over a Pi0.5 model.

    Loads the model once at construction; each ``predict`` call runs one
    inference batch and returns its NumPy action array.
    """

    def __init__(self, model_path: str):
        import torch
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        cfg = build_model_cfg(model_path=model_path)
        t0 = time.time()
        logger.info("loading Pi0.5 (model_path=%s) ...", cfg["model_path"])
        self._model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        self._inference_context = torch.no_grad
        self._policy_sessions = Pi05PolicySessions(
            {p.device.index for p in self._model.parameters() if p.device.type == "cuda"}
        )
        logger.info("model ready in %.1fs", time.time() - t0)
        super().__init__()

    def _register_rpc(self):
        super()._register_rpc()
        self._rpc["vla.release_session"] = self._policy_sessions.release
        self._rpc["vla.protocol"] = lambda: {"protocol": "pi05", "version": 1}

    def predict(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if not isinstance(observation, dict):
            raise TypeError(f"Pi0.5 observation must be a mapping, got {observation!r}")
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise TypeError(f"Pi0.5 options must be a mapping, got {options!r}")
        unexpected_options = set(options) - {"mode", "session_id", "policy_seed", "request_id"}
        if unexpected_options:
            raise ValueError(
                f"unsupported Pi0.5 options: {sorted(unexpected_options)!r}"
            )

        env_obs = dict(observation)
        main_images = np.asarray(env_obs.get("main_images"))
        if main_images.ndim != 4 or main_images.shape[-1] != 3:
            raise ValueError(
                f"main_images must be [B,H,W,3]; got shape {main_images.shape}"
            )
        env_obs["main_images"] = main_images.astype(np.uint8, copy=False)

        batch_size = main_images.shape[0]
        states = np.asarray(env_obs.get("states"), dtype=np.float32)
        if states.ndim != 2 or states.shape[0] != batch_size:
            raise ValueError(
                "states must be [B,state_dim] with the image batch size; "
                f"got shape {states.shape}"
            )
        env_obs["states"] = states

        task_descriptions = env_obs.get("task_descriptions")
        if (
            not isinstance(task_descriptions, list)
            or len(task_descriptions) != batch_size
        ):
            raise ValueError("task_descriptions must contain one string per batch item")
        env_obs["task_descriptions"] = [str(value) for value in task_descriptions]

        for key in ("wrist_images", "extra_view_images"):
            view = env_obs.get(key)
            if view is None:
                env_obs[key] = None
                continue
            array = np.asarray(view)
            if array.ndim != 4 or array.shape[0] != batch_size or array.shape[-1] != 3:
                raise ValueError(f"{key} must be [B,H,W,3]; got shape {array.shape}")
            env_obs[key] = array.astype(np.uint8, copy=False)

        def infer():
            with self._inference_context():
                actions, _ = self._model.predict_action_batch(
                    env_obs, mode=options.get("mode", "eval")
                )
            return (
                actions.detach().cpu().numpy()
                if hasattr(actions, "detach")
                else np.asarray(actions)
            ).astype(np.float32)

        session_fields = {"session_id", "policy_seed", "request_id"}
        provided = session_fields.intersection(options)
        if provided:
            if provided != session_fields:
                raise ValueError("Pi0.5 isolated inference requires session_id, policy_seed and request_id")
            if batch_size != 1:
                raise ValueError("Pi0.5 episode sessions require a single observation")
            return self._policy_sessions.run(
                options["session_id"], options["policy_seed"], options["request_id"], infer
            )
        # Ordinary RPent clients retain the pre-session API.
        with self._policy_sessions.lock:
            return infer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Pi0.5 checkpoint (defaults to PI05_CHECKPOINT_PATH env)",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    model_path = args.model_path or get_pi05_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "PI05_CHECKPOINT_PATH is not set; provide the Pi0.5 checkpoint "
            "path via --model-path or the environment."
        )

    facade = Pi05VLAFacade(model_path=model_path)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
