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

"""RPC server wrapping a single-env LIBERO environment."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.trace import RGBDTrajectoryWriter, TraceError, public_robot_state
from rpent.utils.config import (
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
if "mujoco" in sys.modules and not all(
    os.environ.get(name) for name in ("MUJOCO_GL", "PYOPENGL_PLATFORM")
):
    raise RuntimeError("configure MUJOCO_GL/PYOPENGL_PLATFORM before importing mujoco")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# A spawned RLinf worker imports this module to reconstruct _SensorEnv after
# loading its configured simulator. That is safe when its launch environment
# already selected the renderer; it must not be rejected as a late setup.

logger = get_logger("env_server")

from robots.libero import robosuite_compat

robosuite_compat.apply()

RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# torch and LiberoEnv are only imported at call time (after --cuda-device
# sets CUDA_VISIBLE_DEVICES in main()); LiberoEnv transitively imports torch.
if TYPE_CHECKING:
    import torch  # noqa: F401  (referenced at runtime in _to_numpy_tree)
    from rlinf.envs.libero.libero_env import LiberoEnv


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


class _SensorEnv:
    """Add read-only robot metadata inside the existing simulator worker.

    RLinf's worker already supports ``env_call(target='self')``. This wrapper
    exposes only the real robot base transform; no object or task predicate is
    included, and the existing environment still owns reset and physics.
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def rpent_robot_metadata(self):
        rob = self.env
        while hasattr(rob, "env"):
            rob = rob.env
        robot = rob.robots[0]
        body = robot.robot_model.root_body
        return {
            "base_position": np.asarray(rob.sim.data.get_body_xpos(body)).tolist(),
            # MuJoCo stores wxyz; the public trace contract uses xyzw.
            "base_quaternion": np.asarray(rob.sim.data.get_body_xquat(body))[
                [1, 2, 3, 0]
            ].tolist(),
            "base_body": body,
            "backend_module": type(self.env).__module__,
            "control_frequency_hz": float(rob.control_freq) if hasattr(rob, "control_freq") else None,
        }


def _sensor_env_factory(factory):
    def make():
        return _SensorEnv(factory())

    return make


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 10000,
) -> Any:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": task_suite_name,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": specific_reset_id,
            "video_cfg": {
                "save_video": True,
                "info_on_video": True,
                "video_base_dir": "/tmp/primitive_videos",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
                # Render depth too, so we can back-project pixels to world
                # from depth + camera calibration
                "camera_depths": True,
                "horizon": max_episode_steps,
                # RLinf owns the policy horizon after its 15 settling steps.
                # The robosuite counter includes those initialization actions;
                # its independent horizon must not terminate 15 steps early.
                "ignore_done": True,
                **(
                    {"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                    if os.environ.get("LIBERO_ROBOT_BASE")
                    else {}
                ),
            },
        }
    )
    return cfg


def make_env(
    task_id: int,
    seed: int,
    suite_name: str = "libero_spatial",
    max_episode_steps: int = 10000,
) -> LiberoEnv:
    """Build a single-env LiberoEnv pinned to ``task_id`` / ``seed``."""
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import benchmark as _bench_mod

    suite = _bench_mod.get_benchmark(suite_name)()
    if not 0 <= task_id < suite.get_num_tasks():
        raise ValueError(f"task index {task_id} is outside suite {suite_name!r}")
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    if trials <= 0:
        raise ValueError(f"suite {suite_name!r} task {task_id} has no initial states")
    rid = first_id + (seed % trials)
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )

    class SensorLiberoEnv(LiberoEnv):
        def get_env_fns(self):
            return [_sensor_env_factory(factory) for factory in super().get_env_fns()]

    return SensorLiberoEnv(
        cfg=cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
    )


# ---------------------------------------------------------------------------
# Facade implementing the robots.libero.env_client protocol
# ---------------------------------------------------------------------------


def _to_numpy_tree(x):
    """Recursively convert torch tensors to CPU numpy arrays so the result
    pickles cleanly across the agent/env_server wire."""
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_tree(v) for v in x)
    return x


class LiberoEnvFacade(BaseEnvFacade):
    """Implements :class:`robots.libero.env_client.LiberoEnvClient`
    over :class:`rlinf.envs.libero.libero_env.LiberoEnv`.

    All return values are converted to CPU numpy so the agent process
    (which does not import torch) can consume them after the pickle round
    trip.
    """

    def __init__(self, env: LiberoEnv, *, meta: dict):
        self._env = env
        self._env_idx = 0
        self._closed = False
        self._started = False
        self._task_start_perf = None
        self._terminal_perf = None
        self._episode_steps = 0
        self._terminated = False
        self._truncated = False
        self._robot_metadata: dict[str, Any] | None = None
        self._trajectory: RGBDTrajectoryWriter | None = None
        # Identifies what task/seed this server was launched with — the
        # client compares against its own expected values at construction
        # and refuses to talk to a stale or mis-configured server.
        self._meta = dict(meta)
        self._meta.update(
            protocol_version=2,
            reset_settling_steps=15,
            step_zero="after_reset_settling",
            step_unit="environment_control_step",
            chunk_termination="stop_at_first_terminal_step",
        )
        super().__init__()

    def _register_rpc(self) -> None:
        super()._register_rpc()
        self._rpc.update(
            {
                "env.raw_obs": self.raw_obs,
                "env.render_camera": self.render_camera,
                "env.get_camera_meta": self.get_camera_meta,
                "env.get_task_language": self.get_task_language,
                "env.get_status": self.get_status,
                "env.public_robot_state": self.get_public_robot_state,
                "env.enable_full_trace": self.enable_full_trace,
                "env.finish_trajectory": self.finish_trajectory,
            }
        )
        self._readonly_methods.update(
            {"env.get_task_language", "env.get_status", "env.public_robot_state"}
        )

    # ---- shape helpers ----

    def _strip(self, v):
        """Drop the leading env dim. ``v`` is either a batched numpy array
        (shape ``[B, ...]``), a length-B list (e.g. ``task_descriptions``),
        or ``None`` (optional images). LiberoEnv runs ``num_envs=1`` so
        index ``self._env_idx`` is always present."""
        if v is None:
            return None
        return v[self._env_idx]

    def _strip_obs(self, obs: dict) -> dict:
        """Strip the leading env dim from every value of a LIBERO obs dict."""
        return {k: self._strip(v) for k, v in obs.items()}

    def _expand_action(self, action) -> np.ndarray:
        """Inject the env dim onto a single-env action shaped ``[action_dim]``."""
        return np.asarray(action)[None]

    def _expand_chunk(self, actions) -> np.ndarray:
        """Inject the env dim onto a single-env chunk shaped
        ``[chunk_size, action_dim]``."""
        return np.asarray(actions)[None]

    # ---- gym-like surface ----

    def reset(self):
        if self._trajectory is not None and self._trajectory.last_env_step is not None:
            raise TraceError(
                "a traced episode cannot reset; use a fresh environment and output directory"
            )
        obs, info = self._env.reset()
        self._episode_steps = 0
        self._terminated = False
        self._truncated = False
        self._started = True
        self._task_start_perf = time.perf_counter()
        self._terminal_perf = None
        self._robot_metadata = None
        # RLinf reset performs 15 settling actions then resets elapsed_steps.
        # Control-step 0 is the resulting initialized observation. Settling is
        # outside the policy horizon and is explicitly described in trace meta.
        self._check_step_count()
        self._terminated = bool(self._env.env.workers[self._env_idx].check_success())
        if self._terminated:
            self._terminal_perf = self._task_start_perf
        obs = self._strip_obs(_to_numpy_tree(obs))
        if self._trajectory is not None:
            self._record_trajectory(action=None, trace_context=None)
        return obs, {**_to_numpy_tree(info), "rpent_status": self.get_status()}

    def _check_step_count(self):
        actual = int(np.asarray(self._env.elapsed_steps).reshape(-1)[self._env_idx])
        if actual != self._episode_steps:
            raise RuntimeError(
                f"LIBERO control-step count mismatch: expected {self._episode_steps}, actual {actual}"
            )

    def step(self, action, *, trace_context: dict | None = None):
        if not self._started:
            raise RuntimeError("env.step requires an initial reset")
        if self._terminated or self._truncated:
            raise RuntimeError("env.step called after the episode ended")
        action = np.asarray(action)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("LIBERO action must be a finite vector of length 7")
        if self._trajectory is not None:
            self._trajectory.preflight()
        before = self._episode_steps
        try:
            obs, rew, term, trunc, info = self._env.step(
                self._expand_action(action), auto_reset=False
            )
            self._episode_steps += 1
            self._check_step_count()
            self._terminated = bool(np.asarray(_to_numpy_tree(term)).any())
            self._truncated = bool(np.asarray(_to_numpy_tree(trunc)).any()) or (
                self._episode_steps >= self._meta["max_episode_steps"]
            )
            if self._terminated or self._truncated:
                self._terminal_perf = time.perf_counter()
            obs = self._strip_obs(_to_numpy_tree(obs))
            if self._trajectory is not None:
                self._record_trajectory(
                    action={
                        "env_steps_before": before,
                        "env_steps_after": self._episode_steps,
                        "env_action": action,
                    },
                    trace_context=trace_context,
                )
            return (
                obs,
                self._strip(_to_numpy_tree(rew)),
                self._terminated,
                self._truncated,
                {**_to_numpy_tree(info), "rpent_status": self.get_status()},
            )
        except Exception as exc:
            if self._trajectory is not None:
                self._trajectory.fail(exc)
            raise

    def chunk_step(
        self,
        actions,
        *,
        return_all_frames: bool = False,
        trace_context: dict | None = None,
    ):
        """Run a full action chunk in one RPC. ``actions`` shape
        ``[chunk_size, action_dim]`` (single env).

        Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``. ``obs`` is
        ``list[Obs]`` when ``return_all_frames=True`` (full per-step
        trajectory), or just the final ``Obs`` dict when False (default).
        ``terminated`` / ``truncated`` have one entry per executed action.
        The first terminal control step ends the chunk immediately; no action
        after a success or horizon boundary is executed or archived.
        """
        actions = np.asarray(actions)
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or not len(actions)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("LIBERO chunk must be a nonempty finite [steps,7] array")
        observations, rewards, terminations, truncations, infos = [], [], [], [], []
        for action in actions:
            obs, reward, term, trunc, info = self.step(
                action, trace_context=trace_context
            )
            observations.append(obs)
            rewards.append(reward)
            terminations.append(term)
            truncations.append(trunc)
            infos.append(info)
            if term or trunc:
                break
        return (
            observations if return_all_frames else observations[-1],
            np.asarray(rewards),
            np.asarray(terminations, dtype=bool),
            np.asarray(truncations, dtype=bool),
            infos,
        )

    def get_status(self) -> dict:
        return {
            "official_success": self._terminated,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "done": self._terminated or self._truncated,
            "episode_steps": self._episode_steps,
            "max_episode_steps": self._meta["max_episode_steps"],
            "terminal_reason": (
                "official_success"
                if self._terminated
                else "horizon"
                if self._truncated
                else "agent_stopped"
            ),
            "initialized": self._started,
            "reset_settling_steps": 15,
            "step_zero": "after_reset_settling",
        }

    def get_public_robot_state(self) -> dict:
        if not self._started:
            raise RuntimeError("robot state is unavailable before reset")
        if self._robot_metadata is None:
            self._robot_metadata = self._env.env.workers[self._env_idx].env_call(
                "rpent_robot_metadata", target="self"
            )
        raw = self.raw_obs()
        state = {
            "eef_position": raw["robot0_eef_pos"],
            "eef_quaternion": raw["robot0_eef_quat"],
            "gripper_qpos": raw["robot0_gripper_qpos"],
            "base_position": self._robot_metadata["base_position"],
            "base_quaternion": self._robot_metadata["base_quaternion"],
        }
        for name, raw_name in {
            "joint_position": "robot0_joint_pos",
            "joint_velocity": "robot0_joint_vel",
            "joint_position_cos": "robot0_joint_pos_cos",
            "joint_position_sin": "robot0_joint_pos_sin",
            "gripper_qvel": "robot0_gripper_qvel",
            "eef_linear_velocity": "robot0_eef_vel_lin",
            "eef_angular_velocity": "robot0_eef_vel_ang",
        }.items():
            if raw_name in raw:
                state[name] = raw[raw_name]
        return public_robot_state(state)

    def enable_full_trace(self, output_dir: str, *, mode: str = "full_rgb"):
        if self._trajectory is not None or self._episode_steps:
            raise TraceError(
                "attach one trace before the episode's initial control step"
            )
        self._trajectory = RGBDTrajectoryWriter(
            Path(output_dir),
            cameras=("agentview", "wrist"),
            mode=mode,
            initialization={
                "settling_control_steps": 15,
                "settling_in_policy_horizon": False,
                "settling_recorded": False,
                "env_step_zero": "after_reset_settling",
                "step_unit": "environment_control_step",
            },
        )

    def finish_trajectory(self):
        if self._trajectory is not None:
            from rpent.trace import _atomic_bytes, _json_bytes
            _atomic_bytes(self._trajectory.root / "public/deployment-clock.json", _json_bytes({
                "control_frequency_hz": (self._robot_metadata or {}).get("control_frequency_hz"),
                "env_steps": self._episode_steps,
                "start_perf_s": self._task_start_perf,
                "end_perf_s": self._terminal_perf if self._terminal_perf is not None else time.perf_counter(),
                "reset_settling_included": False,
            }))
            return self._trajectory.finish(self._episode_steps)
        return None

    def _record_trajectory(self, *, action, trace_context):
        try:
            raw = self.raw_obs()
            captures = {}
            for camera, native in (
                ("agentview", "agentview"),
                ("wrist", "robot0_eye_in_hand"),
            ):
                rgb = np.asarray(raw[f"{native}_image"])[::-1]
                capture = {"rgb": rgb}
                if self._trajectory.mode == "full_rgbd":
                    metadata = dict(self.get_camera_meta(native, *rgb.shape[:2]))
                    metadata["intrinsic"] = metadata.get(
                        "intrinsic", metadata.get("intrinsic_K")
                    )
                    depth = np.asarray(raw[f"{native}_depth"], dtype=np.float32)
                    if depth.ndim == 3:
                        depth = depth[..., 0]
                    near, far = metadata["depth_near"], metadata["depth_far"]
                    capture.update(
                        depth_m=(near / (1 - depth * (1 - near / far)))[::-1],
                        metadata=metadata,
                    )
                captures[camera] = capture
            self._trajectory.append(
                env_step=self._episode_steps,
                action=action,
                robot_state=self.get_public_robot_state(),
                captures=captures,
                correlation=trace_context,
            )
        except Exception as exc:
            self._trajectory.fail(exc)
            raise TraceError(
                f"LIBERO trace capture failed at step {self._episode_steps}: {exc}"
            ) from exc

    def raw_obs(self) -> dict:
        return _to_numpy_tree(self._env.current_raw_obs[self._env_idx])

    def get_env_meta(self) -> dict:
        """Return the meta info this server was launched with."""
        return dict(self._meta)

    def close(self) -> None:
        """Release the server-owned LIBERO environment once."""
        if self._closed:
            return
        try:
            self.finish_trajectory()
        finally:
            self._env.env.close()
            self._closed = True

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return _to_numpy_tree(
            self._env.render_camera(
                camera_name=camera_name,
                height=height,
                width=width,
                depth=depth,
            )
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return _to_numpy_tree(
            self._env.get_camera_meta(
                camera_name=camera_name, height=height, width=width
            )
        )

    def get_task_language(self) -> str | None:
        return self._env.task_descriptions[self._env_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device to pin MuJoCo EGL rendering and the torch "
        "default device to (physical CUDA ordinal).",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        # Deliberately do NOT set CUDA_VISIBLE_DEVICES. robosuite (imported
        # transitively via libero) asserts at import time that
        # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` (substring check),
        # which assumes the EGL index equals the CUDA ordinal and crashes on
        # multi-GPU boxes where the EGL order differs. That assertion is gated
        # on ``CUDA_VISIBLE_DEVICES != ""``, so leaving it unset skips it in
        # both this process and the multiprocessing-spawned render workers
        # (which inherit the env). Pin the two backends directly instead:
        #   - MuJoCo render device <- MUJOCO_EGL_DEVICE_ID (configure_egl_device)
        #   - torch default device  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev,
                args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from rpent.utils.egl import configure_egl_device

        configure_egl_device(args.cuda_device)
        import torch

        torch.cuda.set_device(args.cuda_device)

    raw_env = make_env(
        args.task,
        args.seed,
        suite_name=args.suite,
        max_episode_steps=args.max_episode_steps,
    )
    facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        facade.close()


if __name__ == "__main__":
    main()
