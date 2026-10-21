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

"""Compatibility shims for the installed robosuite / liberopro pair.

rpent-liberopro 0.2.0's ``OnTheGroundPanda`` (floor-standing scenes used by
libero_object_task and friends) declares ``default_base = None``. robosuite
1.5.2's ``robot_base_factory`` does not handle None: the missing key resolves
to the error-message string, which is then called and raises
``TypeError: 'str' object is not callable`` while loading the robot. robosuite
1.5's API represents a mount-less robot with ``NullMount``, so map None to it.
"""

from __future__ import annotations

_applied = False


def apply() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    try:
        from liberopro.liberopro.envs.robots.on_the_ground_panda import (
            OnTheGroundPanda,
        )
        from robosuite.models.bases import BASE_MAPPING
    except ImportError:
        return
    if "NullMount" not in BASE_MAPPING:
        return
    getter = OnTheGroundPanda.default_base
    if isinstance(getter, property) and getter.fget(None) is None:
        OnTheGroundPanda.default_base = property(lambda self: "NullMount")
