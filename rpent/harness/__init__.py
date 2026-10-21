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

"""Load and bind minimal file-backed RPent harnesses.

The module's complete caller-facing workflow is two operations:

``load_harness(path)``
    Validate and snapshot a Git-versioned package without touching a robot.

``bind_harness(..., capabilities=..., episode_active=...)``
    Resolve the declared aliases and return instructions plus a restricted
    Toolkit view for one live episode.
"""

from rpent.harness.contracts import (
    BoundHarness,
    CapabilityBinding,
    CapabilitySpec,
    HarnessError,
    HarnessHost,
    HarnessIdentity,
    HarnessPackage,
    PlannerRequest,
    ToolDecision,
    ToolBinding,
)
from rpent.harness.package import load_harness
from rpent.harness.runtime import (
    HarnessToolkitView,
    PolicyRuntime,
    SkillRuntime,
    bind_harness,
    inspect_harness,
)

__all__ = [
    "BoundHarness",
    "CapabilityBinding",
    "CapabilitySpec",
    "HarnessError",
    "HarnessHost",
    "HarnessIdentity",
    "HarnessPackage",
    "HarnessToolkitView",
    "PlannerRequest",
    "PolicyRuntime",
    "ToolDecision",
    "SkillRuntime",
    "ToolBinding",
    "bind_harness",
    "inspect_harness",
    "load_harness",
]
