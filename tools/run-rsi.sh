#!/usr/bin/env bash
# Explicit config/run arguments use the existing persistent schema-2 runner.
set -euo pipefail
VISTA_LAUNCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$VISTA_LAUNCH_ROOT/configs/env.sh"
cd "$VISTA_LAUNCH_ROOT"
exec "$RPENT_PYTHON" -m meta_harness.cli run "$@"
