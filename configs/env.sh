#!/usr/bin/env bash
# Source this file from Bash or Zsh. Credentials belong in env.local.sh.
if [ -n "${BASH_VERSION:-}" ]; then
    VISTA_ENV_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    VISTA_ENV_FILE="${(%):-%x}"
else
    printf '%s\n' 'Source configs/env.sh from Bash or Zsh.' >&2
    return 1
fi
export RPENT_REPO_ROOT="$(cd -- "$(dirname -- "$VISTA_ENV_FILE")/.." && pwd -P)"
export RPENT_REPO="$RPENT_REPO_ROOT"
if [ -f "$RPENT_REPO_ROOT/configs/env.local.sh" ]; then
    source "$RPENT_REPO_ROOT/configs/env.local.sh"
fi
export RPENT_PYTHON="${RPENT_PYTHON:-$(command -v python)}"
export PYTHONPATH="$RPENT_REPO_ROOT/tools/runtime_python:$RPENT_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export LIBERO_TYPE="${LIBERO_TYPE:-pro}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-$NO_PROXY}"
unset VISTA_ENV_FILE
