#!/bin/sh
set -eu
root=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
model=$1
shift
case "$model" in
 sf3d-mps|spar3d-mps) adapter=native_adapter ;;
 trellis-clean) adapter=backend ;;
 *) echo "Unknown backend: $model" >&2; exit 2 ;;
esac
export PYTHONPATH="$root/backends/$model:$root/shared/proto"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_CACHE="$root/.runtime/huggingface"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export U2NET_HOME="$root/.runtime/rembg"
export NO_ALBUMENTATIONS_UPDATE=1
cd "$root/backends/$model"
if [ "$model" = trellis-clean ]; then
 export TRELLIS_LIBRARY
 TRELLIS_LIBRARY=$(cat "$root/.runtime/trellis-library")
 export DYLD_LIBRARY_PATH
 DYLD_LIBRARY_PATH=$(dirname "$TRELLIS_LIBRARY")
fi
exec "$root/.venv/bin/python" -u "$adapter/server.py" "$@"
