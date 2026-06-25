#!/bin/sh
# Launches slm-rag with the bundled uv. serve.py fetches the model GGUFs on first
# run. Args pass straight through, e.g.  ./run.sh --cli   or   ./run.sh --port 8080
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
case "$(uname -s)" in
    Darwin) UV="$DIR/bin/uv.mac" ;;
    Linux)  UV="$DIR/bin/uv.linux" ;;
    *)      echo "Use run.bat on Windows" >&2; exit 1 ;;
esac
chmod +x "$UV" 2>/dev/null || true
# On Linux the generator/embedder run in-process via llama-cpp-python; install it
# from the prebuilt CPU wheel index rather than building from source.
exec "$UV" run --no-build-package llama-cpp-python \
    --index https://abetlen.github.io/llama-cpp-python/whl/cpu \
    "$DIR/serve.py" "$@"
