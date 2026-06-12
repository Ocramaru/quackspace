#!/usr/bin/env bash
# Install the `quack` CLI globally. Built to be run straight from a URL:
#
#   curl -fsSL https://raw.githubusercontent.com/Ocramaru/quackspace/main/install.sh | bash
#
# It ensures `uv` is present, then installs the `quackspace` package — which
# provides the `quack` and `quack-mcp` commands — onto your PATH. When it's
# done, run `quack init <name>` to create and scaffold a new knowledge space.
set -euo pipefail

log() { printf '\033[1;36m▸\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# 1. uv — the installer/runtime we lean on (no system Python required).
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv is unavailable; see https://docs.astral.sh/uv/"

# 2. Choose where to install from. Order of preference:
#    QUACK_SOURCE env  >  a local checkout (this script sits in it)  >  PyPI.
src="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd -P "$(dirname "$src")" 2>/dev/null && pwd || true)"
if [ -n "${QUACK_SOURCE:-}" ]; then
  SOURCE="$QUACK_SOURCE"
elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ] \
     && grep -q 'name = "quackspace"' "$SCRIPT_DIR/pyproject.toml" 2>/dev/null; then
  SOURCE="$SCRIPT_DIR"                       # running from a checkout
else
  SOURCE="quackspace"                        # piped from the web → PyPI
fi

log "Installing quack from: $SOURCE"
uv tool install --force "$SOURCE"

# 3. Make sure the tool bin is on PATH now and in future shells.
uv tool update-shell >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"

if command -v quack >/dev/null 2>&1; then
  ok "quack installed → $(command -v quack)"
else
  ok "quack installed."
  log "Add uv's tool bin (e.g. ~/.local/bin) to PATH, then restart your shell."
fi

cat <<'EOF'

Next:
  quack init my-workspace   # create & scaffold a new space (or `quack init` here)
  cd my-workspace
  quack mcp install         # connect an LLM (Claude Code, Kiro, …) over MCP
EOF
