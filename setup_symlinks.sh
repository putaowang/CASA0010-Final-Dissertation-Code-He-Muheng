#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
sudo mkdir -p "/Users/muhenghe/Documents/BYLW"
sudo ln -sfn "$ROOT/start" "/Users/muhenghe/Documents/BYLW/start"
sudo ln -sfn "$ROOT/项目初始" "/Users/muhenghe/Documents/BYLW/项目初始"
echo "✅ Symlinks ready."
