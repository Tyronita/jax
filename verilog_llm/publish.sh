#!/usr/bin/env bash
# Turn this directory into its own standalone GitHub repo.
# Usage: ./publish.sh <repo-name> [--public|--private]
set -e
cd "$(dirname "$0")"
NAME="${1:?usage: ./publish.sh <repo-name> [--public|--private]}"
VIS="${2:---public}"
VIS="${VIS#--}"   # public | private

# fresh history rooted at this directory
rm -rf .git
git init -b main
git add .
git commit -m "Tiny GPT-style LLM in Verilog

Decoder-only transformer in synthesizable Verilog (Q16.16 fixed-point),
trained weights, autoregressive text generation, bit-exact against a Python
golden model. Includes architecture, FPGA->ASIC path, and cost analysis."

if command -v gh >/dev/null 2>&1; then
    gh repo create "$NAME" "--$VIS" --source=. --remote=origin --push
    echo "Published: $(gh repo view --json url -q .url)"
else
    echo "gh CLI not found. Create the repo on GitHub manually, then:"
    echo "  git remote add origin git@github.com:<you>/$NAME.git"
    echo "  git push -u origin main"
fi
