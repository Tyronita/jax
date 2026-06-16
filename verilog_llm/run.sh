#!/usr/bin/env bash
# Train (if needed), then compile + simulate the Verilog LLM.
set -e
cd "$(dirname "$0")"

if [ "$1" = "train" ] || [ ! -f weights/golden.hex ]; then
    echo "== training + exporting weights =="
    (cd ref && python3 train_and_export.py)
fi

echo "== compiling RTL =="
iverilog -g2012 -I rtl -o build/sim tb/tb_gpt.v rtl/gpt.v

echo "== simulating (weights loaded from ./weights) =="
vvp build/sim
