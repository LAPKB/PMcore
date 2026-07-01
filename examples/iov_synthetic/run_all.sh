#!/bin/bash
# SDE IOV Verification — fully automated reproduction pipeline.
#
# Usage: ./run_all.sh
#
# 1. Generates synthetic data (Python, no deps)
# 2. Runs the verification pipeline (Rust, release build)
# 3. Generates the plot (Python + matplotlib in a venv)
#
# Output: data.csv, surface.csv, optimizer.csv, verification.png

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 1/3: Generate synthetic data ==="
python3 generate_data.py

echo ""
echo "=== Step 2/3: Run verification pipeline (cargo --release) ==="
cd ../..
cargo run --release --example iov_verify
cd examples/iov_synthetic

echo ""
echo "=== Step 3/3: Generate plot ==="

# Set up venv if needed
if [ ! -d .venv ]; then
	echo "  Creating venv..."
	python3 -m venv .venv
fi

# Install deps if not present
if ! .venv/bin/python -c "import matplotlib" 2>/dev/null; then
	echo "  Installing matplotlib + numpy..."
	.venv/bin/pip install --quiet matplotlib numpy
fi

.venv/bin/python plot.py

echo ""
echo "Done. Outputs:"
echo "  data.csv           — synthetic subjects (trueskee=0.08)"
echo "  surface.csv        — high-precision likelihood sweep"
echo "  optimizer.csv      — NelderMead recovery per particle count"
echo "  verification.png   — plot"
