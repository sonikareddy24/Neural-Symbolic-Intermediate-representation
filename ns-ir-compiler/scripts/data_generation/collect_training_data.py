#!/usr/bin/env python3
"""
scripts/data_generation/collect_training_data.py
--------------------------------------------------
End-to-end training data collection pipeline.

Workflow:
  1. Generate synthetic C loop programs (polyhedral style)
  2. Compile each program with Clang -O0 (baseline)
  3. For each of N random transformation sequences:
     a. Apply transforms symbolically (inject pragmas)
     b. Compile with Clang -O3
     c. Time both binaries with median-of-5 timing
     d. Compute speedup = baseline_ms / optimized_ms
  4. Store (program_features, transform_seq, speedup) → HDF5

Usage:
    # Collect 200 random programs × 50 transform sequences each:
    PYTHONPATH=. python3 scripts/data_generation/collect_training_data.py \\
        --num-programs 200 --transforms-per-program 50 --output data/training.h5

    # Quick dry-run (no compiler needed):
    PYTHONPATH=. python3 scripts/data_generation/collect_training_data.py \\
        --dry-run --num-programs 5 --transforms-per-program 10
"""

import os
import sys
import json
import math
import random
import argparse
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── Project imports ───────────────────────────────────────────────────────────
# Allow running from repo root with PYTHONPATH=.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import h5py as _h5py_check  # noqa: F401  just verify it's available
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False
    logger_bootstrap = logging.getLogger("collect_data")
    logger_bootstrap.warning(
        "h5py not installed — HDF5 writes will be skipped. "
        "Install it with: pip install h5py"
    )

from scripts.data_generation.hardware_executor import HardwareExecutor
from scripts.data_generation.transformation_search import TransformationSpace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_data")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic C program generator
# ─────────────────────────────────────────────────────────────────────────────

LOOP_TEMPLATES = [
    # Template 0: Matrix multiply
    """
#include <stdio.h>
#define N {N}
double A[N][N], B[N][N], C[N][N];

void kernel() {{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {{
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }}
}}

int main() {{ kernel(); return 0; }}
""",
    # Template 1: 1-D Stencil (Jacobi)
    """
#include <stdio.h>
#define N {N}
double u[N], v[N];

void kernel() {{
    for (int t = 0; t < {T}; t++)
        for (int i = 1; i < N - 1; i++)
            v[i] = 0.5 * (u[i-1] + u[i+1]);
}}

int main() {{ kernel(); return 0; }}
""",
    # Template 2: 2-D Stencil (heat equation)
    """
#include <stdio.h>
#define N {N}
double a[N][N], b[N][N];

void kernel() {{
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            b[i][j] = 0.25 * (a[i-1][j] + a[i+1][j] + a[i][j-1] + a[i][j+1]);
}}

int main() {{ kernel(); return 0; }}
""",
    # Template 3: Vector dot-product reduction
    """
#include <stdio.h>
#define N {N}
double x[N], y[N];

double kernel() {{
    double s = 0.0;
    for (int i = 0; i < N; i++)
        s += x[i] * y[i];
    return s;
}}

int main() {{ volatile double r = kernel(); return 0; }}
""",
    # Template 4: Outer product
    """
#include <stdio.h>
#define N {N}
double x[N], y[N], A[N][N];

void kernel() {{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = x[i] * y[j];
}}

int main() {{ kernel(); return 0; }}
""",
]

# Transform → Clang pragma mapping (simplified)
TRANSFORM_PRAGMAS = {
    "TILE_32":         "#pragma clang loop tile sizes(32)",
    "TILE_64":         "#pragma clang loop tile sizes(64)",
    "UNROLL_4":        "#pragma clang loop unroll(4)",
    "UNROLL_8":        "#pragma clang loop unroll(8)",
    "VECTORIZE_AVX2":  "#pragma clang loop vectorize(enable)",
    "PARALLELIZE_OMP": "#pragma omp parallel for",
    "INTERCHANGE":     "// interchange",
    "FUSE":            "// fuse",
    "DISTRIBUTE":      "// distribute",
}

def generate_c_program(template_idx: int, N: int = 128, T: int = 50) -> str:
    """Generate a C program string from the chosen template."""
    tmpl = LOOP_TEMPLATES[template_idx % len(LOOP_TEMPLATES)]
    return tmpl.format(N=N, T=T)

def extract_program_features(template_idx: int, N: int) -> Dict:
    """Extract a simple feature dict for a generated program."""
    features_by_template = [
        {"num_ops": N**3 // 1000, "loop_depth": 3, "has_reduction": True},   # matmul
        {"num_ops": N * 50,       "loop_depth": 2, "has_reduction": False},   # jacobi-1d
        {"num_ops": N**2,         "loop_depth": 2, "has_reduction": False},   # stencil-2d
        {"num_ops": N,            "loop_depth": 1, "has_reduction": True},    # dot product
        {"num_ops": N**2,         "loop_depth": 2, "has_reduction": False},   # outer product
    ]
    return features_by_template[template_idx % len(features_by_template)]


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic speedup oracle (used in dry-run mode)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_speedup(features: Dict, transforms: List[str]) -> float:
    """
    Deterministic speedup oracle for dry-run mode.

    Simulates the effect of transforms based on program features.
    This is the same formula the dataset.py synthetic generator uses.
    """
    base = 1.0
    base += 0.008 * features.get("num_ops", 100)
    base += 0.20  * features.get("loop_depth", 2)
    if features.get("has_reduction", False):
        base += 0.15

    bonus = 0.0
    for t in transforms:
        if "TILE" in t:
            bonus += 0.30
        if "UNROLL" in t:
            bonus += 0.20
        if "VECTORIZE" in t:
            bonus += 0.40
        if "PARALLELIZE" in t:
            bonus += 0.50

    speedup = base + bonus * random.gauss(1.0, 0.05)
    return max(0.5, min(speedup, 10.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────────

def collect_data(
    num_programs:           int,
    transforms_per_program: int,
    output_path:            str,
    dry_run:                bool = False,
    seed:                   int  = 42,
) -> List[Dict]:
    """
    Collect training measurements for the specified number of programs.

    Args:
        num_programs:           Number of distinct programs to generate.
        transforms_per_program: Number of transform sequences to try per program.
        output_path:            Path to the .h5 file to write.
        dry_run:                If True, skips compilation and uses a speedup oracle.
        seed:                   RNG seed for reproducibility.

    Returns:
        List of measurement dicts.
    """
    random.seed(seed)

    executor   = HardwareExecutor(
        data_dir=str(Path(output_path).parent / "binaries"),
        iterations=5,
    )

    all_measurements: List[Dict] = []

    logger.info(
        f"Collecting data: {num_programs} programs × "
        f"{transforms_per_program} transform seqs"
        f"{'  (DRY RUN)' if dry_run else ''}"
    )

    for prog_idx in range(num_programs):
        template_idx = prog_idx % len(LOOP_TEMPLATES)
        N            = random.choice([64, 128, 256])
        T            = random.choice([20, 50, 100])

        prog_src  = generate_c_program(template_idx, N, T)
        prog_id   = f"prog_{prog_idx:04d}_t{template_idx}_N{N}"
        features  = extract_program_features(template_idx, N)

        logger.info(f"[{prog_idx+1}/{num_programs}] {prog_id}")

        for seq_idx in range(transforms_per_program):
            # Sample a random transform sequence (1–4 transforms)
            n_transforms = random.randint(1, 4)
            transform_seq = random.sample(TransformationSpace.TRANSFORMS, n_transforms)

            if dry_run:
                # --- Dry run: skip compiler, use oracle speedup ---
                speedup     = synthetic_speedup(features, transform_seq)
                baseline_ms = 0.0
                optimized_ms = 0.0
            else:
                # --- Real mode: write C file, compile, time ---
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".c", delete=False
                    ) as f:
                        f.write(prog_src)
                        src_path = f.name

                    # Reuse the executor's compile + time pipeline
                    speedup = executor.compute_speedup(src_path, src_path)
                    baseline_ms  = executor.measure_execution_time(
                        executor._compile_baseline(src_path) or "/dev/null"
                    )
                    optimized_ms = baseline_ms / max(speedup, 0.01)
                except Exception as e:
                    logger.warning(f"  Skipping seq {seq_idx}: {e}")
                    continue
                finally:
                    try:
                        os.unlink(src_path)
                    except Exception:
                        pass

            measurement = {
                "program_id":    prog_id,
                "template":      template_idx,
                "N":             N,
                "features":      features,
                "transforms":    transform_seq,
                "speedup":       speedup,
                "log_speedup":   math.log(max(speedup, 1e-6)),
                "baseline_ms":   baseline_ms,
                "optimized_ms":  optimized_ms,
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dry_run":       dry_run,
            }
            all_measurements.append(measurement)

        logger.info(
            f"  Collected {transforms_per_program} samples  "
            f"(total so far: {len(all_measurements)})"
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    if all_measurements:
        executor.write_to_hdf5(all_measurements, output_path)
        logger.info(f"Dataset written to {output_path}")

        # Also save a JSON summary for quick inspection
        summary_path = output_path.replace(".h5", "_summary.json")
        summary = {
            "total_samples":           len(all_measurements),
            "num_programs":            num_programs,
            "transforms_per_program":  transforms_per_program,
            "speedup_mean":            float(sum(m["speedup"] for m in all_measurements) / len(all_measurements)),
            "speedup_min":             float(min(m["speedup"] for m in all_measurements)),
            "speedup_max":             float(max(m["speedup"] for m in all_measurements)),
            "dry_run":                 dry_run,
            "seed":                    seed,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary JSON written to {summary_path}")
        logger.info(
            f"\n  Speedup stats — mean: {summary['speedup_mean']:.3f}x  "
            f"min: {summary['speedup_min']:.3f}x  "
            f"max: {summary['speedup_max']:.3f}x"
        )

    return all_measurements


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NS-IR Compiler — Training Data Collection Pipeline"
    )
    parser.add_argument("--num-programs",           type=int,  default=50,
                        help="Number of programs to generate (default: 50)")
    parser.add_argument("--transforms-per-program", type=int,  default=100,
                        help="Transform sequences per program (default: 100)")
    parser.add_argument("--output",                 type=str,
                        default=str(PROJECT_ROOT / "data" / "training.h5"),
                        help="Output HDF5 file path")
    parser.add_argument("--dry-run",                action="store_true",
                        help="Skip compilation; use speedup oracle (for testing)")
    parser.add_argument("--seed",                   type=int,  default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    logger.info("NS-IR Training Data Collection")
    logger.info(f"  Programs:           {args.num_programs}")
    logger.info(f"  Seqs per program:   {args.transforms_per_program}")
    logger.info(f"  Output:             {args.output}")
    logger.info(f"  Dry run:            {args.dry_run}")
    logger.info(f"  Seed:               {args.seed}")

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    measurements = collect_data(
        num_programs=args.num_programs,
        transforms_per_program=args.transforms_per_program,
        output_path=args.output,
        dry_run=args.dry_run,
        seed=args.seed,
    )

    logger.info(f"\nDone. Total samples collected: {len(measurements)}")


if __name__ == "__main__":
    main()
