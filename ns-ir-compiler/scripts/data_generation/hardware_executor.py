import sys
import os
import subprocess
import time
import logging
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # write_to_hdf5 will log a clear error if this is needed

class HardwareExecutor:
    """
    Compiles and executes permutations on bare-metal to map "True" Target execution times.
    Requires isolation (no perf frequency scaling ideally) to measure deterministically.
    """
    def __init__(self, data_dir="../../data/measurements", iterations=10):
        self.data_dir   = data_dir
        self.iterations = iterations
        self.logger     = logging.getLogger(self.__class__.__name__)
        os.makedirs(self.data_dir, exist_ok=True)

        
    def _compile_baseline(self, src_file):
        """Compile with -O0 as the slow bound reference"""
        out_bin = os.path.join(self.data_dir, "baseline.out")
        cmd = ["clang", "-O0", src_file, "-o", out_bin]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_bin
        except subprocess.CalledProcessError:
            print(f"Failed compiling {src_file}")
            return None
            
    def _compile_optimized(self, src_file, temp_tiramisu_optimized_file):
        """Compile transformed IR with -O2 backend passes"""
        out_bin = os.path.join(self.data_dir, "optimized.out")
        cmd = ["clang", "-O2", temp_tiramisu_optimized_file, "-o", out_bin]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_bin
        except subprocess.CalledProcessError:
            return None
            
    def measure_execution_time(self, executable_path: str, repeats: int = 5,
                                timeout_s: float = 30.0) -> float:
        """
        Measure wall-clock execution time for a compiled binary.

        Runs the binary ``repeats`` times and returns the **median** elapsed
        time in milliseconds to reduce measurement noise.

        Args:
            executable_path: Path to the compiled binary.
            repeats:         Number of timing runs (default: 5).
            timeout_s:       Max seconds per run before aborting (default: 30).

        Returns:
            Median execution time in milliseconds.  Returns -1.0 if the binary
            fails to run (non-zero exit code or timeout).
        """
        import subprocess, time, statistics, os

        if not os.path.isfile(executable_path):
            self.logger.error(f"Binary not found: {executable_path}")
            return -1.0

        timings_ms = []

        for run_idx in range(repeats):
            try:
                t0 = time.perf_counter()
                result = subprocess.run(
                    [executable_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout_s,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1_000.0

                if result.returncode != 0:
                    self.logger.warning(
                        f"Run {run_idx+1}/{repeats}: non-zero exit code {result.returncode}"
                    )
                    continue

                timings_ms.append(elapsed_ms)

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Run {run_idx+1}/{repeats}: timed out after {timeout_s}s")
                return -1.0
            except FileNotFoundError:
                self.logger.error(f"Executable not found: {executable_path}")
                return -1.0

        if not timings_ms:
            self.logger.error(f"All {repeats} runs failed for {executable_path}")
            return -1.0

        median_ms = statistics.median(timings_ms)
        self.logger.debug(
            f"Execution times: {[f'{t:.2f}' for t in timings_ms]}ms  "
            f"→ median {median_ms:.2f}ms"
        )
        return median_ms

    def compute_speedup(self, src_file, optimized_src_file):
        """Returns the multiplicative speedup mapping"""
        base_bin = self._compile_baseline(src_file)
        opt_bin  = self._compile_optimized(src_file, optimized_src_file)

        if not base_bin or not opt_bin:
            return 1.0  # Compilation failed — no speedup

        base_med = self.measure_execution_time(base_bin)
        opt_med  = self.measure_execution_time(opt_bin)

        if base_med <= 0 or opt_med <= 0:
            return 1.0  # Execution failed

        # Clip absurd outliers (hardware stuttering)
        import numpy as np
        speedup = base_med / (opt_med + 1e-9)
        return float(np.clip(speedup, 0.5, 10.0))
        
    def write_to_hdf5(self, measurements: list, output_path: str) -> None:
        """
        Persist a list of measurement dicts to an HDF5 file.

        Falls back to JSON if h5py is not installed.
        """
        import os, json
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self.logger.info(f"Writing {len(measurements)} measurements → {output_path}")

        if h5py is None:
            # Fallback: write as newline-delimited JSON for easy loading
            json_path = output_path.replace(".h5", ".jsonl")
            with open(json_path, "w") as f:
                for m in measurements:
                    # Ensure JSON-serializable
                    record = {k: (v if not isinstance(v, (list, dict, str, bool, float, int)) else v)
                              for k, v in m.items()}
                    f.write(json.dumps(record) + "\n")
            self.logger.info(f"h5py unavailable — wrote JSONL fallback: {json_path}")
            return

        import numpy as np
        with h5py.File(output_path, "a") as f:
            for m in measurements:
                pid   = str(m.get("program_id", "unknown")).replace("/", "_")
                grp   = f.require_group(pid)
                ds_key = str(len(grp.keys()))
                ds    = grp.create_dataset(ds_key, data=np.float32(float(m.get("speedup", 1.0))))
                ds.attrs["speedup"]      = float(m.get("speedup",      1.0))
                ds.attrs["log_speedup"]  = float(m.get("log_speedup",  0.0))
                ds.attrs["baseline_ms"]  = float(m.get("baseline_ms",  0.0))
                ds.attrs["optimized_ms"] = float(m.get("optimized_ms", 0.0))
                ds.attrs["transforms"]   = str(m.get("transforms", []))
                ds.attrs["timestamp"]    = str(m.get("timestamp",   ""))
        self.logger.info(f"HDF5 write complete: {output_path}")


if __name__ == "__main__":
    executor = HardwareExecutor()
    # Mocking generating a placeholder C file
    test_src = os.path.join(executor.data_dir, "test.c")
    with open(test_src, "w") as f:
        f.write("int main() { return 0; }")
        
    speedup = executor.compute_speedup(test_src, test_src)
    executor.write_to_hdf5("test_program_01", ["UNROLL_4", "TILE_32"], speedup)
