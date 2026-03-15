import os
import urllib.request
import tarfile
import random

class BenchmarkLoader:
    """Downloads, generates or stages benchmark suites for processing (e.g. PolyBench)."""
    
    def __init__(self, data_dir="../../data/programs"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_polybench_scaffold(self):
        """Mock out loading polybench benchmark kernels (2D stencils, matrix mults)"""
        # In actual deployment, we would pull the polybench-c 4.2.1 tarball.
        # Here we scaffold generating C-files dynamically for the parser to consume
        
        kernels = ["gemm", "2mm", "3mm", "syrk", "heat-3d", "fdtd-2d", "jacobi-1d", "seidel-2d"]
        templates = []
        for k in kernels:
            filepath = os.path.join(self.data_dir, f"{k}.c")
            with open(filepath, "w") as f:
                f.write(self._mock_c_template(k))
            templates.append(filepath)
            
        print(f"Scaffolded {len(kernels)} PolyBench surrogate kernels in {self.data_dir}")
        return templates

    def generate_synthetic_loop_nests(self, num_programs=100):
        """Dynamically generate varying loop bounds for large scale dataset building"""
        synth_dir = os.path.join(self.data_dir, "synthetic")
        os.makedirs(synth_dir, exist_ok=True)
        
        generated = []
        for i in range(num_programs):
            depth = random.randint(1, 5)
            # Create a mock nested C loop
            code = f"void synth_{i}(float* A, float* B, int N) {{\n"
            indent = "  "
            for d in range(depth):
                bound = random.choice([10, 100, 1000, "N"])
                code += f"{indent}for(int i_{d}=0; i_{d} < {bound}; i_{d}++) {{\n"
                indent += "  "
                
            # Inner body
            code += f"{indent}A[i_0] += B[i_{depth-1}] * 2.0f;\n"
            
            # Close loops
            for d in range(depth):
                indent = indent[:-2]
                code += f"{indent}}}\n"
                
            code += "}\n"
            
            filepath = os.path.join(synth_dir, f"synth_{i}.c")
            with open(filepath, "w") as f:
                f.write(code)
            generated.append(filepath)
            
        print(f"Generated {num_programs} synthetic loop variants in {synth_dir}")
        return generated
        
    def _mock_c_template(self, name):
        return f"""
// PolyBench Surrogate for {name}
void kernel_{name}(int n, double alpha, double beta, double C[n][n], double A[n][n], double B[n][n]) {{
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {{
            C[i][j] *= beta;
            for (k = 0; k < n; k++)
                C[i][j] += alpha * A[i][k] * B[k][j];
        }}
}}
"""

if __name__ == "__main__":
    loader = BenchmarkLoader()
    loader.fetch_polybench_scaffold()
    loader.generate_synthetic_loop_nests(10)
