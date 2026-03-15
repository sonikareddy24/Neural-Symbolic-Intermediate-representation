class LoopAnalyzer:
    """Extracts loop characteristic features from the AST/Polyhedral sets"""
    
    def analyze_loop_nest(self, tiramisu_schedule: list):
        """
        Calculates nesting depth from the loop structure.
        """
        # Ex mapping from [ {type: loop, level: 0}, {type: loop, level: 1} ] -> depth 2
        depths = [s['level'] for s in tiramisu_schedule if s.get('type') == 'loop']
        max_depth = max(depths) + 1 if depths else 0
        return {"nesting_depth": max_depth}
        
    def compute_iteration_space_volume(self, tiramisu_domain: str):
        """
        A pseudo volume heuristic for a polyhedral domain { [i,j] : 0 <= i < 100 } -> 100
        In production, this relies on ISL (Integer Set Library) via ISLpy bindings.
        """
        import re
        # Extremely simplified metric looking at constants
        numbers = [int(n) for n in re.findall(r'\b\d+\b', tiramisu_domain)]
        volume = 1
        for n in numbers:
            if n > 0:
                volume *= n
        return {"iteration_volume": volume}


class MemoryAnalyzer:
    """Analyze memory access paradigms from IR nodes"""
    
    def compute_memory_footprint(self, llvm_nodes: list):
        """Estimate the byte size requested locally to detect cache miss probability"""
        loads = sum(1 for n in llvm_nodes if 'load' in n['op'])
        stores = sum(1 for n in llvm_nodes if 'store' in n['op'])
        # roughly assuming 4 bytes per access operation (i32 base)
        cache_est = (loads + stores) * 4
        return {"estimated_footprint_bytes": cache_est}
        
    def detect_access_patterns(self, tiramisu_computations):
        """Detect access strides based on polyhedral indexing: A(i, j) vs A(j, i)"""
        # Placeholders
        return {"stride_coalescing": True}

class DependencyAnalyzer:
    def compute_dependency_chains(self, dfg_edges):
        """Calculate max length of the data flow dependency graph (RAW deps)"""
        import networkx as nx
        G = nx.DiGraph(dfg_edges)
        try:
            dag_len = nx.dag_longest_path_length(G)
        except nx.NetworkXUnfeasible:
            # Cycle detected
            dag_len = len(G.nodes)
        return {"max_dependency_depth": dag_len}

class OperationCounter:
    """Counts operation mixes required for hardware bounds"""
    def count_by_opcode(self, nodes):
        arithmetic = 0
        memory = 0
        control = 0
        
        arith_ops = ['add', 'sub', 'mul', 'div', 'fadd', 'fsub', 'fmul', 'fdiv']
        mem_ops = ['load', 'store', 'alloca', 'getelementptr']
        ctrl_ops = ['br', 'switch', 'ret', 'invoke', 'call']
        
        for n in nodes:
            op_text = n['op'].split()[0] if n['op'] else ""
            if op_text in arith_ops: arithmetic += 1
            elif op_text in mem_ops: memory += 1
            elif op_text in ctrl_ops: control += 1
            
        return {
            "num_arithmetic_ops": arithmetic,
            "num_memory_ops": memory,
            "num_control_flow_ops": control,
            "total_ops": len(nodes)
        }
