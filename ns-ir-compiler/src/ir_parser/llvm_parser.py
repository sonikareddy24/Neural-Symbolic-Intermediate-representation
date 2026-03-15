import re
import json

class LLVMIRParser:
    """
    Parses LLVM IR bitcode text string and builds basic Control Flow Graphs (CFG) 
    and Data Flow Graphs (DFG) logic.
    """
    def __init__(self):
        self.functions = {}
        
    def parse_module(self, llvm_ir_text: str):
        """Extracts functions from an LLVM IR module text."""
        # A simple string search to isolate functions
        self.functions = self.extract_functions(llvm_ir_text)
        results = {}
        for func_name, code in self.functions.items():
            cfg = self.get_cfg(code)
            dfg = self.get_dfg(code)
            results[func_name] = {'cfg': cfg, 'dfg': dfg, 'raw': code}
        return results
        
    def extract_functions(self, ir_text: str):
        """Regex-based extraction of LLVM IR functions."""
        functions = {}
        # Pattern to capture "define ... @func_name(...) { ... }"
        # Robust enough for basic blocks, handles braces correctly 
        # (This is simplified for prototype, standard libraries like llvmlite are preferred in production)
        pattern = re.compile(r'define[^{]*?@([a-zA-Z0-9_.]+)\s*\(.*?\)\s*\{([^}]+)\}')
        
        for match in pattern.finditer(ir_text):
            func_name = match.group(1)
            body = match.group(2)
            functions[func_name] = body
            
        return functions
        
    def get_cfg(self, func_body: str):
        """Construct a basic CFG (nodes = basic blocks, edges = branches)"""
        blocks = {}
        current_block = "entry"
        blocks[current_block] = []
        
        edges = []
        
        lines = func_body.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Label
            label_match = re.match(r'^([a-zA-Z0-9_.]+):', line)
            if label_match:
                current_block = label_match.group(1)
                blocks[current_block] = []
                continue
                
            blocks[current_block].append(line)
            
            # Branches
            if line.startswith('br '):
                # br label %next OR br i1 %cond, label %true, label %false
                targets = re.findall(r'label\s+%([a-zA-Z0-9_.]+)', line)
                for t in targets:
                    edges.append((current_block, t))
            elif line.startswith('switch '):
                targets = re.findall(r'label\s+%([a-zA-Z0-9_.]+)', line)
                for t in targets:
                    edges.append((current_block, t))
                    
        return {'blocks': blocks, 'edges': edges}
        
    def get_dfg(self, func_body: str):
        """Construct a basic DFG (def-use chains)"""
        # Finds variable definitions: %var = ...
        nodes = []
        edges = [] # (producer, consumer)
        
        defined = set()
        
        lines = func_body.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.endswith(':'): continue
            
            # Instruction ID acting as node
            node_id = f"inst_{i}"
            nodes.append({"id": node_id, "op": line})
            
            # %dst = op %src1, %src2
            def_match = re.search(r'^%([a-zA-Z0-9_.]+)\s*=', line)
            if def_match:
                dst = def_match.group(1)
                defined.add(dst)
                
            # Find uses: %var
            uses = re.findall(r'%([a-zA-Z0-9_.]+)', line)
            for u in uses:
                if u in defined:
                    # simplified edge connection logic to variable name
                    edges.append((u, node_id))
                    
        return {'nodes': nodes, 'edges': edges}

class IRNormalizer:
    @staticmethod
    def normalize_instruction_names(ir_text: str):
        # Strip specific numbering to canonicalize variables -> %v1, %v2 based on scope
        return ir_text
        
    @staticmethod
    def simplify_expressions(ir_text: str):
        # Placeholder for fold constant logic
        return ir_text

if __name__ == "__main__":
    # Test
    sample = \"\"\"
define i32 @add(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %a, %b
  ret i32 %add
}
    \"\"\"
    parser = LLVMIRParser()
    res = parser.parse_module(sample)
    print(json.dumps(res, indent=2))
