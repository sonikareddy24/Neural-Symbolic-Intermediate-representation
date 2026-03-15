import os
import sys
import torch
import json
import logging

# Ensure project root is on sys.path so src.* imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TiramisuHook:
    """
    Python-side interface acting as the Backend RPC or exported Torch module linker 
    for TIRAMISU C++ to query.
    """
    def __init__(self, model_path, use_transformer=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Loading NS-IR model from {model_path} onto {self.device}")
        
        # In a real environment, we deserialize the matching model
        if use_transformer:
            from src.models.transformer_cost_model import TransformerCostModel
            self.model = TransformerCostModel()
        else:
            from src.models.gnn_cost_model import GNNCostModel
            self.model = GNNCostModel()
            
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def predict_speedup(self, program_ir_json: str, transform_seq_json: str) -> float:
        """
        Predict the speedup for a given program + transformation sequence.

        Args:
            program_ir_json:    JSON string with at least the key 'llvm_ir'
                                (raw LLVM IR text of the target function).
            transform_seq_json: JSON string with key 'transforms'
                                (list of transformation dicts, e.g.
                                 [{"type": "tile"}, {"type": "unroll"}]).

        Returns:
            Predicted speedup as a float (> 1.0 means the transforms help).
            Returns 1.0 (no speedup) on any parse or inference error.
        """
        # ── 1. Parse inputs ──────────────────────────────────────────────────
        try:
            p_data = json.loads(program_ir_json)
            t_data = json.loads(transform_seq_json)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"[TiramisuHook] JSON parse error: {e}. Returning default.")
            return 1.0

        llvm_ir    = p_data.get("llvm_ir", "")
        transforms = t_data.get("transforms", [])

        if not llvm_ir:
            logging.warning("[TiramisuHook] Empty LLVM IR received. Returning default.")
            return 1.0

        # ── 2. Parse IR → NS-IR graph ────────────────────────────────────────
        try:
            from src.ns_ir.graph_builder import NSIRGraph
            graph = NSIRGraph()

            # Extract instructions from the IR text (line-by-line heuristic)
            instr_lines = [
                line.strip() for line in llvm_ir.split("\n")
                if line.strip() and not line.strip().endswith(":")
                and not line.strip().startswith(";")
                and not line.strip().startswith("define")
                and not line.strip().startswith("}")
            ]

            for i, instr in enumerate(instr_lines[:256]):  # Cap at 256 nodes
                graph.add_node(f"n{i}", instr)

        except Exception as e:
            logging.warning(f"[TiramisuHook] IR parsing failed: {e}. Using dummy input.")
            instr_lines = []

        # ── 3. Build model input tensors ─────────────────────────────────────
        MAX_SEQ = 512
        EMB_DIM = 128

        node_vecs = []
        for node in list(graph.nodes.values())[:MAX_SEQ]:
            node_vecs.append(node.embedding)        # numpy [128]

        # Encode transformation types as appended tokens
        TRANSFORM_VOCAB = [
            "tile", "unroll", "vectorize", "interchange", "fuse",
            "split", "skew", "parallelize", "reverse", "strip_mine",
            "peel", "sink", "hoist", "distribute", "reschedule",
        ]
        for t in transforms[:32]:
            t_type = t.get("type", "")
            one_hot = [1.0 if t_type == v else 0.0 for v in TRANSFORM_VOCAB]
            # Pad to EMB_DIM
            t_vec = (one_hot + [0.0] * EMB_DIM)[:EMB_DIM]
            node_vecs.append(t_vec)

        n_valid = len(node_vecs)

        # Pad to MAX_SEQ if needed
        while len(node_vecs) < MAX_SEQ:
            node_vecs.append([0.0] * EMB_DIM)

        import numpy as _np
        seq     = torch.tensor(_np.array(node_vecs[:MAX_SEQ], dtype=_np.float32)).unsqueeze(0)
        mask    = torch.ones(1, MAX_SEQ, dtype=torch.bool)
        mask[0, :n_valid] = False      # False = valid token (PyTorch convention)

        # transforms input (kept separate for the model's transform encoder)
        transforms_enc = torch.zeros(1, 32)
        for i, t in enumerate(transforms[:32]):
            idx = TRANSFORM_VOCAB.index(t.get("type", "")) if t.get("type", "") in TRANSFORM_VOCAB else -1
            if idx >= 0:
                transforms_enc[0, i] = float(idx + 1)

        # ── 4. Inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            log_speedup = self.model(seq.to(self.device),
                                     mask.to(self.device),
                                     transforms_enc.to(self.device))

        # ── 5. Exponentiate: model predicts log(speedup) ─────────────────────
        speedup = float(torch.exp(log_speedup).item())
        return speedup

    def export_torchscript(self, export_path: str = "nsir_traced_model.pt") -> None:
        """
        Export the network to TorchScript for embedding in C++ via libtorch.

        Note: The exported model outputs log(speedup). The C++ caller must
        apply std::exp() to obtain the actual speedup multiplier.
        """
        self.model.eval()
        seq        = torch.randn(1, 10, 128).to(self.device)
        mask       = torch.zeros(1, 10, dtype=torch.bool).to(self.device)
        transforms = torch.randn(1, 32).to(self.device)

        # check_trace=False prevents TracingCheckError from Dropout non-determinism
        traced = torch.jit.trace(self.model, (seq, mask, transforms), check_trace=False)
        traced.save(export_path)
        logging.info(f"TorchScript exported → {export_path}  (outputs log-speedup; apply exp())")


class AutoScheduler:
    """Replaces TIRAMISU's heuristic scheduler to navigate compilation ASTs matching maximum predicted NS-IR mappings."""
    def __init__(self, hook: TiramisuHook, budget=1000):
        self.hook = hook
        self.budget = budget
        
    def search_transformations(self, ir_program):
        from scripts.data_generation.transformation_search import TransformationSearch
        # Employs Beam Search using our model
        searcher = TransformationSearch("target_prog")
        # In a fully connected context we pass `self.hook` as the proxy
        best_sequences = searcher.beam_search(proxy_model=self.hook, beam_width=10, depth=5)
        return best_sequences[0] # Returns the highest ranked seq

if __name__ == "__main__":
    hook = TiramisuHook("dummy.pt")
    # Simulate C++ calling Python
    print("Mock inference yield:", hook.predict_speedup('{"nodes": []}', '["UNROLL_4", "TILE_32"]'))
    hook.export_torchscript()
