import json
import numpy as np
import torch

try:
    from src.ns_ir.learned_embeddings import LearnedEmbeddingGenerator
    _LEARNED_EMB_AVAILABLE = True
except ImportError:
    _LEARNED_EMB_AVAILABLE = False

class NSIRNode:
    """Represents a single IR instruction/node combining symbolic + neural vectors"""
    def __init__(self, node_id: str, symbolic_ir: str, metadata: dict = None):
        self.node_id = node_id
        self.symbolic_ir = symbolic_ir
        self.metadata = metadata or {}
        # Size of the node embedding (e.g. 128 to 512 dimensions)
        # Random initialisation fallback
        self.embedding = np.random.rand(128).astype(np.float32)
        
    def update_embedding(self, new_embedding):
        self.embedding = new_embedding
        
    def serialize(self):
        return {
            "node_id": self.node_id,
            "symbolic_ir": self.symbolic_ir,
            "metadata": self.metadata,
            "embedding_norm": float(np.linalg.norm(self.embedding))
        }

class EmbeddingGenerator:
    """
    Generate initial token embeddings for IR instructions.

    Uses LearnedEmbeddingGenerator (nn.Module) when available.
    Falls back to a deterministic random hash for offline/no-torch environments.
    """
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim

        if _LEARNED_EMB_AVAILABLE:
            self._learned = LearnedEmbeddingGenerator(
                vocab_size=10_000,
                embedding_dim=embedding_dim,
            )
            self._use_learned = True
        else:
            self._use_learned = False

    def encode_instruction(self, instruction_text: str) -> np.ndarray:
        """
        Encode a single instruction to a numpy vector.

        When the learned embedder is available, its current (possibly
        un-trained) weight table is used; the output will improve once the
        model has been trained.  Falls back to deterministic random vectors
        only as a last resort.
        """
        if self._use_learned:
            with torch.no_grad():
                vec = self._learned(instruction_text)
            return vec.numpy()

        # Fallback: deterministic random hash (old behaviour)
        seed_val = hash(instruction_text) % (2 ** 32)
        rng = np.random.RandomState(seed_val)
        return rng.randn(self.embedding_dim).astype(np.float32)

    @property
    def learned_module(self):
        """Expose the underlying nn.Module for joint optimisation."""
        return self._learned if self._use_learned else None


class NSIRGraph:
    """Full program representation linking Nodes and Edges"""
    def __init__(self):
        self.nodes = {}  # ID → NSIRNode
        self.edges = []  # List of tuples (src_id, dst_id, edge_type)
        self.global_features = {}  # Program-level bounds
        self.embedder = EmbeddingGenerator()

    def add_node(self, node_id: str, symbolic_ir: str, metadata: dict = None):
        n = NSIRNode(node_id, symbolic_ir, metadata)
        n.update_embedding(self.embedder.encode_instruction(symbolic_ir))
        self.nodes[node_id] = n
        
    def add_edge(self, src: str, dst: str, edge_type="control_flow"):
        self.edges.append((src, dst, edge_type))
        
    def to_pytorch_geometric(self):
        """Converts internal graph to a distinct PyTorch Geometric format (Data) for GNN models"""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            print("PyTorch Geometric not installed. Returning standard payload.")
            return None
            
        # Map IDs to integers
        id_map = {n_id: i for i, n_id in enumerate(self.nodes.keys())}
        
        node_features = []
        for n_id in id_map.keys():
            node = self.nodes[n_id]
            # Concatenate any global scalar symbolic features with the high-dimensional neural tokens
            node_features.append(node.embedding)
            
        x = torch.tensor(node_features, dtype=torch.float)
        
        edge_indices = [[], []]
        for src, dst, _ in self.edges:
            if src in id_map and dst in id_map:
                edge_indices[0].append(id_map[src])
                edge_indices[1].append(id_map[dst])
                
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)

    def to_transformer_sequence(self):
        """Linearize the IR for Transformer Models"""
        import torch
        # topological/lexical sequence
        sequence = []
        for n in self.nodes.values():
            sequence.append(n.embedding)
        return torch.tensor(sequence, dtype=torch.float)

if __name__ == "__main__":
    # Test builder
    graph = NSIRGraph()
    graph.add_node("b1", "define i32 @add(i32 %a, i32 %b) {")
    graph.add_node("b2", "  %add = add nsw i32 %a, %b")
    graph.add_node("b3", "  ret i32 %add")
    graph.add_edge("b1", "b2", "control")
    graph.add_edge("b2", "b3", "data")
    
    print(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges.")
    print("Transformer Shape:", graph.to_transformer_sequence().shape)
