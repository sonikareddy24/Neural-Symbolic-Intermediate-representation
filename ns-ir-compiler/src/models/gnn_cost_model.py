import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool
except ImportError:
    print("PyTorch Geometric not installed. Model logic will be simulated.")
    SAGEConv = None
    global_mean_pool = None

class GNNCostModel(nn.Module):
    """
    Message Passing Network predicting execution speedup from NS-IR Graph representation.
    """
    def __init__(self, node_input_dim=128, hidden_dim=256, transform_dim=32, num_layers=6):
        super(GNNCostModel, self).__init__()
        self.num_layers = num_layers
        
        # 1. Node embedding projection
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        
        # 2. Graph Convolution Layers (using GraphSAGE for scale)
        self.convs = nn.ModuleList()
        if SAGEConv is not None:
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # 3. Transformation sequence encoder (summarizes the target compiler passes)
        # e.g., TILE_32 + UNROLL_4 encoded into a dense vector
        self.transform_encoder = nn.Sequential(
            nn.Linear(transform_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. Fusion and Prediction Head
        # Concatenate Graph features + Transformation Features -> Speedup scalar
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Target variable: predicted speedup multiplier (e.g. 1.4x)
        )

    def forward(self, x, edge_index, batch, transform_seq):
        """
        Args:
            x:             Node features [N, node_input_dim]
            edge_index:    Graph connectivity [2, E]
            batch:         Node-to-graph assignment [N]  (0…B-1)
            transform_seq: Transformation features [B, transform_dim]

        Returns:
            log_speedup: Predicted log(speedup) [B, 1]
                         Apply torch.exp() to convert to actual speedup multiplier.
        """
        # ── 1. Node projection ───────────────────────────────────────────────
        x = F.relu(self.node_proj(x))

        # ── 2. Message passing (or fallback) ─────────────────────────────────
        if SAGEConv is not None:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            graph_embedding = global_mean_pool(x, batch)       # [B, hidden_dim]
        else:
            # ── FIXED fallback: scatter mean over the batch vector ────────────
            # Old code: torch.mean(x, dim=0) → always a single [1, hidden_dim]
            #           ignores batch dimension entirely — WRONG for B > 1.
            # New code: per-graph mean, respecting batch assignments.
            B = transform_seq.size(0)
            graph_embedding = torch.zeros(B, x.size(1), device=x.device, dtype=x.dtype)
            counts          = torch.zeros(B, 1,          device=x.device, dtype=x.dtype)
            graph_embedding.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            counts.scatter_add_(0, batch.unsqueeze(1), torch.ones(x.size(0), 1, device=x.device))
            counts = counts.clamp(min=1.0)          # avoid division by zero
            graph_embedding = graph_embedding / counts

        # ── 3. Encode compiler transformations ───────────────────────────────
        transform_embedding = self.transform_encoder(transform_seq)     # [B, hidden_dim]

        # ── 4. Fuse and predict ───────────────────────────────────────────────
        fused = torch.cat([graph_embedding, transform_embedding], dim=1)

        # Output is log(speedup) — same convention as TransformerCostModel.
        # Caller applies torch.exp() to get the real speedup multiplier.
        log_speedup = self.prediction_head(fused)
        return log_speedup

if __name__ == "__main__":
    # Test network compilation
    model = GNNCostModel()
    mock_nodes = torch.randn(10, 128)
    mock_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    mock_batch = torch.zeros(10, dtype=torch.long)
    mock_transforms = torch.randn(1, 32)
    
    out = model(mock_nodes, mock_edges, mock_batch, mock_transforms)
    print("Predicted speedup:", out.item())
