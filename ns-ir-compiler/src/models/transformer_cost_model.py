"""
src/models/transformer_cost_model.py  (v2 — State-of-the-Art upgrade)
------------------------------------------------------------------------
Changes vs. v1:
  - CrossAttentionFusion replaces concat+MLP fusion (novel contribution)
  - RoPE replaces sinusoidal positional encoding (better generalisation)
  - PositionalEncoding kept as fallback PE inside TransformerEncoder
  - predict_with_uncertainty() — MC Dropout for confidence intervals
  - get_ir_embedding() exposed for contrastive loss computation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cross_attention_fusion import CrossAttentionFusion, RotaryEmbedding


# ══════════════════════════════════════════════════════════════════════════════
# RoPE-enhanced Multi-Head Self-Attention (for the IR encoder)
# ══════════════════════════════════════════════════════════════════════════════

class RoPETransformerEncoderLayer(nn.Module):
    """
    Pre-LayerNorm Transformer encoder layer with Rotary Positional Embeddings.

    Differences from nn.TransformerEncoderLayer:
      - RoPE applied to Q, K inside self-attention (not external PE addition)
      - Pre-LN (norm_first=True behaviour, but hand-rolled for RoPE integration)
      - GELU activation in FFN
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 max_seq: int = 4096):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model   = d_model
        self.nhead     = nhead
        self.head_dim  = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.rope  = RotaryEmbedding(self.head_dim, max_seq=max_seq)

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _self_attn(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H, HD   = self.nhead, self.head_dim

        Q = self.q_proj(x).reshape(B, S, H, HD).transpose(1, 2)
        K = self.k_proj(x).reshape(B, S, H, HD).transpose(1, 2)
        V = self.v_proj(x).reshape(B, S, H, HD).transpose(1, 2)

        Q, K = self.rope(Q, K)  # apply RoPE to both Q and K

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(HD)
        if mask is not None:
            # mask: [B, S]  True = padding  → [B, 1, 1, S]
            scores = scores.masked_fill(mask[:, None, None, :], float("-inf"))
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.dropout(attn_w)
        out    = torch.matmul(attn_w, V).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-LN self-attention
        x = x + self.dropout(self._self_attn(self.norm1(x), src_key_padding_mask))
        # Pre-LN FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════

class TransformerCostModel(nn.Module):
    """
    State-of-the-art NS-IR cost model.

    Pipeline:
        node embeddings [B, S, 128]
            → Linear projection → [B, S, d_model]
            → 6 x RoPE-enhanced Pre-LN Transformer encoder layers
            → IR memory  [B, S, d_model]
                  │
                  └─► CrossAttentionFusion(ir_memory, transform_ids)
                              │
                              └─► predicted log(speedup)  [B, 1]

    Novel contributions over baseline:
      1. RoPE in the IR encoder (better length generalisation)
      2. Cross-attention fusion instead of concat+MLP
      3. Gated residual in fusion layer
      4. MC Dropout for uncertainty quantification
    """

    def __init__(self,
                 node_input_dim: int = 128,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 transform_seq_len: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # ── Input projection ─────────────────────────────────────────────────
        self.node_proj = nn.Linear(node_input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # ── RoPE-enhanced Transformer encoder ────────────────────────────────
        self.encoder_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # ── Cross-Attention Fusion (novel contribution) ───────────────────────
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            nhead=nhead,
            transform_seq_len=transform_seq_len,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── IR encoding (exposed for contrastive loss) ────────────────────────────

    def encode_ir(self, src_seq: torch.Tensor,
                  src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode an IR node sequence to a memory tensor.

        Args:
            src_seq:              [B, seq_len, node_input_dim]
            src_key_padding_mask: [B, seq_len]  True = padding

        Returns:
            memory: [B, seq_len, d_model]
        """
        x = self.input_norm(self.node_proj(src_seq))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.encoder_norm(x)

    def get_ir_embedding(self, src_seq: torch.Tensor,
                         src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool IR memory to a single program-level vector.
        Used by the contrastive loss.

        Returns: [B, d_model]
        """
        memory = self.encode_ir(src_seq, src_key_padding_mask)
        valid  = (~src_key_padding_mask).unsqueeze(-1).float()
        return (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(self,
                src_seq: torch.Tensor,
                src_key_padding_mask: torch.Tensor,
                transform_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_seq:              [B, seq_len, node_input_dim]
            src_key_padding_mask: [B, seq_len]  True = padding token
            transform_seq:        [B, transform_seq_len]  integer type IDs

        Returns:
            speedup: [B, 1]  strictly positive (via F.softplus inside fusion head)
        """
        # Ensure transform_seq is long tensor for embedding lookup
        if transform_seq.dtype != torch.long:
            transform_ids = transform_seq.long().clamp(0, self.fusion.N_TRANSFORMS - 1)
        else:
            transform_ids = transform_seq.clamp(0, self.fusion.N_TRANSFORMS - 1)

        memory = self.encode_ir(src_seq, src_key_padding_mask)
        log_speedup = self.fusion(memory, src_key_padding_mask, transform_ids)
        return F.softplus(log_speedup)

    # ── Uncertainty quantification (MC Dropout) ───────────────────────────────

    def predict_with_uncertainty(self,
                                 src_seq: torch.Tensor,
                                 src_key_padding_mask: torch.Tensor,
                                 transform_seq: torch.Tensor,
                                 n_samples: int = 30
                                 ):
        """
        Monte Carlo Dropout uncertainty estimation.

        Runs n_samples forward passes with dropout ENABLED (model in train mode
        for dropout only) and returns mean and std of the predictions.

        Args:
            n_samples: Number of stochastic forward passes.

        Returns:
            mean_speedup: [B, 1]
            std_speedup:  [B, 1]  — higher = more uncertain
        """
        # Enable dropout (train mode) but disable gradient computation
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(src_seq, src_key_padding_mask, transform_seq))
        self.eval()

        stacked = torch.stack(preds, dim=0)   # [n_samples, B, 1]
        return stacked.mean(dim=0), stacked.std(dim=0)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = TransformerCostModel()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")

    B = 4
    seq        = torch.randn(B, 50, 128)
    mask       = torch.ones(B, 50, dtype=torch.bool)
    mask[:, :20] = False
    t_ids      = torch.randint(0, 15, (B, 32))

    # Standard forward
    out = model(seq, mask, t_ids)
    print(f"Forward output shape: {out.shape}")
    assert out.shape == (B, 1)
    assert (out > 0).all(), "All outputs must be positive"

    # IR embedding for contrastive loss
    emb = model.get_ir_embedding(seq, mask)
    print(f"IR embedding shape: {emb.shape}")
    assert emb.shape == (B, 256)

    # Uncertainty
    mean, std = model.predict_with_uncertainty(seq, mask, t_ids, n_samples=10)
    print(f"Uncertainty — mean: {mean.squeeze().tolist()}")
    print(f"Uncertainty — std:  {std.squeeze().tolist()}")

    print("\nAll checks PASSED")
