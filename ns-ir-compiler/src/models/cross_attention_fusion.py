"""
src/models/cross_attention_fusion.py
--------------------------------------
Novel architectural contribution: Cross-Attention Fusion between IR graph
embeddings and transformation sequence embeddings.

Key idea (vs. prior work):
  - Tiramisu (NeurIPS 2021): LSTM over schedule tree, no IR encoding
  - Ansor/TVM (OSDI 2020):   MLP over handcrafted feature vector, no cross-attention
  - Ours: IR token sequence attends over transformation tokens → the model
    learns WHICH instructions are relevant to EACH transformation type.

Additionally implements Rotary Positional Embeddings (RoPE, Su et al. 2021)
which generalise better to program lengths not seen during training than the
sinusoidal PE used in the original Transformer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding (RoPE)
# ══════════════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021 — RoFormer).

    Unlike sinusoidal PE which adds position information, RoPE *rotates*
    query and key vectors.  This encodes relative positions natively inside
    the attention score, making the model length-extrapolation-friendly.

    Usage:
        rope = RotaryEmbedding(dim=head_dim)
        q, k = rope(q, k)   # q, k: [B, heads, seq, head_dim]
    """

    def __init__(self, dim: int, max_seq: int = 4096, base: int = 10_000):
        super().__init__()
        assert dim % 2 == 0, "RoPE requires even head dimension"
        self.dim = dim

        # Precompute inverse frequencies  [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache cos/sin tables up to max_seq
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # [seq, dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)      # [seq, dim]
        self.register_buffer("cos_cache", emb.cos()[None, None])  # [1,1,seq,dim]
        self.register_buffer("sin_cache", emb.sin()[None, None])

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        Args:
            q, k: [B, heads, seq_len, head_dim]
        Returns:
            q_rot, k_rot with rotary embedding applied.
        """
        seq_len = q.shape[2]
        if seq_len > self.cos_cache.shape[2]:
            self._build_cache(seq_len)

        cos = self.cos_cache[:, :, :seq_len, :]
        sin = self.sin_cache[:, :, :seq_len, :]

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Attention Fusion Module
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion between IR graph context and transformation tokens.

    Architecture:
        IR tokens (query)  ──┐
                              ├─► Multi-Head Cross-Attention ─► fused representation
        Transform tokens (kv)─┘           │
                                           └─► Gated residual ─► MLP head ─► speedup

    This allows the model to learn WHICH IR instructions are implicated by
    WHICH transformation types — something neither concat-MLP nor pure
    self-attention can express.

    The transformation sequence is first expanded into a per-type token
    sequence using a small embedding table, so the cross-attention has
    fine-grained per-transformation information to attend over.
    """

    TRANSFORM_VOCAB = [
        "tile", "unroll", "vectorize", "interchange", "fuse",
        "split", "skew", "parallelize", "reverse", "strip_mine",
        "peel", "sink", "hoist", "distribute", "reschedule",
        "<pad>",
    ]
    N_TRANSFORMS = len(TRANSFORM_VOCAB)

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 transform_seq_len: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model          = d_model
        self.nhead            = nhead
        self.head_dim         = d_model // nhead
        self.transform_seq_len = transform_seq_len

        assert d_model % nhead == 0

        # ── Transform token embedding (type + position) ───────────────────────
        self.transform_type_emb = nn.Embedding(self.N_TRANSFORMS, d_model,
                                               padding_idx=self.N_TRANSFORMS - 1)
        self.transform_pos_emb  = nn.Embedding(transform_seq_len, d_model)

        # ── Cross-attention projections ───────────────────────────────────────
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE for query side only (IR tokens)
        self.rope = RotaryEmbedding(self.head_dim)

        # ── Gated residual ────────────────────────────────────────────────────
        self.gate = nn.Linear(d_model * 2, d_model)

        # ── Post-fusion FFN ───────────────────────────────────────────────────
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        # ── Prediction head ───────────────────────────────────────────────────
        self.head_norm = nn.LayerNorm(d_model)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.transform_type_emb.weight, std=0.02)
        nn.init.normal_(self.transform_pos_emb.weight,  std=0.02)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_transforms(self, transform_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            transform_ids: [B, transform_seq_len]  integer type IDs (0-14 or pad=15)
        Returns:
            [B, transform_seq_len, d_model]
        """
        B, L = transform_ids.shape
        pos = torch.arange(L, device=transform_ids.device).unsqueeze(0)  # [1, L]
        return self.transform_type_emb(transform_ids) + self.transform_pos_emb(pos)

    def _cross_attention(self,
                         query: torch.Tensor,        # [B, S_q, D]
                         key_value: torch.Tensor,    # [B, S_kv, D]
                         q_mask: torch.Tensor,       # [B, S_q]  True=pad  (used only for pooling, not here)
                         ) -> torch.Tensor:          # [B, S_q, D]
        B, S_q, D  = query.shape
        S_kv       = key_value.shape[1]
        H, HD      = self.nhead, self.head_dim

        Q = self.q_proj(query).reshape(B, S_q,  H, HD).transpose(1, 2)   # [B,H,Sq,HD]
        K = self.k_proj(key_value).reshape(B, S_kv, H, HD).transpose(1, 2)
        V = self.v_proj(key_value).reshape(B, S_kv, H, HD).transpose(1, 2)

        # Apply RoPE to Q only (IR side)
        Q, _ = self.rope(Q, Q)

        scale  = math.sqrt(HD)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B,H,Sq,Skv]

        # NOTE: In cross-attention we do NOT mask query rows.
        # Masking query rows (IR padding) makes entire attention rows all-inf → NaN.
        # IR padding tokens are excluded at the pooling stage (step 5 in forward).
        # No key masking needed either: transform pad tokens have zero embeddings
        # (padding_idx in Embedding) so they contribute negligibly to attention.

        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.dropout(attn_w)
        out    = torch.matmul(attn_w, V)               # [B,H,Sq,HD]
        out    = out.transpose(1, 2).reshape(B, S_q, D)
        return self.out_proj(out)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self,
                ir_memory: torch.Tensor,         # [B, S, d_model]  from Transformer encoder
                ir_mask: torch.Tensor,           # [B, S]           True = padding
                transform_ids: torch.Tensor,     # [B, transform_seq_len]  integer type IDs
                ) -> torch.Tensor:               # [B, 1]  predicted log-speedup
        """
        Args:
            ir_memory:     IR token representations from the Transformer encoder.
            ir_mask:       Padding mask for IR tokens (True = ignore).
            transform_ids: Integer type IDs for each step in the transform sequence.

        Returns:
            Predicted log(speedup) of shape [B, 1].
            Caller should apply torch.exp() or F.softplus() depending on convention.
        """
        # 1. Build transformation token sequence  [B, L_t, D]
        trans_tokens = self._encode_transforms(transform_ids)
        trans_tokens = self.attn_norm(trans_tokens)

        # 2. Cross-attention: IR tokens query into transformation tokens
        cross_out = self._cross_attention(
            query=ir_memory,
            key_value=trans_tokens,
            q_mask=ir_mask,
        )                                                # [B, S_q, D]

        # 3. Gated residual: gate controls how much cross-attn info flows through
        gate_input  = torch.cat([ir_memory, cross_out], dim=-1)   # [B, S, 2D]
        gate_weight = torch.sigmoid(self.gate(gate_input))         # [B, S, D]
        fused       = gate_weight * cross_out + (1 - gate_weight) * ir_memory

        # 4. FFN with pre-norm
        fused = fused + self.ffn(self.ffn_norm(fused))

        # 5. Pool over valid IR tokens
        valid = (~ir_mask).unsqueeze(-1).float()        # [B, S, 1]
        pooled = (fused * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)

        # 6. Predict
        pooled = self.head_norm(pooled)
        return self.prediction_head(pooled)             # [B, 1]


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, S, D = 4, 50, 256
    L_t = 8

    fusion = CrossAttentionFusion(d_model=D, nhead=8, transform_seq_len=L_t)
    total  = sum(p.numel() for p in fusion.parameters())
    print(f"CrossAttentionFusion parameters: {total:,}")

    ir_mem  = torch.randn(B, S, D)
    ir_mask = torch.zeros(B, S, dtype=torch.bool)
    ir_mask[:, 40:] = True  # last 10 tokens are padding
    t_ids   = torch.randint(0, 15, (B, L_t))

    out = fusion(ir_mem, ir_mask, t_ids)
    print(f"Output shape:    {out.shape}")   # [4, 1]
    assert out.shape == (B, 1)
    print("CrossAttentionFusion self-test PASSED")
