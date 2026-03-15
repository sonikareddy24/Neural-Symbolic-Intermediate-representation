"""
src/ns_ir/learned_embeddings.py
---------------------------------
Learned embedding module that replaces the previous broken approach of
hashing instruction strings into random vectors.

Architecture:
    LLVM IR instruction string
        → InstructionTokenizer (normalise + tokenise → IDs)
        → nn.Embedding (vocab lookup table, trained end-to-end)
        → Positional Encoding (1-D, for token order in instruction)
        → Average Pool over non-padding tokens
        → [embedding_dim] vector

This module is a drop-in replacement for EmbeddingGenerator and is designed
to be trained jointly with the Transformer cost model via a shared optimizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from src.ns_ir.instruction_tokenizer import InstructionTokenizer


class LearnedEmbeddingGenerator(nn.Module):
    """
    Neural instruction-to-vector encoder.

    Semantically equivalent instructions (same opcode, same type, different
    register names) map to nearly identical embeddings after training, enabling
    proper generalisation across programs.

    Usage:
        emb_gen = LearnedEmbeddingGenerator(vocab_size=10000, embedding_dim=128)
        emb_gen.tokenizer.build_vocab(my_instruction_list)

        # During model forward pass:
        vec = emb_gen("%a = add i32 %x, %y")  # shape: [128]
    """

    def __init__(self, vocab_size: int = 10_000, embedding_dim: int = 128,
                 max_instr_tokens: int = 32):
        super().__init__()
        self.embedding_dim     = embedding_dim
        self.max_instr_tokens  = max_instr_tokens
        self.tokenizer         = InstructionTokenizer(vocab_size)

        # Learnable token embedding table
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,          # <PAD> does not contribute to gradients
        )

        # Learnable positional encoding (token position within an instruction)
        self.pos_embedding = nn.Embedding(max_instr_tokens, embedding_dim)

        # Small projection head to allow non-linear mixing of token embeddings
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight,   mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    # ── Core forward ─────────────────────────────────────────────────────────

    def forward(self, instruction: str) -> torch.Tensor:
        """
        Encode one LLVM IR instruction string to a [embedding_dim] vector.

        Args:
            instruction: Raw LLVM IR instruction string.

        Returns:
            Tensor of shape [embedding_dim].
        """
        # Tokenise → list of int IDs
        token_ids = self.tokenizer.encode(instruction, max_length=self.max_instr_tokens)
        ids_t     = torch.tensor(token_ids, dtype=torch.long)           # [L]
        pos_t     = torch.arange(len(ids_t), dtype=torch.long)         # [L]

        # Embed tokens + positions
        tok_emb = self.token_embedding(ids_t)                           # [L, D]
        pos_emb = self.pos_embedding(pos_t)                             # [L, D]
        combined = tok_emb + pos_emb                                    # [L, D]

        # Masked average pool (ignore <PAD> tokens)
        pad_id = self.tokenizer.SPECIAL_TOKENS["<PAD>"]
        mask   = (ids_t != pad_id).float().unsqueeze(-1)               # [L, 1]
        pooled = (combined * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)

        # Non-linear projection
        return self.proj(pooled)                                        # [D]

    def batch_encode(self, instructions: List[str]) -> torch.Tensor:
        """
        Encode a batch of instructions efficiently.

        Args:
            instructions: List of N instruction strings.

        Returns:
            Tensor of shape [N, embedding_dim].
        """
        all_ids = torch.tensor(
            [self.tokenizer.encode(ins, max_length=self.max_instr_tokens) for ins in instructions],
            dtype=torch.long,
        )                                                               # [N, L]

        L   = all_ids.shape[1]
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)          # [1, L]

        tok_emb  = self.token_embedding(all_ids)                       # [N, L, D]
        pos_emb  = self.pos_embedding(pos)                             # [1, L, D]
        combined = tok_emb + pos_emb                                   # [N, L, D]

        pad_id = self.tokenizer.SPECIAL_TOKENS["<PAD>"]
        mask   = (all_ids != pad_id).float().unsqueeze(-1)            # [N, L, 1]
        pooled = (combined * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.proj(pooled)                                       # [N, D]

    # ── Vocabulary building helper ────────────────────────────────────────────

    def build_vocab_from_programs(self, program_nodes: List[dict]) -> None:
        """
        Build the tokenizer vocabulary from a list of program node dicts.

        Args:
            program_nodes: List of {'instruction': str, ...} dicts from NSIRGraph.
        """
        instructions = [n.get("symbolic_ir", n.get("instruction", "")) for n in program_nodes]
        self.tokenizer.build_vocab(instructions)

    def save_vocab(self, path: str) -> None:
        self.tokenizer.save(path)

    def load_vocab(self, path: str) -> None:
        self.tokenizer.load(path)
        # Resize embedding table to match loaded vocab
        new_size = len(self.tokenizer)
        if new_size != self.token_embedding.num_embeddings:
            old_weight = self.token_embedding.weight.data
            self.token_embedding = nn.Embedding(new_size, self.embedding_dim, padding_idx=0)
            n = min(new_size, old_weight.size(0))
            self.token_embedding.weight.data[:n] = old_weight[:n]
