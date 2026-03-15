"""
src/ns_ir/instruction_tokenizer.py
-----------------------------------
Tokenizes LLVM IR instructions into discrete token IDs for lookup-table
embedding, similar to BPE/WordPiece used in NLP transformer models.

Instead of using random hashes (the old broken approach), this tokenizer
builds a real vocabulary from the instruction corpus and assigns
deterministic, consistent IDs to every token.
"""

import re
import json
from collections import Counter
from typing import Dict, List, Optional


class InstructionTokenizer:
    """
    Tokenizes LLVM IR instructions into integer token IDs.

    Canonical normalization removes operand-specific information (register
    names, numeric literals) so semantically similar instructions share token
    IDs, enabling the downstream embedding layer to learn generalizable
    representations.

    Example:
        "%a = add i32 %x, %y"   → ["add", "i32", "<REG>", "<REG>"]
        "%b = add i32 %p, %q"   → ["add", "i32", "<REG>", "<REG>"]  (same!)
        "%c = mul f64 %x, %y"   → ["mul", "f64", "<REG>", "<REG>"]
        "%d = load i32, i32* %p"→ ["load", "i32", "i32", "<REG>"]
    """

    # Reserved special tokens
    SPECIAL_TOKENS: Dict[str, int] = {
        "<PAD>":   0,   # padding placeholder
        "<UNK>":   1,   # out-of-vocabulary token
        "<REG>":   2,   # generic register (%var)
        "<CONST>": 3,   # numeric constant (123, 3.14)
        "<LABEL>": 4,   # basic-block label reference
    }

    # LLVM opcodes we care about most
    KNOWN_OPCODES = [
        "add", "sub", "mul", "sdiv", "udiv", "srem", "urem",
        "fadd", "fsub", "fmul", "fdiv", "frem",
        "and", "or", "xor", "shl", "lshr", "ashr",
        "load", "store", "alloca", "getelementptr",
        "br", "ret", "switch", "call", "invoke", "unreachable",
        "icmp", "fcmp", "select", "phi",
        "sext", "zext", "trunc", "bitcast", "ptrtoint", "inttoptr",
        "extractelement", "insertelement", "shufflevector",
    ]

    # Common LLVM IR types
    KNOWN_TYPES = [
        "i1", "i8", "i16", "i32", "i64", "i128",
        "f32", "f64", "f128", "half",
        "void", "ptr", "null",
    ]

    def __init__(self, vocab_size: int = 10_000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}

        # Pre-register all known opcodes and types (they get stable low IDs)
        for token in self.KNOWN_OPCODES + self.KNOWN_TYPES:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    # ── Vocabulary building ──────────────────────────────────────────────────

    def build_vocab(self, instructions: List[str]) -> None:
        """
        Build vocabulary from an instruction corpus.

        Call this once on the training set before calling encode().

        Args:
            instructions: List of raw LLVM IR instruction strings.
        """
        counter: Counter = Counter()
        for instr in instructions:
            tokens = self._tokenize(instr)
            counter.update(tokens)

        budget = self.vocab_size - len(self.token_to_id)
        for token, _ in counter.most_common(budget):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        print(f"[InstructionTokenizer] Vocabulary built: {len(self.token_to_id)} tokens")

    # ── Tokenisation ─────────────────────────────────────────────────────────

    def _tokenize(self, instruction: str) -> List[str]:
        """
        Normalize and split a single IR instruction into tokens.

        Normalization removes operand-specific noise so structurally
        equivalent instructions produce the same token stream.
        """
        s = instruction.strip().lower()

        # Strip result assignment prefix (e.g. "%a = ") so we focus on the op
        s = re.sub(r"^%[a-z0-9_.]+\s*=\s*", "", s)

        # Replace register operands with generic placeholder
        s = re.sub(r"%[a-z0-9_.]+", "<REG>", s)

        # Replace floating-point constants (must come before integer below)
        s = re.sub(r"\b\d+\.\d+\b", "<CONST>", s)

        # Replace integer constants
        s = re.sub(r"\b\d+\b", "<CONST>", s)

        # Replace label references
        s = re.sub(r"label\s+<REG>", "<LABEL>", s)

        # Remove punctuation clutter (commas, brackets, asterisks)
        s = re.sub(r"[,\[\]\(\)\*!]", " ", s)

        tokens = [t for t in s.split() if t]
        return tokens

    def encode(self, instruction: str, max_length: int = 32) -> List[int]:
        """
        Convert one instruction string into a fixed-length list of token IDs.

        Sequences shorter than max_length are right-padded with <PAD> (0).
        Sequences longer are truncated.

        Args:
            instruction: Raw LLVM IR instruction string.
            max_length:  Output sequence length.

        Returns:
            List of token IDs, length == max_length.
        """
        tokens = self._tokenize(instruction)
        unk_id = self.SPECIAL_TOKENS["<UNK>"]
        pad_id = self.SPECIAL_TOKENS["<PAD>"]

        ids = [self.token_to_id.get(t, unk_id) for t in tokens]

        # Pad or truncate
        if len(ids) < max_length:
            ids += [pad_id] * (max_length - len(ids))
        else:
            ids = ids[:max_length]

        return ids

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist vocabulary to a JSON file so it can be reloaded."""
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "token_to_id": self.token_to_id}, f, indent=2)
        print(f"[InstructionTokenizer] Vocabulary saved: {path}")

    def load(self, path: str) -> None:
        """Load a previously saved vocabulary."""
        with open(path) as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        print(f"[InstructionTokenizer] Vocabulary loaded: {len(self.token_to_id)} tokens")

    def __len__(self) -> int:
        return len(self.token_to_id)
