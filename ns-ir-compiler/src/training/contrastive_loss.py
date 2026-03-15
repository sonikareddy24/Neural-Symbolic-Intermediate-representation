"""
src/training/contrastive_loss.py
---------------------------------
NT-Xent (Normalized Temperature-scaled Cross Entropy) contrastive loss
for IR program embeddings — inspired by SimCLR (Chen et al., 2020).

Objective: Pull together embeddings of programs with SIMILAR predicted speedup;
push apart embeddings of programs with DIFFERENT speedup.

Integration with the training loop:
    total_loss = huber_loss + lambda_c * contrastive_loss

This auxiliary objective encourages the IR encoder to produce a semantically
structured embedding space — programs with similar optimization potential cluster
together — which improves generalisation and extrapolation to unseen programs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy loss (NT-Xent).

    Given a batch of B program embeddings and their speedup labels, we treat
    pairs within a user-configurable speedup margin as positive pairs and all
    others as negatives.

    Args:
        temperature:  Scaling factor for logits (default 0.07 following SimCLR).
        speedup_margin: Two samples are 'positive' if |log(s_i) - log(s_j)| < margin.
    """

    def __init__(self, temperature: float = 0.07, speedup_margin: float = 0.15):
        super().__init__()
        self.temperature     = temperature
        self.speedup_margin  = speedup_margin

    def forward(self,
                embeddings: torch.Tensor,    # [B, d_model] L2-normalised IR embeddings
                log_speedups: torch.Tensor,  # [B] or [B,1]  ground-truth log-speedups
                ) -> torch.Tensor:
        """
        Args:
            embeddings:   Program-level IR embeddings (NOT normalised yet).
            log_speedups: Log-space speedup labels for the batch.

        Returns:
            Scalar contrastive loss.
        """
        B = embeddings.shape[0]
        if B < 2:
            return embeddings.new_zeros(1).squeeze()

        # L2-normalise embeddings
        z = F.normalize(embeddings, dim=-1)       # [B, D]

        # Pairwise cosine similarity matrix, scaled by temperature
        sim = torch.matmul(z, z.T) / self.temperature   # [B, B]

        # Build positive mask:  |log_s_i - log_s_j| < margin  (and i != j)
        ls = log_speedups.view(B)
        diff = (ls.unsqueeze(0) - ls.unsqueeze(1)).abs()         # [B, B]
        pos_mask = (diff < self.speedup_margin)                   # [B, B]
        # Remove self-pairs
        pos_mask.fill_diagonal_(False)

        # NT-Xent:  for each anchor i, positives = pos_mask[i], negatives = rest
        # If no positive exists for a row, skip it (avoid NaN)
        has_positive = pos_mask.any(dim=1)
        if not has_positive.any():
            return embeddings.new_zeros(1).squeeze()

        # Mask out self-similarity from denominator
        self_mask = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        sim_no_self = sim.masked_fill(self_mask, float("-inf"))

        # Log-softmax over all non-self similarities
        log_prob = F.log_softmax(sim_no_self, dim=-1)             # [B, B]

        # For each anchor: mean log-prob over positives, guarded against 0 positives
        n_pos = pos_mask.float().sum(dim=1)                        # [B]
        has_pos = n_pos > 0                                        # [B] bool
        if not has_pos.any():
            return embeddings.new_zeros(1).squeeze()

        pos_log_prob = torch.zeros(B, device=embeddings.device)
        pos_log_prob[has_pos] = (
            (log_prob * pos_mask.float()).sum(dim=1)[has_pos]
            / n_pos[has_pos]
        )

        # Only average over anchors that have at least one positive
        loss = -pos_log_prob[has_pos].mean()
        return loss


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loss_fn = NTXentLoss(temperature=0.07, speedup_margin=0.15)
    B, D = 8, 256
    emb    = torch.randn(B, D)
    # Create speedup labels where first 4 are similar, last 4 are different
    labels = torch.cat([torch.ones(4) * 0.5, torch.ones(4) * 2.0])
    loss   = loss_fn(emb, labels)
    print(f"NT-Xent loss: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss must be non-negative"
    print("NTXentLoss self-test PASSED")
