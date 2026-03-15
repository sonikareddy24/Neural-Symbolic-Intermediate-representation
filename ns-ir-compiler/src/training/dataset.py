import os
import torch
import math
from torch.utils.data import Dataset, DataLoader

# Fixed weight vectors so train/val share the SAME latent function (reproducible signal)
torch.manual_seed(42)
_NODE_WEIGHT   = torch.randn(128)   # how each node feature dimension contributes
_TRANS_WEIGHT  = torch.randn(15)    # one weight per transform type (0-14)
N_TRANSFORM_TYPES = 15

def _compute_speedup(seq: torch.Tensor, mask: torch.Tensor,
                     transform_ids: torch.Tensor) -> float:
    """
    Deterministic speedup label derived from node features and integer transform IDs.
    Creates a learnable signal: the model has a real pattern to discover.

    speedup = softplus( node_signal + transform_signal ) * scale
    """
    # Average over valid (non-masked) tokens
    valid = (~mask).float().unsqueeze(-1)           # [50, 1]
    node_sum = (seq * valid).sum(dim=0)             # [128]
    valid_count = valid.sum().clamp(min=1.0)
    node_mean = node_sum / valid_count              # [128]

    node_signal = torch.dot(node_mean, _NODE_WEIGHT) / math.sqrt(128)

    # Sum embedding weights for each transform type present
    trans_one_hot = torch.zeros(N_TRANSFORM_TYPES)
    for tid in transform_ids:
        trans_one_hot[tid.clamp(0, N_TRANSFORM_TYPES - 1)] += 1.0
    trans_signal = torch.dot(trans_one_hot, _TRANS_WEIGHT) / math.sqrt(N_TRANSFORM_TYPES)

    raw = node_signal * 0.6 + trans_signal * 0.4
    # Map to realistic speedup range [0.8, 3.5] using a smooth function
    speedup = 0.8 + 2.7 * torch.sigmoid(raw)
    return speedup.item()



class CompilerDataset(Dataset):
    """
    Synthetic compiler cost-model dataset with REALISTIC correlated labels.

    Pre-generates all samples at construction time into in-memory tensors.
    This makes per-epoch iteration ~50x faster than per-item torch.manual_seed().
    """
    def __init__(self, split='train'):
        n     = 50_000 if split == 'train' else 5_000
        off   = 0      if split == 'train' else 100_000
        rng   = torch.Generator(); rng.manual_seed(off)

        seqs   = torch.zeros(n, 50, 128)
        masks  = torch.ones(n, 50, dtype=torch.bool)
        trans  = torch.full((n, 32), N_TRANSFORM_TYPES - 1, dtype=torch.long)
        labels = torch.zeros(n, 1)

        for i in range(n):
            nn_  = int(torch.randint(8, 48, (1,), generator=rng).item())
            seq  = torch.randn(nn_, 128, generator=rng)
            seqs[i, :nn_]  = seq
            masks[i, :nn_] = False
            nt   = int(torch.randint(1, 9, (1,), generator=rng).item())
            tids = torch.randint(0, N_TRANSFORM_TYPES, (nt,), generator=rng)
            trans[i, :nt]  = tids
            sv   = _compute_speedup(seqs[i], masks[i], tids)
            noise = torch.randn(1, generator=rng).item() * 0.05
            sv    = max(0.5, sv + noise)
            labels[i, 0] = math.log(sv)

        self.seqs, self.masks, self.trans, self.labels = seqs, masks, trans, labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {
            'seq':        self.seqs[idx],
            'mask':       self.masks[idx],
            'transforms': self.trans[idx],
            'speedup':    self.labels[idx],
        }


def get_dataloader(batch_size=64, split='train'):
    ds = CompilerDataset(split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=0,
        pin_memory=False,
    )


class MapeLoss(torch.nn.Module):
    """Mean Absolute Percentage Error (monitored but not optimised directly)."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / target.clamp(min=1e-6))) * 100.0

