"""
tests/test_model_quality.py
---------------------------
Automated quality checks for the NS-IR Transformer cost model.
Run from the project root:

    cd "/Users/apple/Desktop/compiler design /ns-ir-compiler"
    python3 tests/test_model_quality.py
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer_cost_model import TransformerCostModel
from src.training.dataset import get_dataloader, MapeLoss

PASS = "✅ PASSED"
FAIL = "❌ FAILED"

# ─────────────────────────────────────────────────────────────
# Test 1 — Shape & Positivity Sanity Check
# ─────────────────────────────────────────────────────────────
def test_output_shape_and_positivity():
    print("\n[1/3] Shape & positivity check...")
    model = TransformerCostModel()
    model.eval()

    B = 8
    seq        = torch.randn(B, 50, 128)
    mask       = torch.ones(B, 50, dtype=torch.bool)
    mask[:, :20] = False           # first 20 tokens valid
    transforms = torch.randn(B, 32)

    with torch.no_grad():
        out = model(seq, mask, transforms)

    assert out.shape == (B, 1),    f"Expected shape ({B}, 1), got {out.shape}"
    assert (out > 0).all(),        f"All outputs must be positive. Got: {out}"
    assert not torch.isnan(out).any(), "NaN detected in output!"
    print(f"       Output shape : {out.shape}      {PASS}")
    print(f"       All positive : {(out > 0).all().item()}   {PASS}")

# ─────────────────────────────────────────────────────────────
# Test 2 — Single-Batch Overfit Test
# Model MUST be able to memorise 8 samples (proves learning works)
# ─────────────────────────────────────────────────────────────
def test_single_batch_overfit():
    print("\n[2/3] Single-batch overfit test (100 steps)...")
    model = TransformerCostModel()
    model.train()

    torch.manual_seed(0)
    seq        = torch.randn(8, 50, 128)
    mask       = torch.ones(8, 50, dtype=torch.bool)
    mask[:, :15] = False
    transforms = torch.randn(8, 32)
    targets    = torch.tensor([1.2, 1.8, 2.5, 1.0, 3.0, 1.5, 2.2, 1.1])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.HuberLoss()

    initial_loss = None
    for step in range(200):
        preds = model(seq, mask, transforms).squeeze(-1)
        loss  = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    improved   = final_loss < initial_loss * 0.5   # must halve the loss

    print(f"       Initial loss : {initial_loss:.4f}")
    print(f"       Final loss   : {final_loss:.4f}")
    print(f"       Improved >50%: {improved}   {PASS if improved else FAIL}")
    assert improved, f"Model failed to overfit single batch! Loss: {initial_loss:.4f} → {final_loss:.4f}"

# ─────────────────────────────────────────────────────────────
# Test 3 — MAPE Target After Short Training
# After 5 quick epochs on the improved dataset, Val MAPE must be < 25%
# (Full 50-epoch run targets < 10%)
# ─────────────────────────────────────────────────────────────
def test_mape_improves_with_training():
    print("\n[3/3] MAPE improvement test (5 training epochs)...")
    model      = TransformerCostModel()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    huber      = nn.HuberLoss(delta=0.5)
    mape_fn    = MapeLoss()

    train_loader = get_dataloader(batch_size=64, split='train')
    val_loader   = get_dataloader(batch_size=64, split='val')

    def validate():
        """Compute MAPE on the REAL (exponentiated) speedup scale, not log-space."""
        model.eval()
        total = 0.0
        with torch.no_grad():
            for b in val_loader:
                # Model predicts log(speedup); exponentiate both pred and target
                # to get real speedup values before computing percentage error.
                log_pred   = model(b['seq'], b['mask'], b['transforms']).squeeze(-1)
                log_target = b['speedup'].squeeze(-1)     # stored as log(speedup)

                real_pred   = torch.exp(log_pred)         # ← KEY FIX: real space
                real_target = torch.exp(log_target)       # ← KEY FIX: real space

                total += mape_fn(real_pred, real_target).item()
        return total / len(val_loader)

    initial_mape = validate()
    print(f"       Initial Val MAPE : {initial_mape:.2f}%")

    for epoch in range(5):
        model.train()
        for b in train_loader:
            log_pred   = model(b['seq'], b['mask'], b['transforms']).squeeze(-1)
            log_target = b['speedup'].squeeze(-1)

            # Huber loss trains in the stable log-space
            loss = huber(log_pred, log_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    final_mape = validate()
    improved   = final_mape < initial_mape
    target_ok  = final_mape < 25.0

    print(f"       Final Val MAPE   : {final_mape:.2f}%")
    print(f"       MAPE improved    : {improved}   {PASS if improved else FAIL}")
    print(f"       MAPE < 25%       : {target_ok}   {PASS if target_ok else FAIL}")
    assert improved,  "Val MAPE did not improve after 5 epochs!"
    assert target_ok, f"Val MAPE {final_mape:.2f}% still above 25% after 5 epochs!"


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  NS-IR Compiler — Model Quality Test Suite")
    print("=" * 55)

    try:
        test_output_shape_and_positivity()
        test_single_batch_overfit()
        test_mape_improves_with_training()

        print("\n" + "=" * 55)
        print("  ALL TESTS PASSED ✅")
        print("=" * 55)

    except AssertionError as e:
        print(f"\n  TEST FAILED ❌: {e}")
        sys.exit(1)
