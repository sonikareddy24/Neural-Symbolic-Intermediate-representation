"""
src/training/trainer.py  (v2 — State-of-the-Art upgrade)
----------------------------------------------------------
Upgrades vs. v1:
  - 150 default epochs (was 50)
  - CosineAnnealingWarmRestarts instead of linear-warmup + cosine decay
  - Stochastic Weight Averaging (SWA) over final 20% of epochs
  - NT-Xent contrastive auxiliary loss on IR embeddings
  - Training curve saved to JSON for ablation / plotting
  - Detailed per-epoch log with all metrics
"""

import os, sys, math, json, time
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.training.dataset       import get_dataloader, MapeLoss
from src.training.contrastive_loss import NTXentLoss


class NsIrTrainer:
    """
    Orchestrates State-of-the-Art training for the NS-IR cost model.

    Key upgrades vs. v1:
      1. CosineAnnealingWarmRestarts  — escapes local minima via periodic LR spikes
      2. Stochastic Weight Averaging  — flatter minima, better generalisation
      3. NT-Xent contrastive loss     — structured IR embedding space
      4. 150-epoch default            — full convergence with SWA plateau
    """

    def __init__(self, model: nn.Module, model_dir: str = "models/checkpoints/"):
        self.model      = model
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_dir  = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.mape_fn      = MapeLoss()
        self.huber_fn     = nn.HuberLoss(delta=0.5)
        self.ntxent_fn    = NTXentLoss(temperature=0.07, speedup_margin=0.15)

    # ── Training entry point ──────────────────────────────────────────────────

    def train(self,
              epochs: int          = 150,
              lr: float            = 3e-4,
              batch_size: int      = 64,
              patience: int        = 20,
              lambda_contrastive: float = 0.1,
              swa_start_frac: float     = 0.80,  # start SWA at 80% of epochs
              T0: int              = 30,          # cosine restart period
              ) -> float:
        """
        Full training loop.

        Args:
            epochs:             Total training epochs.
            lr:                 Initial learning rate.
            batch_size:         Batch size for train and val loaders.
            patience:           Early-stopping patience (epochs without val improvement).
            lambda_contrastive: Weight of NT-Xent loss in total loss.
            swa_start_frac:     Fraction of total epochs after which SWA is activated.
            T0:                 Period of first cosine restart (CosineAnnealingWarmRestarts).

        Returns:
            Best validation MAPE achieved during training.
        """
        train_loader = get_dataloader(batch_size=batch_size, split="train")
        val_loader   = get_dataloader(batch_size=batch_size, split="val")

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=1e-4, betas=(0.9, 0.999),
        )

        # CosineAnnealingWarmRestarts: LR spikes every T0 epochs (T0, 2*T0, 4*T0, …)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2,
                                               eta_min=1e-6)

        # SWA model averages weights during the final 20% of training
        swa_start     = int(epochs * swa_start_frac)
        swa_model     = AveragedModel(self.model)
        swa_scheduler = SWALR(optimizer, swa_lr=5e-5, anneal_epochs=10)
        swa_active    = False

        best_val_mape    = float("inf")
        prev_val_mape    = float("inf")
        patience_counter = 0
        history          = []   # list of per-epoch metric dicts

        print(f"\n{'='*72}")
        print(f"  NS-IR  State-of-the-Art Training")
        print(f"  Device: {self.device}  |  Epochs: {epochs}  |  LR: {lr}")
        print(f"  Dataset: 50K train / 5K val  |  Batch: {batch_size}")
        print(f"  Contrastive lambda: {lambda_contrastive}  |  SWA start: epoch {swa_start}")
        print(f"  Cosine restarts: T0={T0}")
        print(f"{'='*72}\n")

        for epoch in range(epochs):
            t0 = time.time()

            # ── Activate SWA regime ───────────────────────────────────────────
            if epoch == swa_start and not swa_active:
                swa_active = True
                print(f"  [SWA] Stochastic Weight Averaging activated at epoch {epoch+1}")

            train_metrics = self._train_epoch(
                train_loader, optimizer,
                lambda_contrastive=lambda_contrastive,
            )

            # ── LR scheduling ─────────────────────────────────────────────────
            if swa_active:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                current_lr = swa_scheduler.get_last_lr()[0]
            else:
                scheduler.step(epoch + 1)
                current_lr = optimizer.param_groups[0]["lr"]

            val_mape, val_loss = self._validate(val_loader)
            delta              = prev_val_mape - val_mape
            prev_val_mape      = val_mape
            elapsed            = time.time() - t0

            row = {
                "epoch":          epoch + 1,
                "train_huber":    train_metrics["huber"],
                "train_mape":     train_metrics["mape"],
                "train_contrast": train_metrics["contrastive"],
                "val_huber":      val_loss,
                "val_mape":       val_mape,
                "lr":             current_lr,
                "elapsed_s":      round(elapsed, 1),
            }
            history.append(row)

            print(
                f"Epoch {epoch+1:03d}/{epochs} | "
                f"Huber {train_metrics['huber']:.4f} | "
                f"Contrast {train_metrics['contrastive']:.4f} | "
                f"TrainMAPE {train_metrics['mape']:.2f}% | "
                f"ValMAPE {val_mape:.2f}% ({'+' if delta<0 else ''}{-delta:.2f}%) | "
                f"LR {current_lr:.2e} | {elapsed:.1f}s"
                + (" [SWA]" if swa_active else "")
            )

            # ── Checkpoint best model ─────────────────────────────────────────
            if val_mape < best_val_mape:
                best_val_mape    = val_mape
                patience_counter = 0
                ckpt = os.path.join(self.model_dir, "best_model.pt")
                torch.save(self.model.state_dict(), ckpt)
                print(f"    Best model saved -> {ckpt}  (val MAPE {best_val_mape:.2f}%)")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs.")
                break

        # ── Finalise SWA model ────────────────────────────────────────────────
        if swa_active:
            print("\n  Updating SWA BatchNorm statistics…")
            update_bn(train_loader, swa_model, device=self.device)
            swa_ckpt = os.path.join(self.model_dir, "swa_model.pt")
            torch.save(swa_model.module.state_dict(), swa_ckpt)
            print(f"  SWA model saved -> {swa_ckpt}")

        # ── Save training curve ───────────────────────────────────────────────
        curve_path = os.path.join(self.model_dir, "training_curve.json")
        with open(curve_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n{'='*72}")
        print(f"  Training Complete  |  Best Val MAPE: {best_val_mape:.2f}%")
        print(f"  Training curve saved -> {curve_path}")
        print(f"{'='*72}\n")
        return best_val_mape

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _train_epoch(self, loader, optimizer, lambda_contrastive: float):
        self.model.train()
        total_huber, total_mape, total_contrast, n = 0., 0., 0., 0

        for batch in loader:
            seq        = batch["seq"].to(self.device)
            mask       = batch["mask"].to(self.device)
            transforms = batch["transforms"].to(self.device)
            targets    = batch["speedup"].to(self.device)

            # Convert float transform_seq → integer IDs for new model API
            t_ids = transforms.long().clamp(0, 14)

            optimizer.zero_grad()

            log_pred    = self.model(seq, mask, t_ids).squeeze(-1)
            log_target  = targets.squeeze(-1)

            huber_loss  = self.huber_fn(log_pred, log_target)

            # Contrastive loss on IR embeddings
            ir_emb      = self.model.get_ir_embedding(seq, mask)       # [B, D]
            contrast    = self.ntxent_fn(ir_emb, log_target)

            loss = huber_loss + lambda_contrastive * contrast

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                mape = self.mape_fn(
                    torch.exp(log_pred), torch.exp(log_target)
                ).item()

            total_huber    += huber_loss.item()
            total_contrast += contrast.item()
            total_mape     += mape
            n              += 1

        return {
            "huber":       total_huber    / max(n, 1),
            "mape":        total_mape     / max(n, 1),
            "contrastive": total_contrast / max(n, 1),
        }

    def _validate(self, loader):
        self.model.eval()
        total_mape, total_loss, n = 0., 0., 0
        with torch.no_grad():
            for batch in loader:
                seq        = batch["seq"].to(self.device)
                mask       = batch["mask"].to(self.device)
                transforms = batch["transforms"].to(self.device)
                targets    = batch["speedup"].to(self.device)

                t_ids      = transforms.long().clamp(0, 14)
                log_pred   = self.model(seq, mask, t_ids).squeeze(-1)
                log_target = targets.squeeze(-1)

                total_loss += self.huber_fn(log_pred, log_target).item()
                total_mape += self.mape_fn(
                    torch.exp(log_pred), torch.exp(log_target)
                ).item()
                n += 1
        return total_mape / max(n, 1), total_loss / max(n, 1)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from src.models.transformer_cost_model import TransformerCostModel

    ap = argparse.ArgumentParser(description="Train NS-IR cost model (SOA v2)")
    ap.add_argument("--epochs",        type=int,   default=150)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--batch-size",    type=int,   default=64)
    ap.add_argument("--lambda-c",      type=float, default=0.1,
                    help="Contrastive loss weight")
    ap.add_argument("--swa-start",     type=float, default=0.80,
                    help="SWA start fraction of total epochs")
    ap.add_argument("--T0",            type=int,   default=30,
                    help="CosineAnnealingWarmRestarts period")
    ap.add_argument("--model-dir",     type=str,   default=None)
    args = ap.parse_args()

    if args.model_dir is None:
        root      = Path(__file__).resolve().parent.parent.parent
        model_dir = str(root / "models" / "checkpoints")
    else:
        model_dir = args.model_dir

    model  = TransformerCostModel()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    trainer = NsIrTrainer(model, model_dir=model_dir)
    trainer.train(
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size,
        lambda_contrastive=args.lambda_c,
        swa_start_frac=args.swa_start,
        T0=args.T0,
    )
