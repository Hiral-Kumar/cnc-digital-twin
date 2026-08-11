"""
trainer.py — Training Loop for PC-NDT

Handles:
- Physics constraint warm-up schedule (epochs 0-20: λ=0, 20-50: linear ramp)
- Gradient clipping
- Early stopping on val_rmse
- Checkpoint saving (best model only)
- Per-epoch logging of all loss components
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config: dict, device: str = 'cpu'):
        self.model  = model.to(device)
        self.config = config
        self.device = device

        cfg_train  = config['training']
        cfg_phys   = config['physics']

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg_train['learning_rate'],
            weight_decay=cfg_train['weight_decay'],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg_train['epochs']
        )

        self.max_epochs     = cfg_train['epochs']
        self.grad_clip      = cfg_train['grad_clip_max_norm']
        self.patience       = cfg_train['early_stopping_patience']
        self.checkpoint_dir = cfg_train['checkpoint_dir']
        self.log_every      = cfg_train['log_every_n_steps']

        self.warmup_epochs  = cfg_phys['warmup_epochs']
        self.rampup_epochs  = cfg_phys['rampup_epochs']
        self.lambda_grid    = cfg_phys['lambda_grid']

        # Final lambda values (set after warmup by auto-scaling)
        self.lambda_archard = 0.0
        self.lambda_paris   = 0.0
        self.lambda_fourier = 0.0
        self._lambdas_set   = False

        self.best_val_rmse  = float('inf')
        self.no_improve     = 0
        self.history        = []

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_lambda(self, epoch: int) -> tuple:
        """Physics loss weights follow warm-up → ramp → full schedule."""
        if epoch < self.warmup_epochs:
            return 0.0, 0.0, 0.0
        ramp_progress = min(
            1.0,
            (epoch - self.warmup_epochs) / max(self.rampup_epochs, 1)
        )
        return (
            ramp_progress * self.lambda_archard,
            ramp_progress * self.lambda_paris,
            ramp_progress * self.lambda_fourier,
        )

    def _auto_set_lambdas(self, physics_losses: dict, pred_loss: float):
        """
        At the end of warmup, set lambda values so that each physics term
        contributes ~10% of the prediction loss.
        """
        for key, target_attr in [
            ('archard', 'lambda_archard'),
            ('paris',   'lambda_paris'),
            ('fourier', 'lambda_fourier'),
        ]:
            phys_val = physics_losses[key].item()
            if phys_val > 1e-10:
                lam = 0.1 * pred_loss / phys_val
                # Clip to the grid range
                lam = float(np.clip(lam, self.lambda_grid[0], self.lambda_grid[-1]))
            else:
                lam = self.lambda_grid[0]
            setattr(self, target_attr, lam)

        logger.info(
            f"  Lambda auto-set: "
            f"Archard={self.lambda_archard:.4f}, "
            f"Paris={self.lambda_paris:.4f}, "
            f"Fourier={self.lambda_fourier:.4f}"
        )
        self._lambdas_set = True

    def train_epoch(self, train_loader: DataLoader,
                    epoch: int, physics_module) -> dict:
        self.model.train()
        losses = {'pred': [], 'archard': [], 'paris': [], 'fourier': [], 'total': []}
        la, lp, lf = self._get_lambda(epoch)
        criterion = nn.MSELoss()

        for step, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            out    = self.model(X_batch)
            rul    = out['rul']
            dh_dt  = out['dh_dt']
            h      = out['h_final']
            A      = out['adjacency']

            l_pred = criterion(rul, y_batch)

            phys = physics_module.compute_all(
                predicted_derivatives=dh_dt,
                hidden_state=h,
                adjacency=A,
                features=X_batch[:, -1, :, :],   # last timestep features [B,N,F]
                lambda_archard=la,
                lambda_paris=lp,
                lambda_fourier=lf,
            )

            # Auto-set lambdas at end of warmup (once)
            if epoch == self.warmup_epochs and not self._lambdas_set:
                self._auto_set_lambdas(phys, l_pred.item())
                la, lp, lf = self._get_lambda(epoch)

            total_loss = l_pred + phys['total']
            total_loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            losses['pred'].append(l_pred.item())
            losses['archard'].append(phys['archard'].item())
            losses['paris'].append(phys['paris'].item())
            losses['fourier'].append(phys['fourier'].item())
            losses['total'].append(total_loss.item())

            if step % self.log_every == 0:
                logger.debug(
                    f"  E{epoch} step {step}: "
                    f"pred={l_pred.item():.5f} "
                    f"total={total_loss.item():.5f}"
                )

        return {k: float(np.mean(v)) for k, v in losses.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        preds, targets = [], []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            out = self.model(X_batch)
            preds.append(out['rul'].cpu().numpy())
            targets.append(y_batch.numpy())
        preds   = np.concatenate(preds)
        targets = np.concatenate(targets)
        return float(np.sqrt(np.mean((preds - targets) ** 2)))

    def fit(self, train_loader: DataLoader,
            val_loader: DataLoader,
            physics_module) -> list:
        """Main training loop. Returns per-epoch history."""
        logger.info(f"Starting training for {self.max_epochs} epochs")

        for epoch in range(self.max_epochs):
            train_metrics = self.train_epoch(train_loader, epoch, physics_module)
            val_rmse      = self.evaluate(val_loader)
            self.scheduler.step()

            epoch_log = {
                'epoch':    epoch,
                'val_rmse': val_rmse,
                **train_metrics,
            }
            self.history.append(epoch_log)

            logger.info(
                f"Epoch {epoch:03d} | "
                f"train_total={train_metrics['total']:.5f} | "
                f"val_rmse={val_rmse:.5f}"
            )

            # Early stopping + checkpointing
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.no_improve    = 0
                ckpt_path = os.path.join(
                    self.checkpoint_dir, 'best_model.pt'
                )
                torch.save({
                    'epoch':       epoch,
                    'model_state': self.model.state_dict(),
                    'val_rmse':    val_rmse,
                    'history':     self.history,
                }, ckpt_path)
                logger.info(f"  ✓ New best val_rmse={val_rmse:.5f} — saved")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break

        return self.history
