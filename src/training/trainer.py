"""
trainer.py — Training Loop for PC-NDT (V3: anti-collapse loss tracking)

Builds on the V2 fix (early-stop/warmup race condition). Adds tracking
and logging of the new anti-collapse regularizer loss term introduced
in constraints.py to close the degenerate dh_dt-collapse solution.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config: dict, device: str = 'cpu', normalizer=None):
        self.model  = model.to(device)
        self.config = config
        self.device = device
        self.normalizer = normalizer

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

        self.lambda_archard = 0.0
        self.lambda_paris   = 0.0
        self.lambda_fourier = 0.0
        self._lambdas_set   = False

        self.best_val_rmse  = float('inf')
        self.no_improve     = 0
        self.history        = []

        self.min_epochs_before_stopping = self.warmup_epochs + self.rampup_epochs
        logger.info(
            f"  Early stopping will not fire before epoch "
            f"{self.min_epochs_before_stopping} "
            f"(warmup={self.warmup_epochs} + rampup={self.rampup_epochs})"
        )

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_lambda(self, epoch: int) -> tuple:
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
        for key, target_attr in [
            ('archard', 'lambda_archard'),
            ('paris',   'lambda_paris'),
            ('fourier', 'lambda_fourier'),
        ]:
            phys_val = physics_losses[key].item()
            if phys_val > 1e-10:
                lam = 0.1 * pred_loss / phys_val
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
        losses = {'pred': [], 'archard': [], 'paris': [], 'fourier': [],
                  'anticollapse': [], 'total': []}
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
                features=X_batch[:, -1, :, :],
                lambda_archard=la,
                lambda_paris=lp,
                lambda_fourier=lf,
            )

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
            losses['anticollapse'].append(phys['anticollapse'].item())
            losses['total'].append(total_loss.item())

            if step % self.log_every == 0:
                logger.debug(
                    f"  E{epoch} step {step}: "
                    f"pred={l_pred.item():.5f} "
                    f"anticollapse={phys['anticollapse'].item():.5f} "
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

    @torch.no_grad()
    def measure_activity(self, loader: DataLoader) -> float:
        """
        Diagnostic: measure average dh_dt magnitude across a loader.
        Used to confirm the anti-collapse fix is working — this value
        should stay meaningfully above min_activity_threshold throughout
        training, unlike the previous run where it silently collapsed.
        """
        self.model.eval()
        activities = []
        for X_batch, _ in loader:
            X_batch = X_batch.to(self.device)
            out = self.model(X_batch)
            activities.append(out['dh_dt'].norm(dim=-1).mean().item())
        return float(np.mean(activities))

    def _build_normalizer_state(self):
        if self.normalizer is None:
            logger.warning("  No normalizer was passed to Trainer — checkpoint will NOT include normalizer state.")
            return None
        if not getattr(self.normalizer, '_fitted', False):
            logger.warning("  Normalizer exists but is not fitted — skipping save")
            return None
        return {
            'feat_min': self.normalizer.feat_min,
            'feat_max': self.normalizer.feat_max,
        }

    def fit(self, train_loader: DataLoader,
            val_loader: DataLoader,
            physics_module) -> list:
        logger.info(f"Starting training for {self.max_epochs} epochs")

        for epoch in range(self.max_epochs):
            train_metrics = self.train_epoch(train_loader, epoch, physics_module)
            val_rmse      = self.evaluate(val_loader)
            activity      = self.measure_activity(val_loader)
            self.scheduler.step()

            epoch_log = {
                'epoch':    epoch,
                'val_rmse': val_rmse,
                'dh_dt_activity': activity,
                **train_metrics,
            }
            self.history.append(epoch_log)

            la, lp, lf = self._get_lambda(epoch)
            logger.info(
                f"Epoch {epoch:03d} | "
                f"train_total={train_metrics['total']:.5f} | "
                f"val_rmse={val_rmse:.5f} | "
                f"dh_dt_activity={activity:.5f} | "
                f"anticollapse={train_metrics['anticollapse']:.5f} | "
                f"lambda=({la:.3f},{lp:.3f},{lf:.3f})"
            )

            can_stop_early = epoch >= self.min_epochs_before_stopping

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.no_improve    = 0
                ckpt_path = os.path.join(self.checkpoint_dir, 'best_model.pt')

                torch.save({
                    'epoch':             epoch,
                    'model_state':       self.model.state_dict(),
                    'val_rmse':          val_rmse,
                    'history':           self.history,
                    'normalizer_state':  self._build_normalizer_state(),
                }, ckpt_path)
                logger.info(f"  New best val_rmse={val_rmse:.5f} — saved")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    if can_stop_early:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(physics fully active since epoch {self.min_epochs_before_stopping})"
                        )
                        break
                    else:
                        logger.info(
                            f"  Early-stop condition met at epoch {epoch} but "
                            f"SUPPRESSED — physics not yet fully active "
                            f"(activates at epoch {self.min_epochs_before_stopping}). "
                            f"Continuing training."
                        )
                        self.no_improve = 0

        return self.history