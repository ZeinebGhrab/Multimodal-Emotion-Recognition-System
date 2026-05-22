"""
src/utils/early_stopping.py
────────────────────────────
Generic early-stopping callback for PyTorch training loops.

Supports monitoring val_loss (mode='min') or val_acc (mode='max').
Saves the best model checkpoint automatically and restores it on stop.

Usage:
    from src.utils.early_stopping import EarlyStopping

    es = EarlyStopping(patience=7, min_delta=0.001, mode='min')

    for epoch in range(epochs):
        val_loss, val_acc, _, _ = evaluate(...)

        if es(val_loss, model, checkpoint_path):   # True → stop
            print(f"Early stopping at epoch {epoch}")
            break

    # Best weights are already loaded back into `model`
"""

import torch
import os


class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    Args:
        patience    : epochs to wait after last improvement before stopping
        min_delta   : minimum change to qualify as an improvement
        mode        : 'min' (loss) or 'max' (accuracy / F1)
        restore_best: reload best weights when stopping (default True)
        verbose     : print progress messages
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-3,
                 mode: str = "min", restore_best: bool = True,
                 verbose: bool = True):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience     = patience
        self.min_delta    = min_delta
        self.mode         = mode
        self.restore_best = restore_best
        self.verbose      = verbose

        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self._best_path   = None          # path where best weights are cached

    # ── public interface ──────────────────────────────────────────────────────

    def __call__(self, metric: float,
                 model: "torch.nn.Module | None" = None,
                 checkpoint_path: "str | None" = None) -> bool:
        """
        Call at the end of each epoch.

        Args:
            metric          : value to monitor (val_loss or val_acc)
            model           : model whose state_dict is saved on improvement
            checkpoint_path : where to persist the best weights; if None, uses
                              an in-memory buffer (torch.save to BytesIO)

        Returns:
            True  → training should stop
            False → continue
        """
        if self.best_score is None:
            # First epoch — always an improvement
            self._update_best(metric, model, checkpoint_path)
        elif self._is_improvement(metric):
            self._update_best(metric, model, checkpoint_path)
        else:
            self.counter += 1
            if self.verbose:
                direction = "↓" if self.mode == "min" else "↑"
                print(
                    f"  [EarlyStopping] No improvement {direction} "
                    f"({self.counter}/{self.patience})  "
                    f"best={self.best_score:.5f}  current={metric:.5f}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best and model is not None:
                    self._restore(model)
                return True

        return False

    def reset(self):
        """Reset state — useful for k-fold cross-validation."""
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self._best_path = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_score - self.min_delta
        return metric > self.best_score + self.min_delta

    def _update_best(self, metric: float, model, path):
        improved_by = (
            f"{abs(metric - self.best_score):.5f}" if self.best_score is not None
            else "—"
        )
        self.best_score = metric
        self.counter    = 0

        if model is not None:
            if path is not None:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                torch.save(model.state_dict(), path)
                self._best_path = path
            else:
                # In-memory fallback (avoids disk I/O in quick experiments)
                import io
                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                self._best_state = buf.getvalue()
                self._best_path  = None

        if self.verbose:
            direction = "↓" if self.mode == "min" else "↑"
            print(
                f"  [EarlyStopping] Improvement {direction}  "
                f"best={self.best_score:.5f}  Δ={improved_by}  "
                f"(counter reset)"
            )

    def _restore(self, model):
        if self._best_path and os.path.exists(self._best_path):
            model.load_state_dict(
                torch.load(self._best_path, map_location="cpu"))
            if self.verbose:
                print(f"  [EarlyStopping] Best weights restored from {self._best_path}")
        elif hasattr(self, "_best_state"):
            import io
            buf = io.BytesIO(self._best_state)
            model.load_state_dict(torch.load(buf, map_location="cpu"))
            if self.verbose:
                print("  [EarlyStopping] Best weights restored from memory buffer")

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, "
            f"min_delta={self.min_delta}, mode='{self.mode}', "
            f"counter={self.counter}/{self.patience}, "
            f"best={self.best_score})"
        )