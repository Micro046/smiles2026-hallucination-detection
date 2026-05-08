"""
probe.py - Hallucination probe classifier.

Regularized MLP with dropout, weight decay, and early stopping on a small
internal validation slice. Threshold is tuned on the external validation
set in fit_hyperparameters() to maximize F1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HallucinationProbe(nn.Module):
    """Binary MLP probe with regularization and early stopping."""

    HIDDEN_1 = 512
    HIDDEN_2 = 128
    DROPOUT = 0.3
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 300
    PATIENCE = 20
    INTERNAL_VAL_FRAC = 0.1
    SEED = 42

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5
        self._device = _device()

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Sequential(
            nn.Linear(input_dim, self.HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.HIDDEN_1, self.HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.HIDDEN_2, 1),
        )
        self._net.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Network has not been built yet. Call fit() first.")
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._scaler.fit_transform(X)
        self._build_network(X_scaled.shape[1])

        # Carve a small internal val slice for early stopping (signature unchanged).
        if len(y) > 50 and len(np.unique(y)) > 1:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_scaled, y,
                test_size=self.INTERNAL_VAL_FRAC,
                random_state=self.SEED,
                stratify=y,
            )
        else:
            X_tr, X_va, y_tr, y_va = X_scaled, X_scaled, y, y

        X_tr_t = torch.from_numpy(X_tr).float().to(self._device)
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(self._device)
        X_va_t = torch.from_numpy(X_va).float().to(self._device)
        y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(self._device)

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32, device=self._device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY
        )

        best_val = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        for _ in range(self.MAX_EPOCHS):
            self.train()
            optimizer.zero_grad()
            loss = criterion(self(X_tr_t), y_tr_t)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_loss = criterion(self(X_va_t), y_va_t).item()

            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.PATIENCE:
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        X_t = torch.from_numpy(X_scaled).float().to(self._device)
        self.eval()
        with torch.no_grad():
            prob_pos = torch.sigmoid(self(X_t)).cpu().numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
