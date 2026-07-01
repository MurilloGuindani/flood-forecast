# claude-sonnet-4-6
"""
Autoregressive seq2seq tide prediction with Bahdanau-style attention.

Architecture:
  Encoder: LSTM/GRU over the 72h lookback window -> all hidden states (B, T, H)
  Decoder: runs 12 steps (t+1h .. t+12h), free-running (autoregressive).
           At each step:
             - attention over encoder hidden states, conditioned on decoder state
             - decoder input = [previous prediction, astro tide for that hour]
             - decoder cell -> small head -> scalar prediction for that hour
             - prediction feeds into the next step

Astro tide for t+1h..t+12h is known at inference time (it's an astronomical
forecast, not derived from observations) and is injected into the decoder at
every step, exactly like TideRNN's encoder gets the known future astro values
as a flat feature -- the difference here is the decoder also gets it as direct
per-step input rather than a flattened encoder feature.

Outputs (data/models/seq2seq/):
  - lstm_seq2seq_best.pt / gru_seq2seq_best.pt
  - metrics.csv             RMSE, MAE, R², MAPE, QL per model x horizon (val + test)
  - plots/predicted_vs_observed_*.png
  - plots/residuals_*.png
  - plots/train_curve_*.png
  - plots/attention_*.png   sample attention heatmap

Logs: logs/<date>-<hour>/<arch>_seq2seq.log

Usage:
    python tide_rnn_seq2seq.py --model both
    python tide_rnn_seq2seq.py --model lstm
"""

import argparse
import itertools
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from config import *

# ── Shared metric config (mirrors tide_rnn_model.py) ──────────────────────────

QUANTILES      = [0.05, 0.25, 0.5, 0.75, 0.95]
QUANTILE_ALPHA = 0.3

# Seq2seq output goes to its own subdir so it doesn't clobber TideRNN checkpoints
SEQ2SEQ_MODELS_DIR = MODELS_DIR.parent / "seq2seq"
SEQ2SEQ_PLOTS_DIR  = SEQ2SEQ_MODELS_DIR / "plots"


def quantile_loss(y: np.ndarray, preds: np.ndarray, q: float) -> float:
    e = y - preds
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))


def mape(y: np.ndarray, preds: np.ndarray) -> float:
    mask = np.abs(y) >= 1.0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y[mask] - preds[mask]) / y[mask])) * 100)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(arch_name: str, run_dir: str) -> logging.Logger:
    log_dir = Path("logs") / run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{arch_name.lower()}_seq2seq.log"

    logger = logging.getLogger(f"seq2seq.{arch_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    return logger


# ── Data ──────────────────────────────────────────────────────────────────────

def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    X            = torch.tensor(data["X"], dtype=torch.float32)             # (N, T, F)
    y            = torch.tensor(data["y"], dtype=torch.float32)             # (N, H)
    astro_future = torch.tensor(data["astro_future"], dtype=torch.float32)  # (N, H)
    print(f"[load]  X={tuple(X.shape)}  y={tuple(y.shape)}  astro_future={tuple(astro_future.shape)}")
    return X, y, astro_future


def normalize(X_tr, y_tr, af_tr, X_val, y_val, af_val, X_te, y_te, af_te):
    """
    Robust scaling using train statistics only.
    astro_future is scaled with the SAME y_scaler as the target, since both
    live in tide-cm space and the decoder needs them on a consistent scale.
    """
    def _scale_X(scaler, X, fit=False):
        flat = X.reshape(-1, X.shape[-1]).cpu().numpy()
        if fit:
            scaler.fit(flat)
        return torch.as_tensor(
            scaler.transform(flat), dtype=X.dtype, device=X.device
        ).reshape(X.shape)

    def _scale_y(scaler, y, fit=False):
        arr = y.cpu().numpy()
        if fit:
            scaler.fit(arr)
        return torch.as_tensor(scaler.transform(arr), dtype=y.dtype, device=y.device)

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_tr_s  = _scale_X(x_scaler, X_tr,  fit=True)
    X_val_s = _scale_X(x_scaler, X_val)
    X_te_s  = _scale_X(x_scaler, X_te)

    y_tr_s  = _scale_y(y_scaler, y_tr,  fit=True)
    y_val_s = _scale_y(y_scaler, y_val)
    y_te_s  = _scale_y(y_scaler, y_te)

    # astro_future shares y_scaler (same units, same scale center as the target)
    af_tr_s  = _scale_y(y_scaler, af_tr)
    af_val_s = _scale_y(y_scaler, af_val)
    af_te_s  = _scale_y(y_scaler, af_te)

    return (
        TensorDataset(X_tr_s,  y_tr_s,  af_tr_s),
        TensorDataset(X_val_s, y_val_s, af_val_s),
        TensorDataset(X_te_s,  y_te_s,  af_te_s),
        x_scaler,
        y_scaler,
    )


# ── Loss (identical to tide_rnn_model.py) ─────────────────────────────────────

class PinballLoss(nn.Module):
    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.unsqueeze(-1)
        p = preds.unsqueeze(-1)
        q = self.quantiles.view(1, 1, -1)
        e = t - p
        loss = torch.where(e >= 0, q * e, (q - 1) * e)
        return loss.mean()


# ── Attention ─────────────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """Additive attention: score(s_t, h_i) = v^T tanh(W_s s_t + W_h h_i)."""
    def __init__(self, dec_hidden: int, enc_hidden: int, attn_dim: int = 64):
        super().__init__()
        self.W_s = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.W_h = nn.Linear(enc_hidden, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_state: torch.Tensor, enc_outputs: torch.Tensor):
        """
        dec_state:   (B, dec_hidden)
        enc_outputs: (B, T, enc_hidden)
        returns: context (B, enc_hidden), weights (B, T)
        """
        T = enc_outputs.shape[1]
        s = self.W_s(dec_state).unsqueeze(1).expand(-1, T, -1)   # (B, T, attn_dim)
        h = self.W_h(enc_outputs)                                # (B, T, attn_dim)
        scores  = self.v(torch.tanh(s + h)).squeeze(-1)          # (B, T)
        weights = F.softmax(scores, dim=-1)                      # (B, T)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)  # (B, enc_hidden)
        return context, weights


# ── Encoder / Decoder ──────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, input_size, cell="LSTM", hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "LSTM" else nn.GRU
        self.is_lstm = cell == "LSTM"
        self.rnn = rnn_cls(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """Returns all hidden states (B, T, H) and final state for decoder init."""
        outputs, final = self.rnn(x)
        return outputs, final


class AttnDecoderCell(nn.Module):
    """Single-step decoder cell: input = [prev_pred, astro_t] + attention context."""
    def __init__(self, hidden_size, cell="LSTM", dropout=0.2, attn_dim=64):
        super().__init__()
        self.is_lstm = cell == "LSTM"
        cell_cls = nn.LSTMCell if cell == "LSTM" else nn.GRUCell

        self.attn = BahdanauAttention(hidden_size, hidden_size, attn_dim)
        # decoder cell input: prev_pred(1) + astro(1) + context(hidden_size)
        self.cell = cell_cls(2 + hidden_size, hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 2, 8), 1),
        )

    def forward(self, prev_pred, astro_t, state, enc_outputs):
        """
        prev_pred:   (B, 1)
        astro_t:     (B, 1)
        state:       (h, c) for LSTM or h for GRU, each (B, hidden_size)
        enc_outputs: (B, T, hidden_size)
        """
        dec_h = state[0] if self.is_lstm else state
        context, attn_weights = self.attn(dec_h, enc_outputs)        # (B, H), (B, T)
        cell_input = torch.cat([prev_pred, astro_t, context], dim=-1)

        if self.is_lstm:
            h, c = self.cell(cell_input, state)
            new_state = (h, c)
        else:
            h = self.cell(cell_input, state)
            new_state = h

        pred = self.head(h)   # (B, 1)
        return pred, new_state, attn_weights


class TideSeq2Seq(nn.Module):
    """
    Encoder-decoder with attention. Encoder and decoder share hidden_size so
    the encoder's final state can directly seed the decoder.
    """
    def __init__(self, input_size, cell="LSTM", hidden_size=128, num_layers=2,
                 dropout=0.2, output_size=len(HORIZONS), attn_dim=64):
        super().__init__()
        self.is_lstm   = cell == "LSTM"
        self.output_size = output_size
        self.encoder = Encoder(input_size, cell, hidden_size, num_layers, dropout)
        self.decoder = AttnDecoderCell(hidden_size, cell, dropout, attn_dim)
        # seed token for prev_pred at step 0 (replaced by tide_obs_lag1h in run())
        self.register_buffer("seed_pred", torch.zeros(1, 1))

    def forward(self, x, astro_future, init_pred=None, return_attn=False):
        """
        x:            (B, T, F) encoder input window
        astro_future: (B, H) known astro tide for t+1h..t+Hh
        init_pred:    (B, 1) last observed tide value to seed the decoder
                      (defaults to zeros if not given)
        """
        B = x.shape[0]
        enc_outputs, enc_final = self.encoder(x)

        if self.is_lstm:
            h, c = enc_final
            state = (h[-1], c[-1])           # (B, hidden_size) from last layer
        else:
            h = enc_final
            state = h[-1]

        prev_pred = init_pred if init_pred is not None else self.seed_pred.expand(B, -1)

        preds, attn_maps = [], []
        for i in range(self.output_size):
            astro_t = astro_future[:, i:i+1]
            pred, state, attn_w = self.decoder(prev_pred, astro_t, state, enc_outputs)
            preds.append(pred)
            if return_attn:
                attn_maps.append(attn_w)
            prev_pred = pred   # free-running: feed own prediction forward

        out = torch.cat(preds, dim=1)   # (B, H)
        if return_attn:
            return out, torch.stack(attn_maps, dim=1)   # (B, H, T)
        return out


MODEL_CLS = {"LSTM": TideSeq2Seq, "GRU": TideSeq2Seq}


def _model_params(params: dict):
    return {k: v for k, v in params.items() if k not in ("lr", "weight_decay")}


# ── Training loop ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_loss(model, loader, mse_fn, pinball_fn, device) -> float:
    model.eval()
    total = 0.0
    for xb, yb, afb in loader:
        xb, yb, afb = xb.to(device), yb.to(device), afb.to(device)
        init_pred = xb[:, -1, :1]   # placeholder; replaced with real lag1 col index in run()
        p = model(xb, afb, init_pred=None)
        total += (mse_fn(p, yb) + QUANTILE_ALPHA * pinball_fn(p, yb)).item() * len(xb)
    return total / len(loader.dataset)


def _run_train(
    model, train_ds, val_ds, params: dict, device, *,
    patience: int, record_curves: bool, logger: logging.Logger, label: str = "",
):
    mse_fn     = nn.MSELoss()
    pinball_fn = PinballLoss(QUANTILES).to(device)

    opt = torch.optim.Adam(
        model.parameters(), lr=params["lr"],
        weight_decay=params.get("weight_decay", 0.0),
    )
    sched    = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    t_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    v_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    best_val, best_state = float("inf"), None
    patience_counter      = 0
    tr_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb, afb in t_loader:
            xb, yb, afb = xb.to(device), yb.to(device), afb.to(device)
            opt.zero_grad()
            p    = model(xb, afb, init_pred=None)   # free-running: no teacher forcing
            loss = mse_fn(p, yb) + QUANTILE_ALPHA * pinball_fn(p, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item() * len(xb)
        tr = total / len(t_loader.dataset)

        val = _eval_loss(model, v_loader, mse_fn, pinball_fn, device)
        sched.step(val)

        if record_curves:
            tr_losses.append(tr)
            val_losses.append(val)

        if val < best_val:
            best_val, best_state, patience_counter = val, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if record_curves and (epoch % 10 == 0 or epoch == 1):
            lr_now = opt.param_groups[0]["lr"]
            logger.info(
                f"{label}  epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={tr:.4f}  val_loss={val:.4f}  lr={lr_now:.2e}"
            )

    return best_val, best_state, tr_losses, val_losses


def grid_search(arch_name, input_size, train_ds, val_ds, device, logger):
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))

    logger.info(f"Grid search: {len(combos)} configs")
    logger.info(f"  {'config':>6}  {'hidden':>6}  {'layers':>6}  {'drop':>5}  {'lr':>7}  {'l2':>7}  val_loss")
    logger.info("  " + "─" * 60)

    best_val, best_params, best_state = float("inf"), None, None

    for idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        model  = TideSeq2Seq(input_size, cell=arch_name,
                             **_model_params(params)).to(device)

        val_loss, state, _, _ = _run_train(
            model, train_ds, val_ds, params, device,
            patience=20, record_curves=False, logger=logger,
            label=f"[gs {idx}/{len(combos)}]",
        )

        marker = " ◄ best" if val_loss < best_val else ""
        logger.info(
            f"  {idx:>6}  {params['hidden_size']:>6}  {params['num_layers']:>6}  "
            f"{params['dropout']:>5.2f}  {params['lr']:>7.0e}  "
            f"{params.get('weight_decay', 0.0):>7.0e}  "
            f"{val_loss:.4f}{marker}"
        )

        if val_loss < best_val:
            best_val, best_params, best_state = val_loss, params, state

    logger.info(f"Best config: {best_params}  val_loss={best_val:.4f}")
    print(f"  [{arch_name}] Grid search done. Best val_loss={best_val:.4f}  params={best_params}")
    return best_params, best_state, best_val


def train_best(arch_name, input_size, best_params, train_ds, val_ds, device, logger):
    model = TideSeq2Seq(input_size, cell=arch_name, **_model_params(best_params)).to(device)

    logger.info(f"Final training — best params: {best_params}")
    print(f"\n  Final training with best params:")

    best_val, best_state, tr_losses, val_losses = _run_train(
        model, train_ds, val_ds, best_params, device,
        patience=15, record_curves=True, logger=logger, label="[final]",
    )

    model.load_state_dict(best_state)
    logger.info(f"Final training done. best_val_loss={best_val:.4f}")
    return model, tr_losses, val_losses


# ── Inference & metrics ───────────────────────────────────────────────────────

@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb, afb in loader:
        p = model(xb.to(device), afb.to(device), init_pred=None)
        preds.append(p.cpu())
        targets.append(yb)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


@torch.no_grad()
def predict_one_with_attn(model, x, astro_future, device):
    """For attention visualisation: single sample, returns preds + attn map."""
    model.eval()
    p, attn = model(x.to(device), astro_future.to(device), init_pred=None, return_attn=True)
    return p.cpu().numpy(), attn.cpu().numpy()


def metrics_dict(preds, targets, split: str, model_name: str, y_scaler) -> list[dict]:
    preds   = y_scaler.inverse_transform(preds)
    targets = y_scaler.inverse_transform(targets)
    rows = []
    for i, h in enumerate(HORIZONS):
        p, t = preds[:, i], targets[:, i]
        row = {
            "model": model_name, "horizon": f"t+{h}h", "split": split,
            "rmse": np.sqrt(mean_squared_error(t, p)),
            "mae":  mean_absolute_error(t, p),
            "r2":   r2_score(t, p),
            "mape": mape(t, p),
        }
        for q in QUANTILES:
            row[f"ql_q{int(q*100):02d}"] = quantile_loss(t, p, q)
        rows.append(row)
    return rows


def _log_metrics(rows, logger):
    for row in rows:
        ql_str = "  ".join(
            f"QL{int(q*100):02d}={row[f'ql_q{int(q*100):02d}']:.3f}" for q in QUANTILES
        )
        msg = (
            f"  {row['horizon']}  RMSE={row['rmse']:.3f}  MAE={row['mae']:.3f}  "
            f"R²={row['r2']:.4f}  MAPE={row['mape']:.2f}%  {ql_str}"
        )
        print(msg)
        logger.info(msg)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_train_curve(tr_losses, val_losses, name, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr_losses,  label="Train (MSE + QL)", linewidth=1.2, color="#4C72B0")
    ax.plot(val_losses, label="Val (MSE + QL)",   linewidth=1.2, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Combined loss")
    ax.set_title(f"{name} — Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"train_curve_{name.lower()}_seq2seq.png", dpi=120)
    plt.close(fig)


def plot_predicted_vs_observed(preds, targets, name, horizon, split, plots_dir):
    p, t = preds[:, HORIZONS.index(horizon)], targets[:, HORIZONS.index(horizon)]
    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    fig.suptitle(f"{name} (seq2seq) — t+{horizon}h  [{split}]  Predicted vs Observed",
                 fontsize=13, fontweight="bold")

    axes[0].plot(t, linewidth=0.7, label="Observed",  color="#4C72B0", alpha=0.9)
    axes[0].plot(p, linewidth=0.7, label="Predicted", color="orange",  alpha=0.8)
    axes[0].set_ylabel("Tide (cm)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(t, p, s=2, alpha=0.3, color="#4C72B0")
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    axes[1].plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
    axes[1].set_xlabel("Observed (cm)")
    axes[1].set_ylabel("Predicted (cm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / f"predicted_vs_observed_{name.lower()}_seq2seq_{split}_t{horizon}h.png", dpi=120)
    plt.close(fig)


def plot_residuals(preds, targets, name, horizon, split, plots_dir):
    p   = preds[:, HORIZONS.index(horizon)]
    t   = targets[:, HORIZONS.index(horizon)]
    res = t - p

    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(f"{name} (seq2seq) — t+{horizon}h  [{split}]  Residuals",
                 fontsize=13, fontweight="bold")

    axes[0].plot(res, linewidth=0.6, color="#D9534F", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].fill_between(range(len(res)), res, 0, where=res > 0, alpha=0.3, color="#D9534F")
    axes[0].fill_between(range(len(res)), res, 0, where=res < 0, alpha=0.3, color="#4C72B0")
    axes[0].set_ylabel("Residual (cm)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(res, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0,          color="black",  linewidth=1)
    axes[1].axvline(res.mean(), color="red",    linestyle="--", linewidth=1,
                    label=f"Mean {res.mean():.2f} cm")
    axes[1].axvline(res.mean() + res.std(), color="orange", linestyle=":", linewidth=1,
                    label=f"±1σ {res.std():.2f} cm")
    axes[1].axvline(res.mean() - res.std(), color="orange", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Residual (cm)")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(p, res, s=2, alpha=0.3, color="#4C72B0")
    axes[2].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("Predicted (cm)")
    axes[2].set_ylabel("Residual (cm)")
    axes[2].set_title("Residuals vs Predicted")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / f"residuals_{name.lower()}_seq2seq_{split}_t{horizon}h.png", dpi=120)
    plt.close(fig)


def plot_attention(attn_map, name, plots_dir, sample_idx=0):
    """attn_map: (H, T) for one sample — decoder steps x encoder timesteps."""
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(attn_map, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Encoder hour (lookback, 0=oldest)")
    ax.set_ylabel("Decoder step (t+1h .. t+12h)")
    ax.set_yticks(range(len(HORIZONS)))
    ax.set_yticklabels([f"t+{h}h" for h in HORIZONS])
    ax.set_title(f"{name} (seq2seq) — Attention weights (sample {sample_idx})")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(plots_dir / f"attention_{name.lower()}_seq2seq.png", dpi=120)
    plt.close(fig)


# ── Runner ────────────────────────────────────────────────────────────────────

def run(arch_name, input_size, train_ds, val_ds, test_ds, device,
        all_metrics, y_scaler, run_dir):
    print(f"\n{'═'*60}")
    print(f"  {arch_name} (seq2seq + attention)")
    print(f"{'═'*60}")

    logger = setup_logger(arch_name, run_dir)

    best_params, _, _ = grid_search(arch_name, input_size, train_ds, val_ds, device, logger)

    model, tr_losses, val_losses = train_best(
        arch_name, input_size, best_params, train_ds, val_ds, device, logger
    )

    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    val_preds,  val_targets  = predict_all(model, val_loader,  device)
    test_preds, test_targets = predict_all(model, test_loader, device)

    val_rows  = metrics_dict(val_preds,  val_targets,  "val",  arch_name, y_scaler)
    test_rows = metrics_dict(test_preds, test_targets, "test", arch_name, y_scaler)

    print(f"\n  ── {arch_name} VAL Results ──")
    logger.info(f"── {arch_name} VAL Results ──")
    _log_metrics(val_rows, logger)

    print(f"\n  ── {arch_name} TEST Results ──")
    logger.info(f"── {arch_name} TEST Results ──")
    _log_metrics(test_rows, logger)

    all_metrics.extend(val_rows)
    all_metrics.extend(test_rows)

    ckpt = SEQ2SEQ_MODELS_DIR / f"{arch_name.lower()}_seq2seq_best.pt"
    torch.save({"model_state": model.state_dict(),
                "model": arch_name, "best_params": best_params}, ckpt)
    print(f"\n  Saved → {ckpt}")
    logger.info(f"Saved → {ckpt}")

    plot_train_curve(tr_losses, val_losses, arch_name, SEQ2SEQ_PLOTS_DIR)
    for h in HORIZONS:
        for split, preds, targets in [("val",  val_preds,  val_targets),
                                      ("test", test_preds, test_targets)]:
            plot_predicted_vs_observed(preds, targets, arch_name, h, split, SEQ2SEQ_PLOTS_DIR)
            plot_residuals(preds, targets, arch_name, h, split, SEQ2SEQ_PLOTS_DIR)

    # Attention heatmap for one test sample
    xb, yb, afb = test_ds[0]
    _, attn = predict_one_with_attn(
        model, xb.unsqueeze(0), afb.unsqueeze(0), device
    )
    plot_attention(attn[0], arch_name, SEQ2SEQ_PLOTS_DIR)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["lstm", "gru", "both"])
    args = parser.parse_args()

    SEQ2SEQ_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SEQ2SEQ_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    run_dir = datetime.now().strftime("%Y%m%d-%H")

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    print(f"[device] {device}")

    X_tr,  y_tr,  af_tr  = load_npz(FEATURES_DIR / "ml_features_sequence_tr.npz")
    X_val, y_val, af_val = load_npz(FEATURES_DIR / "ml_features_sequence_val.npz")
    X_te,  y_te,  af_te  = load_npz(FEATURES_DIR / "ml_features_sequence_te.npz")

    train_ds, val_ds, test_ds, _, y_scaler = normalize(
        X_tr, y_tr, af_tr, X_val, y_val, af_val, X_te, y_te, af_te
    )

    input_size = X_tr.shape[2]
    print(f"[info]   features={input_size}  lookback={X_tr.shape[1]}h  horizons={HORIZONS}")
    print(f"[splits] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    all_metrics = []
    run_kwargs  = dict(
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        device=device, all_metrics=all_metrics, y_scaler=y_scaler, run_dir=run_dir,
    )

    if args.model in ("lstm", "both"):
        run("LSTM", input_size, **run_kwargs)
    if args.model in ("gru", "both"):
        run("GRU", input_size, **run_kwargs)

    df = pd.DataFrame(all_metrics)
    csv_path = SEQ2SEQ_MODELS_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'─'*60}")
    print("Summary")
    print("─" * 60)
    print(df.to_string(index=False))
    print(f"\n[saved] {csv_path}")
    print(f"[saved] models and plots in {SEQ2SEQ_MODELS_DIR}")
    print(f"[logs]  logs/{run_dir}/")


if __name__ == "__main__":
    main()
