"""
Tide level prediction: LSTM and GRU on ml_features_sequence.npz

Outputs (data/models/rnn/):
  - lstm_best.pt / gru_best.pt          best checkpoint (lowest val MSE)
  - metrics.csv                          RMSE, MAE, R² per model × horizon (val + test)
  - plots/predicted_vs_observed_*.png
  - plots/residuals_*.png
  - plots/train_curve_*.png

Hyper-parameter grid is searched on the VAL set.
TEST set is only touched once, after the best config is chosen.

Usage:
    python tide_rnn_model.py --npz data/features/ml_features_sequence.npz --model both
    python tide_rnn_model.py --npz data/features/ml_features_sequence.npz --model lstm
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import RobustScaler

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "data" / "models" / "nn"
PLOTS_DIR    = MODELS_DIR / "plots"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
HORIZONS   = [1]
BATCH_SIZE = 64
EPOCHS     = 100
SEED       = 42

# Hyper-parameter grid — searched on VAL set
PARAM_GRID = {
    "hidden_size": [4, 8, 16, 64],
    "num_layers":  [1, 2, 4],
    "dropout":     [0.1, 0.2, 0.3, 0.5],
    "lr": [1e-3, 5e-4, 1e-4],
    "weight_decay": [0.0, 1e-4, 1e-3],
}

# ─────────────────────────────────────────────────────────────────────────────


# ── Data ──────────────────────────────────────────────────────────────────────

def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    X = torch.tensor(data["X"], dtype=torch.float32)   # (N, T, F)
    y = torch.tensor(data["y"], dtype=torch.float32)   # (N, 3)
    print(f"[load]  X={tuple(X.shape)}  y={tuple(y.shape)}")
    return X, y



def normalize(X_tr, y_tr, X_val, y_val, X_te, y_te):
    """
    Robust scaling using train statistics only.

    X shape: (N, T, F)
    y shape: (N, H)

    Returns:
        train_ds
        val_ds
        test_ds
        x_scaler
        y_scaler
    """

    # --------------------------------------------------
    # X scaler
    # --------------------------------------------------

    x_scaler = RobustScaler()

    X_tr_flat = X_tr.reshape(-1, X_tr.shape[-1]).cpu().numpy()

    x_scaler.fit(X_tr_flat)

    X_tr_scaled = torch.as_tensor(
        x_scaler.transform(X_tr_flat),
        dtype=X_tr.dtype,
        device=X_tr.device,
    ).reshape(X_tr.shape)

    X_val_scaled = torch.as_tensor(
        x_scaler.transform(
            X_val.reshape(-1, X_val.shape[-1]).cpu().numpy()
        ),
        dtype=X_val.dtype,
        device=X_val.device,
    ).reshape(X_val.shape)

    X_te_scaled = torch.as_tensor(
        x_scaler.transform(
            X_te.reshape(-1, X_te.shape[-1]).cpu().numpy()
        ),
        dtype=X_te.dtype,
        device=X_te.device,
    ).reshape(X_te.shape)

    # --------------------------------------------------
    # y scaler
    # --------------------------------------------------

    y_scaler = RobustScaler()

    y_scaler.fit(y_tr.cpu().numpy())

    y_tr_scaled = torch.as_tensor(
        y_scaler.transform(y_tr.cpu().numpy()),
        dtype=y_tr.dtype,
        device=y_tr.device,
    )

    y_val_scaled = torch.as_tensor(
        y_scaler.transform(y_val.cpu().numpy()),
        dtype=y_val.dtype,
        device=y_val.device,
    )

    y_te_scaled = torch.as_tensor(
        y_scaler.transform(y_te.cpu().numpy()),
        dtype=y_te.dtype,
        device=y_te.device,
    )

    return (
        TensorDataset(X_tr_scaled, y_tr_scaled),
        TensorDataset(X_val_scaled, y_val_scaled),
        TensorDataset(X_te_scaled, y_te_scaled),
        x_scaler,
        y_scaler,
    )
# ── Models ────────────────────────────────────────────────────────────────────
# GPT-5.5

class TideMLP(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
        output_size=len(HORIZONS),
    ):
        super().__init__()

        layers = []

        in_dim = input_size

        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_size

        layers.append(
            nn.Linear(hidden_size, output_size)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)

        return self.net(x)

class TideLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, output_size=len(HORIZONS)):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        ) if hidden_size >= 128 else nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


class TideGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, output_size=len(HORIZONS)):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        ) if hidden_size >= 128 else nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


MODEL_CLS = {"LSTM": TideLSTM, "GRU": TideGRU, "MLP": TideMLP}


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_mse(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item() * len(xb)
    return total / len(loader.dataset)


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu())
        targets.append(yb)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def metrics_dict(preds, targets, split: str, model_name: str, y_scaler) -> list[dict]:

    preds = y_scaler.inverse_transform(preds)
    targets = y_scaler.inverse_transform(targets)

    rows = []

    for i, h in enumerate(HORIZONS):

        p = preds[:, i]
        t = targets[:, i]

        rows.append({
            "model":   model_name,
            "horizon": f"t+{h}h",
            "split":   split,
            "rmse":    np.sqrt(mean_squared_error(t, p)),
            "mae":     mean_absolute_error(t, p),
            "r2":      r2_score(t, p),
        })

    return rows


# ── Grid search (on val) ──────────────────────────────────────────────────────

def grid_search(arch_name, input_size, train_ds, val_ds, device):
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)

    best_val_mse = float("inf")
    best_params  = None
    best_state   = None
    criterion    = nn.MSELoss()

    print(f"\n  Grid search: {total} configs")
    print(f"  {'config':>6}  {'hidden':>6}  {'layers':>6}  {'drop':>5}  {'lr':>7}  {'l2':>7}  val_mse")
    print(f"  {'─'*60}")

    for idx, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        model  = MODEL_CLS[arch_name](input_size, **{k: v for k, v in params.items()
                                                    if k not in ("lr", "weight_decay")}).to(device)
        opt    = torch.optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params.get("weight_decay", 0.0),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        t_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        v_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE)

        best_e_mse, best_e_state = float("inf"), None
        patience_counter = 0
        EARLY_STOP_PATIENCE = 15

        for _ in range(EPOCHS):
            train_epoch(model, t_loader, opt, criterion, device)
            v = eval_mse(model, v_loader, criterion, device)
            sched.step()
            if v < best_e_mse:
                best_e_mse      = v
                best_e_state    = {k: v_.clone() for k, v_ in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    break

        marker = " ◄ best" if best_e_mse < best_val_mse else ""
        print(f"  {idx:>6}  {params['hidden_size']:>6}  {params['num_layers']:>6}  "
            f"{params['dropout']:>5.2f}  {params['lr']:>7.0e}  "
            f"{params.get('weight_decay', 0.0):>7.0e}  "
            f"{best_e_mse:.4f}{marker}")

        if best_e_mse < best_val_mse:
            best_val_mse = best_e_mse
            best_params  = params
            best_state   = best_e_state

    print(f"\n  Best config: {best_params}  val_mse={best_val_mse:.4f}")
    return best_params, best_state, best_val_mse


# ── Final training with best params ──────────────────────────────────────────

def train_best(arch_name, input_size, best_params, train_ds, val_ds, device):
    """Re-train with best params, tracking curve, return (model, train_losses, val_losses)."""
    model = MODEL_CLS[arch_name](
                input_size,
                **{k: v for k, v in best_params.items() if k not in ("lr", "weight_decay")}
            ).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.MSELoss()
    t_loader  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    v_loader  = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    best_val, best_state = float("inf"), None
    tr_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        tr  = train_epoch(model, t_loader, opt, criterion, device)
        val = eval_mse(model, v_loader, criterion, device)
        sched.step()
        tr_losses.append(tr)
        val_losses.append(val)

        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

        if epoch % 10 == 0 or epoch == 1:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    epoch {epoch:3d}/{EPOCHS}  train_mse={tr:.4f}"
                  f"  val_mse={val:.4f}  lr={lr_now:.2e}")

    model.load_state_dict(best_state)
    return model, tr_losses, val_losses


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_train_curve(tr_losses, val_losses, name, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tr_losses,  label="Train MSE", linewidth=1.2, color="#4C72B0")
    ax.plot(val_losses, label="Val MSE",   linewidth=1.2, color="orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (cm²)")
    ax.set_title(f"{name} — Training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"train_curve_{name.lower()}.png", dpi=120)
    plt.close(fig)


def plot_predicted_vs_observed(preds, targets, name, horizon, split, plots_dir):
    p, t = preds[:, HORIZONS.index(horizon)], targets[:, HORIZONS.index(horizon)]
    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    fig.suptitle(f"{name} — t+{horizon}h  [{split}]  Predicted vs Observed",
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
    fig.savefig(plots_dir / f"predicted_vs_observed_{name.lower()}_{split}_t{horizon}h.png",
                dpi=120)
    plt.close(fig)


def plot_residuals(preds, targets, name, horizon, split, plots_dir):
    p   = preds[:, HORIZONS.index(horizon)]
    t   = targets[:, HORIZONS.index(horizon)]
    res = t - p

    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(f"{name} — t+{horizon}h  [{split}]  Residuals",
                 fontsize=13, fontweight="bold")

    axes[0].plot(res, linewidth=0.6, color="#D9534F", alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].fill_between(range(len(res)), res, 0,
                         where=res > 0, alpha=0.3, color="#D9534F")
    axes[0].fill_between(range(len(res)), res, 0,
                         where=res < 0, alpha=0.3, color="#4C72B0")
    axes[0].set_ylabel("Residual (cm)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(res, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0,        color="black",  linewidth=1)
    axes[1].axvline(res.mean(), color="red",  linestyle="--", linewidth=1,
                    label=f"Mean {res.mean():.2f} cm")
    axes[1].axvline(res.mean() + res.std(), color="orange", linestyle=":",
                    linewidth=1, label=f"±1σ {res.std():.2f} cm")
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
    fig.savefig(plots_dir / f"residuals_{name.lower()}_{split}_t{horizon}h.png", dpi=120)
    plt.close(fig)


# ── Runner ────────────────────────────────────────────────────────────────────

def run(arch_name, input_size, train_ds, val_ds, test_ds, device, all_metrics, y_scaler):
    print(f"\n{'═'*60}")
    print(f"  {arch_name}")
    print(f"{'═'*60}")

    if arch_name == "MLP":
        input_size = int(np.prod(train_ds[0][0].shape))

    # 1. Grid search on val
    best_params, _, _ = grid_search(arch_name, input_size, train_ds, val_ds, device)

    # 2. Re-train with best params (tracked curve)
    print(f"\n  Final training with best params:")
    model, tr_losses, val_losses = train_best(
        arch_name, input_size, best_params, train_ds, val_ds, device
    )

    # 3. VAL evaluation (sanity — model was selected here)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    val_preds,  val_targets  = predict_all(model, val_loader,  device)
    test_preds, test_targets = predict_all(model, test_loader, device)

    print(f"\n  ── {arch_name} VAL Results ──")
    for row in metrics_dict(val_preds, val_targets, "val", arch_name, y_scaler):
        print(f"    {row['horizon']}  RMSE={row['rmse']:.3f}  "
              f"MAE={row['mae']:.3f}  R²={row['r2']:.4f}")

    print(f"\n  ── {arch_name} TEST Results ──")
    for row in metrics_dict(test_preds, test_targets, "test", arch_name, y_scaler):
        print(f"    {row['horizon']}  RMSE={row['rmse']:.3f}  "
              f"MAE={row['mae']:.3f}  R²={row['r2']:.4f}")

    all_metrics.extend(metrics_dict(val_preds,  val_targets,  "val",  arch_name, y_scaler))
    all_metrics.extend(metrics_dict(test_preds, test_targets, "test", arch_name, y_scaler))

    # 4. Save checkpoint
    ckpt = MODELS_DIR / f"{arch_name.lower()}_best.pt"
    torch.save({"model_state": model.state_dict(),
                "model": arch_name,
                "best_params": best_params}, ckpt)
    print(f"\n  Saved → {ckpt}")

    # 5. Plots
    plot_train_curve(tr_losses, val_losses, arch_name, PLOTS_DIR)
    for h in HORIZONS:
        for split, preds, targets in [("val",  val_preds,  val_targets),
                                      ("test", test_preds, test_targets)]:
            plot_predicted_vs_observed(preds, targets, arch_name, h, split, PLOTS_DIR)
            plot_residuals(preds, targets, arch_name, h, split, PLOTS_DIR)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both",
                        choices=["mlp","lstm", "gru", "both"])
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    print(f"[device] {device}")

    X_tr, y_tr = load_npz(FEATURES_DIR /"ml_features_sequence_tr.npz")
    X_val, y_val = load_npz(FEATURES_DIR /"ml_features_sequence_val.npz")
    X_te, y_te = load_npz(FEATURES_DIR /"ml_features_sequence_te.npz")
    train_ds, val_ds, test_ds, X_scaler,y_scaler = normalize(X_tr, y_tr, X_val, y_val, X_te, y_te)
    print(y_scaler)
    input_size = X_tr.shape[2]
    print(f"[info]   features={input_size}  lookback={X_tr.shape[1]}h  horizons={HORIZONS}")
    print(f"[splits] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    all_metrics = []

    if args.model in ("mlp", "both"):
        run("MLP",  input_size, train_ds, val_ds, test_ds, device, all_metrics, y_scaler)
    if args.model in ("lstm", "both"):
        run("LSTM", input_size, train_ds, val_ds, test_ds, device, all_metrics, y_scaler)
    if args.model in ("gru", "both"):
        run("GRU",  input_size, train_ds, val_ds, test_ds, device, all_metrics, y_scaler)


    # Save metrics
    df = pd.DataFrame(all_metrics)
    csv_path = MODELS_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'─'*60}")
    print("Summary")
    print('─'*60)
    print(df.to_string(index=False))
    print(f"\n[saved] {csv_path}")
    print(f"[saved] models and plots in {MODELS_DIR}")


if __name__ == "__main__":
    main()