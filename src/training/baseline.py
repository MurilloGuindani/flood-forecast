# claude-sonnet-4-6
"""
Baseline model training: XGBoost and Random Forest.
One model per horizon (t+1h … t+12h) x 2 algorithms = 24 models total.

Outputs (data/models/):
  - xgb_t{h}h.json / rf_t{h}h.joblib     trained models
  - metrics.csv                            RMSE, MAE, R², MAPE, QL per model/horizon
  - plots/predicted_vs_observed_{model}_t{h}h.png   square, Observed vs Predicted (cm)
  - plots/residuals_{model}_t{h}h.png               square, Predicted vs Residual (cm)
  - plots/importance_{model}_t{h}h.png              square, feature importance
  - plots/learning_curve_{model}_t{h}h.png          square, train/val RMSE (cm) for best model overall

Logs: logs/<date>-<hour>/xgb.log  /  logs/<date>-<hour>/rf.log
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

from config import *

# ── Shared metric config ──────────────────────────────────────────────────────

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
FIGSIZE = (6, 6)  # square, fits IEEE single-column width


def quantile_loss(y: np.ndarray, preds: np.ndarray, q: float) -> float:
    """Pinball loss for quantile q (preds = point forecast used as proxy)."""
    e = y - preds
    return float(np.mean(np.where(e >= 0, q * e, (q - 1) * e)))


def mape(y: np.ndarray, preds: np.ndarray) -> float:
    """Mean Absolute Percentage Error, skipping near-zero targets (|y| < 1 cm)."""
    mask = np.abs(y) >= 1.0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y[mask] - preds[mask]) / y[mask])) * 100)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(name: str, run_dir: str) -> logging.Logger:
    log_dir = Path("logs") / run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name.lower()}.log"

    logger = logging.getLogger(f"baseline.{name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    return logger


# ── Data ──────────────────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(FEATURES_DIR / "split_train.parquet")
    val   = pd.read_parquet(FEATURES_DIR / "split_val.parquet")
    test  = pd.read_parquet(FEATURES_DIR / "split_test.parquet")
    return train, val, test


def get_xy(df: pd.DataFrame, horizon: int, astro=False) -> tuple[np.ndarray, np.ndarray]:
    target_cols  = [f"target_t+{h}h" for h in HORIZONS]
    feature_cols = [c for c in df.columns if c != "datetime" and c not in target_cols]
    if astro:
        feature_cols = ["tide_astro_cm"]
    return df[feature_cols].values, df[f"target_t+{horizon}h"].values


def feature_cols(df: pd.DataFrame) -> list[str]:
    target_cols = [f"target_t+{h}h" for h in HORIZONS]
    return [c for c in df.columns if c != "datetime" and c not in target_cols]


# ── Grid search on val set ────────────────────────────────────────────────────

def grid_search(model_cls, param_grid: dict, X_tr, y_tr, X_val, y_val,
                logger: logging.Logger, fit_kwargs: dict = {}) -> tuple[object, dict]:
    configs = list(ParameterGrid(param_grid))

    logger.info(f"Grid search: {len(configs)} configs")
    logger.info(f"  {'config':>6}  {'params':<60}  val_rmse")
    logger.info("  " + "─" * 80)

    best_rmse, best_model, best_params = float("inf"), None, None

    for idx, params in enumerate(configs, 1):
        model = model_cls(**params, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, **fit_kwargs)
        preds = model.predict(X_val)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))

        marker    = " ◄ best" if rmse < best_rmse else ""
        param_str = "  ".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"  {idx:>6}  {param_str:<60}  {rmse:.3f}{marker}")

        if rmse < best_rmse:
            best_rmse, best_model, best_params = rmse, model, params

    logger.info(f"Best params: {best_params}  val RMSE: {best_rmse:.3f}")
    print(f"    best val RMSE: {best_rmse:.3f}  params: {best_params}")
    return best_model, best_params


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    preds = X if model is None else model.predict(X)   # None → astro baseline
    result = {
        "rmse":  np.sqrt(mean_squared_error(y, preds)),
        "mae":   mean_absolute_error(y, preds),
        "r2":    r2_score(y, preds),
        "mape":  mape(y, preds),
        "preds": preds,
    }
    for q in QUANTILES:
        result[f"ql_q{int(q*100):02d}"] = quantile_loss(y, preds, q)
    return result


# ── Plots ─────────────────────────────────────────────────────────────────────
# All figures are square (fit IEEE single-column) and plotted against physical
# quantities (cm) or model iterations — never against calendar dates.

def plot_predicted_vs_observed(y_true, y_pred, model_name, horizon, out_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.scatter(y_true, y_pred, s=3, alpha=0.3, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_box_aspect(1)
    ax.set_xlabel("Observed tide (cm)")
    ax.set_ylabel("Predicted tide (cm)")
    ax.set_title(f"{model_name} — t+{horizon}h")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"predicted_vs_observed_{model_name.lower()}_t{horizon}h.png", dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name, horizon, out_dir):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.scatter(y_pred, residuals, s=3, alpha=0.3, color="#4C72B0")
    ax.axhline(0, color="red", linewidth=1, linestyle="--")
    ax.set_box_aspect(1)
    ax.set_xlabel("Predicted tide (cm)")
    ax.set_ylabel("Residual, observed − predicted (cm)")
    ax.set_title(f"{model_name} — t+{horizon}h")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"residuals_{model_name.lower()}_t{horizon}h.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feat_names, model_name, horizon, out_dir, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.barh(range(top_n), imp[idx], color="#4C72B0", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(np.array(feat_names)[idx], fontsize=7)
    ax.set_box_aspect(1)
    ax.set_xlabel("Importance (a.u.)")
    ax.set_title(f"{model_name} — t+{horizon}h  |  Top {top_n} features")
    ax.grid(True, alpha=0.3, axis="x")
    fig.savefig(out_dir / f"importance_{model_name.lower()}_t{horizon}h.png", dpi=150)
    plt.close(fig)


def plot_learning_curve_xgb(model, model_name, horizon, out_dir):
    """Train/val RMSE (cm) vs boosting round, read from XGBoost's own eval history."""
    evals = model.evals_result()
    keys = list(evals.keys())              # validation_0 = train, validation_1 = val
    metric = list(evals[keys[0]].keys())[0]
    train_curve = evals[keys[0]][metric]
    val_curve = evals[keys[1]][metric] if len(keys) > 1 else None

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(train_curve, label="Train", color="#4C72B0")
    if val_curve is not None:
        ax.plot(val_curve, label="Validation", color="orange")
    ax.set_box_aspect(1)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(f"{metric.upper()} (cm)")
    ax.set_title(f"{model_name} — t+{horizon}h  |  Learning curve (best model)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"learning_curve_{model_name.lower()}_t{horizon}h.png", dpi=150)
    plt.close(fig)


def plot_learning_curve_rf(params, X_tr, y_tr, X_val, y_val, model_name, horizon,
                           out_dir, n_steps=15):
    """Train/val RMSE (cm) vs number of trees, built incrementally via warm_start."""
    n_estimators_final = params.get("n_estimators", 100)
    step = max(1, n_estimators_final // n_steps)
    tree_counts = list(range(step, n_estimators_final + 1, step))
    if tree_counts[-1] != n_estimators_final:
        tree_counts.append(n_estimators_final)

    rf_params = {k: v for k, v in params.items() if k != "n_estimators"}
    model = RandomForestRegressor(**rf_params, n_estimators=step, warm_start=True,
                                   random_state=42, n_jobs=-1)
    train_rmse, val_rmse = [], []
    for n in tree_counts:
        model.set_params(n_estimators=n)
        model.fit(X_tr, y_tr)
        train_rmse.append(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))))
        val_rmse.append(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.plot(tree_counts, train_rmse, label="Train", color="#4C72B0", marker="o", markersize=3)
    ax.plot(tree_counts, val_rmse, label="Validation", color="orange", marker="o", markersize=3)
    ax.set_box_aspect(1)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("RMSE (cm)")
    ax.set_title(f"{model_name} — t+{horizon}h  |  Learning curve (best model)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / f"learning_curve_{model_name.lower()}_t{horizon}h.png", dpi=150)
    plt.close(fig)


# ── Logging helpers ───────────────────────────────────────────────────────────

def _log_metrics(tag, horizon, metrics, logger):
    ql_str = "  ".join(
        f"QL{int(q*100):02d}={metrics[f'ql_q{int(q*100):02d}']:.3f}"
        for q in QUANTILES
    )
    msg = (
        f"  [{tag}] t+{horizon}h  "
        f"RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
        f"R²={metrics['r2']:.4f}  MAPE={metrics['mape']:.2f}%  {ql_str}"
    )
    print(msg)
    logger.info(msg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    run_dir    = datetime.now().strftime("%Y%m%d-%H")
    xgb_logger = setup_logger("xgb", run_dir)
    rf_logger  = setup_logger("rf",  run_dir)

    print("[load] splits")
    train, val, test = load_splits()
    feat_names = feature_cols(train)

    all_metrics = []
    best_overall = {"rmse": float("inf")}  # tracks single best model across all horizons/algos

    for horizon in HORIZONS:
        print(f"\n{'─'*50}")
        print(f"Horizon: t+{horizon}h")
        print(f"{'─'*50}")

        # ── Astro baseline (horizon 1 only) ───────────────────────────────────
        if horizon == 1:
            X_te, y_te    = get_xy(test, horizon, astro=True)
            astro_metrics = evaluate(None, X_te, y_te)
            _log_metrics("Astro", horizon, astro_metrics, xgb_logger)
            plot_predicted_vs_observed(y_te, astro_metrics["preds"].ravel(),
                                       "Astro", horizon, PLOTS_DIR)
            plot_residuals(y_te, astro_metrics["preds"].ravel(),
                           "Astro", horizon, PLOTS_DIR)
            row = {
                "model": "Astro", "horizon": "t+0h",
                "rmse": astro_metrics["rmse"], "mae": astro_metrics["mae"],
                "r2": astro_metrics["r2"],     "mape": astro_metrics["mape"],
            }
            for q in QUANTILES:
                row[f"ql_q{int(q*100):02d}"] = astro_metrics[f"ql_q{int(q*100):02d}"]
            all_metrics.append(row)

        X_tr,  y_tr  = get_xy(train, horizon)
        X_val, y_val = get_xy(val,   horizon)
        X_te,  y_te  = get_xy(test,  horizon)

        # ── XGBoost ───────────────────────────────────────────────────────────
        n_xgb = len(list(ParameterGrid(XGB_GRID)))
        print(f"  [XGB] grid search ({n_xgb} configs) — see logs/{run_dir}/xgb.log")
        xgb_logger.info(f"{'═'*60}")
        xgb_logger.info(f"Horizon t+{horizon}h")
        xgb_logger.info(f"{'═'*60}")

        xgb_model, xgb_best_params = grid_search(
            XGBRegressor, XGB_GRID, X_tr, y_tr, X_val, y_val,
            logger=xgb_logger,
            fit_kwargs={"eval_set": [(X_tr, y_tr), (X_val, y_val)], "verbose": False},
        )

        xgb_metrics = evaluate(xgb_model, X_te, y_te)
        _log_metrics("XGB test", horizon, xgb_metrics, xgb_logger)

        xgb_model.save_model(MODELS_DIR / f"xgb_t{horizon}h.json")
        plot_predicted_vs_observed(y_te, xgb_metrics["preds"], "XGB", horizon, PLOTS_DIR)
        plot_residuals(y_te, xgb_metrics["preds"], "XGB", horizon, PLOTS_DIR)
        plot_feature_importance(xgb_model, feat_names, "XGB", horizon, PLOTS_DIR)

        row = {
            "model": "XGB", "horizon": f"t+{horizon}h",
            "rmse": xgb_metrics["rmse"], "mae": xgb_metrics["mae"],
            "r2":   xgb_metrics["r2"],   "mape": xgb_metrics["mape"],
        }
        for q in QUANTILES:
            row[f"ql_q{int(q*100):02d}"] = xgb_metrics[f"ql_q{int(q*100):02d}"]
        all_metrics.append(row)

        if xgb_metrics["rmse"] < best_overall["rmse"]:
            best_overall = {"rmse": xgb_metrics["rmse"], "model": xgb_model,
                            "model_name": "XGB", "horizon": horizon, "params": xgb_best_params}

        # ── Random Forest ─────────────────────────────────────────────────────
        n_rf = len(list(ParameterGrid(RF_GRID)))
        print(f"  [RF]  grid search ({n_rf} configs) — see logs/{run_dir}/rf.log")
        rf_logger.info(f"{'═'*60}")
        rf_logger.info(f"Horizon t+{horizon}h")
        rf_logger.info(f"{'═'*60}")

        rf_model, rf_best_params = grid_search(
            RandomForestRegressor, RF_GRID, X_tr, y_tr, X_val, y_val,
            logger=rf_logger,
        )

        rf_metrics = evaluate(rf_model, X_te, y_te)
        _log_metrics("RF  test", horizon, rf_metrics, rf_logger)

        joblib.dump(rf_model, MODELS_DIR / f"rf_t{horizon}h.joblib")
        plot_predicted_vs_observed(y_te, rf_metrics["preds"], "RF", horizon, PLOTS_DIR)
        plot_residuals(y_te, rf_metrics["preds"], "RF", horizon, PLOTS_DIR)
        plot_feature_importance(rf_model, feat_names, "RF", horizon, PLOTS_DIR)

        row = {
            "model": "RF", "horizon": f"t+{horizon}h",
            "rmse": rf_metrics["rmse"], "mae": rf_metrics["mae"],
            "r2":   rf_metrics["r2"],   "mape": rf_metrics["mape"],
        }
        for q in QUANTILES:
            row[f"ql_q{int(q*100):02d}"] = rf_metrics[f"ql_q{int(q*100):02d}"]
        all_metrics.append(row)

        if rf_metrics["rmse"] < best_overall["rmse"]:
            best_overall = {"rmse": rf_metrics["rmse"], "model": rf_model,
                            "model_name": "RF", "horizon": horizon, "params": rf_best_params}

    # ── Learning curve for the single best model across all horizons/algos ─────
    bh = best_overall["horizon"]
    X_tr, y_tr = get_xy(train, bh)
    X_val, y_val = get_xy(val, bh)
    print(f"\n[best model] {best_overall['model_name']} t+{bh}h  test RMSE={best_overall['rmse']:.3f}")
    if best_overall["model_name"] == "XGB":
        plot_learning_curve_xgb(best_overall["model"], "XGB", bh, PLOTS_DIR)
    else:
        plot_learning_curve_rf(best_overall["params"], X_tr, y_tr, X_val, y_val,
                               "RF", bh, PLOTS_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    csv_path   = MODELS_DIR / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)

    print(f"\n{'─'*50}")
    print("Summary")
    print("─" * 50)
    print(metrics_df.to_string(index=False))
    print(f"\n[saved] {csv_path}")
    print(f"[saved] models and plots in {MODELS_DIR}")
    print(f"[logs]  logs/{run_dir}/")


if __name__ == "__main__":
    main()