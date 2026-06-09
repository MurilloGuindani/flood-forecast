"""
Baseline model training: XGBoost and Random Forest.
One model per horizon (t+1h, t+6h, t+24h) x 2 algorithms = 6 models total.

Outputs (data/models/):
  - xgb_t{h}h.json / rf_t{h}h.joblib     trained models
  - metrics.csv                            RMSE, MAE, R² per model/horizon
  - plots/predicted_vs_observed_t{h}h.png
  - plots/residuals_t{h}h.png
"""

# claude-sonnet-4-20250514

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
MODELS_DIR    = PROJECT_ROOT / "data" / "models"
PLOTS_DIR     = MODELS_DIR / "plots"

HORIZONS      = [1, 6, 24]

XGB_GRID = {
    "n_estimators":     [300, 600],
    "max_depth":        [4, 6],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
}

RF_GRID = {
    "n_estimators": [200, 400],
    "max_depth":    [10, 20, None],
    "max_features": [0.5, "sqrt"],
}
# ─────────────────────────────────────────────────────────────────────────────


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(FEATURES_DIR / "split_train.parquet")
    val   = pd.read_parquet(FEATURES_DIR / "split_val.parquet")
    test  = pd.read_parquet(FEATURES_DIR / "split_test.parquet")
    return train, val, test


def get_xy(df: pd.DataFrame, horizon: int, astro=False) -> tuple[np.ndarray, np.ndarray]:
    target_cols = [f"target_t+{h}h" for h in HORIZONS]
    feature_cols = [c for c in df.columns
                    if c != "datetime" and c not in target_cols]
    if astro:
        feature_cols = ["tide_astro_cm"]
    X = df[feature_cols].values
    y = df[f"target_t+{horizon}h"].values
    return X, y


def feature_cols(df: pd.DataFrame) -> list[str]:
    target_cols = [f"target_t+{h}h" for h in HORIZONS]
    return [c for c in df.columns if c != "datetime" and c not in target_cols]


# ── Grid search on val set ────────────────────────────────────────────────────

def grid_search(model_cls, param_grid: dict, X_tr, y_tr, X_val, y_val,
                fit_kwargs: dict = {}) -> tuple[object, dict]:
    best_rmse   = float("inf")
    best_model  = None
    best_params = None

    for params in ParameterGrid(param_grid):
        model = model_cls(**params, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, **fit_kwargs)
        preds = model.predict(X_val)
        rmse  = np.sqrt(mean_squared_error(y_val, preds))
        if rmse < best_rmse:
            best_rmse   = rmse
            best_model  = model
            best_params = params

    print(f"    best params: {best_params}  val RMSE: {best_rmse:.3f}")
    return best_model, best_params


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    if not model:
        preds = X# astronomical tide
    else:
        preds = model.predict(X)
    return {
        "rmse": np.sqrt(mean_squared_error(y, preds)),
        "mae":  mean_absolute_error(y, preds),
        "r2":   r2_score(y, preds),
        "preds": preds,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_predicted_vs_observed(dates, y_true, y_pred, model_name, horizon, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    fig.suptitle(f"{model_name} — t+{horizon}h  |  Predicted vs Observed",
                 fontsize=13, fontweight="bold")

    # Time series
    ax = axes[0]
    ax.plot(dates, y_true, linewidth=0.7, label="Observed", color="#4C72B0", alpha=0.9)
    ax.plot(dates, y_pred, linewidth=0.7, label="Predicted", color="orange", alpha=0.8)
    ax.set_ylabel("Tide (cm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Scatter
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, s=2, alpha=0.3, color="#4C72B0")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax2.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
    ax2.set_xlabel("Observed (cm)")
    ax2.set_ylabel("Predicted (cm)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"predicted_vs_observed_{model_name.lower()}_t{horizon}h.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_residuals(dates, y_true, y_pred, model_name, horizon, out_dir):
    residuals = y_true - y_pred

    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle(f"{model_name} — t+{horizon}h  |  Residuals",
                 fontsize=13, fontweight="bold")

    # Residuals over time
    ax = axes[0]
    ax.plot(dates, residuals, linewidth=0.6, color="#D9534F", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(dates, residuals, 0,
                    where=residuals > 0, alpha=0.3, color="#D9534F")
    ax.fill_between(dates, residuals, 0,
                    where=residuals < 0, alpha=0.3, color="#4C72B0")
    ax.set_ylabel("Residual (cm)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.axvline(residuals.mean(), color="red", linestyle="--", linewidth=1,
                label=f"Mean {residuals.mean():.2f} cm")
    ax2.axvline(residuals.mean() + residuals.std(), color="orange",
                linestyle=":", linewidth=1, label=f"±1σ {residuals.std():.2f} cm")
    ax2.axvline(residuals.mean() - residuals.std(), color="orange",
                linestyle=":", linewidth=1)
    ax2.set_xlabel("Residual (cm)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Residuals vs predicted (heteroscedasticity check)
    ax3 = axes[2]
    ax3.scatter(y_pred, residuals, s=2, alpha=0.3, color="#4C72B0")
    ax3.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax3.set_xlabel("Predicted (cm)")
    ax3.set_ylabel("Residual (cm)")
    ax3.set_title("Residuals vs Predicted")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"residuals_{model_name.lower()}_t{horizon}h.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feat_names, model_name, horizon, out_dir, top_n=20):
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return

    idx  = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), imp[idx], color="#4C72B0", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(np.array(feat_names)[idx], fontsize=7)
    ax.set_title(f"{model_name} — t+{horizon}h  |  Top {top_n} features")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = out_dir / f"importance_{model_name.lower()}_t{horizon}h.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] splits")
    train, val, test = load_splits()
    feat_names = feature_cols(train)

    all_metrics = []

    for horizon in HORIZONS:
        print(f"\n{'─'*50}")
        print(f"Horizon: t+{horizon}h")
        print(f"{'─'*50}")

        # --- Astro baseline
        if horizon==1:
            X_te,  y_te  = get_xy(test,  horizon, True)
            test_dates = pd.to_datetime(test["datetime"].values)
            astro_metrics = evaluate(None, X_te, y_te)
            print(f"  [Astro]  test  RMSE={astro_metrics['rmse']:.3f}  "
                    f"MAE={astro_metrics['mae']:.3f}  R²={astro_metrics['r2']:.4f}")

            plot_predicted_vs_observed(test_dates, y_te, astro_metrics["preds"],
                                        "Astro", horizon, PLOTS_DIR)
            plot_residuals(test_dates, y_te, astro_metrics["preds"].ravel(),
                            "Astro", horizon, PLOTS_DIR)

            all_metrics.append({
                "model": "Astro", "horizon": f"0h",
                "rmse": astro_metrics["rmse"], "mae": astro_metrics["mae"],
                "r2": astro_metrics["r2"],
            })

        X_tr,  y_tr  = get_xy(train, horizon)
        X_val, y_val = get_xy(val,   horizon)
        X_te,  y_te  = get_xy(test,  horizon)

        test_dates = pd.to_datetime(test["datetime"].values)


        # ── XGBoost ───────────────────────────────────────────────────────────
        print(f"  [XGB] grid search ({len(list(ParameterGrid(XGB_GRID)))} configs)")
        xgb_model, _ = grid_search(
            XGBRegressor, XGB_GRID,
            X_tr, y_tr, X_val, y_val,
            fit_kwargs={"eval_set": [(X_val, y_val)], "verbose": False},
        )

        xgb_metrics = evaluate(xgb_model, X_te, y_te)
        print(f"  [XGB] test  RMSE={xgb_metrics['rmse']:.3f}  "
              f"MAE={xgb_metrics['mae']:.3f}  R²={xgb_metrics['r2']:.4f}")

        xgb_model.save_model(MODELS_DIR / f"xgb_t{horizon}h.json")
        plot_predicted_vs_observed(test_dates, y_te, xgb_metrics["preds"],
                                   "XGB", horizon, PLOTS_DIR)
        plot_residuals(test_dates, y_te, xgb_metrics["preds"],
                       "XGB", horizon, PLOTS_DIR)
        plot_feature_importance(xgb_model, feat_names, "XGB", horizon, PLOTS_DIR)

        all_metrics.append({
            "model": "XGB", "horizon": f"t+{horizon}h",
            "rmse": xgb_metrics["rmse"], "mae": xgb_metrics["mae"],
            "r2": xgb_metrics["r2"],
        })

        # ── Random Forest ─────────────────────────────────────────────────────
        print(f"  [RF]  grid search ({len(list(ParameterGrid(RF_GRID)))} configs)")
        rf_model, _ = grid_search(
            RandomForestRegressor, RF_GRID,
            X_tr, y_tr, X_val, y_val,
        )

        rf_metrics = evaluate(rf_model, X_te, y_te)
        print(f"  [RF]  test  RMSE={rf_metrics['rmse']:.3f}  "
              f"MAE={rf_metrics['mae']:.3f}  R²={rf_metrics['r2']:.4f}")

        joblib.dump(rf_model, MODELS_DIR / f"rf_t{horizon}h.joblib")
        plot_predicted_vs_observed(test_dates, y_te, rf_metrics["preds"],
                                   "RF", horizon, PLOTS_DIR)
        plot_residuals(test_dates, y_te, rf_metrics["preds"],
                       "RF", horizon, PLOTS_DIR)
        plot_feature_importance(rf_model, feat_names, "RF", horizon, PLOTS_DIR)

        all_metrics.append({
            "model": "RF", "horizon": f"t+{horizon}h",
            "rmse": rf_metrics["rmse"], "mae": rf_metrics["mae"],
            "r2": rf_metrics["r2"],
        })



    # ── Summary ───────────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(MODELS_DIR / "metrics.csv", index=False)

    print(f"\n{'─'*50}")
    print("Summary")
    print('─'*50)
    print(metrics_df.to_string(index=False))
    print(f"\n[saved] models and plots in {MODELS_DIR}")


if __name__ == "__main__":
    main()