
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "data" / "models"
PLOTS_DIR    = MODELS_DIR / "plots"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"
HORIZONS   = [1]



XGB_GRID = {
    "n_estimators":     [300, 600],
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha":        [0.0, 0.1, 1.0],   # L1
    "reg_lambda":       [1.0, 5.0, 10.0],  # L2
}

RF_GRID = {
    "n_estimators": [200, 400],
    "max_depth":    [10, 20, None],
    "max_features": [0.5, "sqrt"],
    "min_samples_leaf": [1, 5, 10],  # implicit regularization
    "max_samples":      [0.7, 0.9],  # bagging fraction
}

# NNs
# Hyper-parameter grid — searched on VAL set
PARAM_GRID = {
    "hidden_size": [4, 8, 64],
    "num_layers":  [1, 2,  4],
    "dropout":     [0.1, 0.2, 0.3],
    "lr": [1e-3,  1e-5],
    "weight_decay": [0.0, 1e-4,],
}


BATCH_SIZE = 64
EPOCHS     = 250
SEED       = 42
# ─────────────────────────────────────────────────────────────────────────────