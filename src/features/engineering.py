"""
Feature extraction pipeline for tide level prediction.

Produces two artifacts:
  - ml_features_flat.parquet   : flat feature vector per timestep
  - ml_features_sequence.npz  : sliding window tensors (X, y) for recurrent/generative models

Input features:
  - Astronomical tide (known future at inference)
  - Weather: pressure, wind U/V from coastal stations
  - Cyclical time encodings

Targets: observed tide at t+1h, t+6h, t+24h
"""

# claude-sonnet-4-20250514 & GPT 5.5

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


from typing import Dict, List, Tuple

import numpy as np
import pandas as pd



# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
CLEANED_DIR   = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR  = PROJECT_ROOT / "data" / "features"

TARGET_STATION_GLOB = "2951_*.parquet"
TIDE_HOURLY         = CLEANED_DIR / "tide_with_reconstructed_sun.parquet"
# TIDE_HOURLY         = CLEANED_DIR / "tide_with_reconstructed_sun.parquet"
TIDE_RESIDUAL       = CLEANED_DIR / "tide_residual.parquet"
FLOOD_EVENTS = ['29/07/2025',
                '29/01/2025',
                #'29/12/2025',
                '29/04/2025',
                # '11/05/2026',
                '14/03/2025',
                #'24/06/2025',
                #'29/05/2025',
                '04/01/2026',
                #'30/03/2025',
                '28/04/2025',
                #'16/01/2025',
                #'11/12/2025',
                '17/01/2025']



# Lookback window for sequence models (hours)
LOOKBACK_H    = 72

# Forecast horizons (hours ahead)
HORIZONS      = [1]

# Lag steps for flat features
LAG_STEPS     = [1, 2, 3, 6, 12, 24, 48, 72]

# Rolling windows for flat features
ROLL_WINDOWS  = [3, 6, 12, 24, 48, 72]

# Weather variables to extract per station
WEATHER_VARS  = [
    "atm_pressure_mb",
    "wind_speed_avg_ms",
    "wind_dir_avg_deg",
]
# ─────────────────────────────────────────────────────────────────────────────


def cyclical_encode(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    sin = np.sin(2 * np.pi * series / period)
    cos = np.cos(2 * np.pi * series / period)
    return sin, cos


def wind_components(speed: pd.Series, direction_deg: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Decompose wind into U (east) and V (north) components.
    Positive V = wind FROM south → storm surge risk for Florianópolis."""
    rad = np.radians(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


# Feature selection


def remove_redundant_features(
    df: pd.DataFrame,
    std_threshold: float = 0.01,
    corr_threshold: float = 0.95,
    missing_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Remove redundant features and return a report explaining why.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    std_threshold : float
        Features with std < threshold are removed.
    corr_threshold : float
        One feature from each pair with abs(corr) > threshold is removed.
    missing_threshold : float
        Features with missing fraction > threshold are removed.

    Returns
    -------
    filtered_df : pd.DataFrame
    report : dict
        Dictionary containing removed features by criterion.
    """

    X = df.copy()

    report = {
        "constant_features": [],
        "near_zero_variance": [],
        "high_missingness": [],
        "duplicate_columns": [],
        "high_correlation": [],
    }

    # ------------------------------------------------------------------
    # 1. Constant features
    # ------------------------------------------------------------------
    constant_cols = [
        c for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]

    report["constant_features"] = constant_cols
    X = X.drop(columns=constant_cols)

    # ------------------------------------------------------------------
    # 2. Near-zero variance (numeric only)
    # ------------------------------------------------------------------
    numeric_cols = X.select_dtypes(include=np.number).columns

    low_var_cols = [
        c for c in numeric_cols
        if X[c].std(skipna=True) < std_threshold
    ]

    report["near_zero_variance"] = low_var_cols
    X = X.drop(columns=low_var_cols)

    # ------------------------------------------------------------------
    # 3. High missingness
    # ------------------------------------------------------------------
    missing_cols = [
        c for c in X.columns
        if X[c].isna().mean() > missing_threshold
    ]

    report["high_missingness"] = missing_cols
    X = X.drop(columns=missing_cols)

    # ------------------------------------------------------------------
    # 4. Duplicate columns
    # ------------------------------------------------------------------
    duplicate_cols = []

    cols = list(X.columns)

    for i in range(len(cols)):
        col_i = cols[i]

        if col_i in duplicate_cols:
            continue

        for j in range(i + 1, len(cols)):
            col_j = cols[j]

            if X[col_i].equals(X[col_j]):
                duplicate_cols.append(col_j)

    report["duplicate_columns"] = duplicate_cols
    X = X.drop(columns=duplicate_cols)
    # ------------------------------------------------------------------
    # 5. High correlation
    # Keep first feature, remove later ones.
    # ------------------------------------------------------------------

    numeric_cols = X.select_dtypes(include=np.number).columns

    corr_matrix = X[numeric_cols].corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    corr_remove = []
    corr_report = []

    for col in upper.columns:

        correlated_with = upper.index[upper[col] > corr_threshold]

        if len(correlated_with) > 0:

            kept_col = correlated_with[0]

            corr_value = float(
                upper.loc[kept_col, col]
            )

            corr_remove.append(col)

            corr_report.append(
                {
                    "removed": col,
                    "kept": kept_col,
                    "correlation": round(corr_value, 4),
                }
            )

    report["high_correlation"] = corr_report

    X = X.drop(columns=corr_remove)

    return X, report

# ── Load target ───────────────────────────────────────────────────────────────

def load_target(cleaned_dir: Path) -> pd.Series:
    matches = sorted(cleaned_dir.glob(TARGET_STATION_GLOB))
    if not matches:
        raise FileNotFoundError(f"Target station not found: {TARGET_STATION_GLOB}")
    df = pd.read_parquet(matches[0], columns=["datetime", "tide_level_cm"])
    print(df.columns)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index().resample("1h").mean()
    print(f"[target] {matches[0].name}  {len(df)} rows  "
          f"{df.index.min()} → {df.index.max()}")
    return df["tide_level_cm"]


def build_index(target: pd.Series) -> pd.DatetimeIndex:
    return pd.date_range(
        target.index.min().floor("h"),
        target.index.max().ceil("h"),
        freq="1h",
    )



# ── Tide features ─────────────────────────────────────────────────────────────

def tide_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)

    if not TIDE_HOURLY.exists():
        print(f"[warn] {TIDE_HOURLY} not found — skipping astronomical tide features")
        return out

    astro = (pd.read_parquet(TIDE_HOURLY)
             .set_index("datetime")["tide_cm"]
             .reindex(index, method="nearest", tolerance="30min"))

    out["tide_astro_cm"]        = astro
    out["tide_velocity_cm_h"]   = astro.diff(1)
    out["tide_accel_cm_h2"]     = astro.diff(1).diff(1)

    # Hours to next predicted high tide
    vals = astro.fillna(method="ffill").values
    peaks, _ = find_peaks(vals, distance=6)
    peak_times = index[peaks]
    next_high = np.full(len(index), np.nan)
    for i, t in enumerate(index):
        future = peak_times[peak_times > t]
        if len(future):
            next_high[i] = min((future[0] - t).total_seconds() / 3600, 13)
    out["hours_to_high_tide"] = next_high

    # Known future astronomical tide at each horizon (available at inference time)
    for h in HORIZONS:
        out[f"tide_astro_t+{h}h"] = astro.shift(-h)

    # Residual
    if TIDE_RESIDUAL.exists():
        res = (pd.read_parquet(TIDE_RESIDUAL)
               .set_index("datetime")[["residual_cm"]]
               .reindex(index, method="nearest", tolerance="30min"))
        out["tide_residual_cm"] = res["residual_cm"]

    return out


# ── Weather features ──────────────────────────────────────────────────────────

def load_station(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    cols = [c for c in WEATHER_VARS if c in df.columns]
    if not cols:
        return pd.DataFrame(index=index)
    return df[cols].resample("1h").mean().reindex(index)


def weather_features(index: pd.DatetimeIndex, cleaned_dir: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=index)

    target_names  = {p.name for p in cleaned_dir.glob(TARGET_STATION_GLOB)}
    station_files = [p for p in sorted(cleaned_dir.glob("*.parquet"))
                     if p.name not in target_names]

    all_pressure = []
    all_wind_u   = []
    all_wind_v   = []

    for path in station_files:
        print(path)
        sid = path.stem[:30]
        try:
            sdf = load_station(path, index)
        except Exception as e:
            print(f"[warn] {path.stem}: {e}")
            continue

        has_wind = ("wind_speed_avg_ms" in sdf.columns and
                    "wind_dir_avg_deg"  in sdf.columns)

        if has_wind:
            u, v = wind_components(sdf["wind_speed_avg_ms"], sdf["wind_dir_avg_deg"])
            sdf["wind_u"] = u
            sdf["wind_v"] = v
            all_wind_u.append(u.rename(sid))
            all_wind_v.append(v.rename(sid))

        work_cols = []
        if "atm_pressure_mb" in sdf.columns:
            work_cols.append("atm_pressure_mb")
            all_pressure.append(sdf["atm_pressure_mb"].rename(sid))
        if has_wind:
            work_cols += ["wind_u", "wind_v"]

        for var in work_cols:
            col = sdf[var]
            out[f"{sid}__{var}"] = col

            for lag in LAG_STEPS:
                out[f"{sid}__{var}__lag{lag}h"] = col.shift(lag)

            for window in ROLL_WINDOWS:
                out[f"{sid}__{var}__roll{window}h_mean"] = col.rolling(window, min_periods=1).mean()
                out[f"{sid}__{var}__roll{window}h_std"]  = col.rolling(window, min_periods=1).std()

            out[f"{sid}__{var}__diff1h"]  = col.diff(1)
            out[f"{sid}__{var}__diff6h"]  = col.diff(6)
            out[f"{sid}__{var}__diff24h"] = col.diff(24)

    # ── Spatial aggregates ────────────────────────────────────────────────────
    if all_pressure:
        p = pd.concat(all_pressure, axis=1)
        out["spatial__pressure_mean"]   = p.mean(axis=1)
        out["spatial__pressure_min"]    = p.min(axis=1)
        out["spatial__pressure_grad"]   = p.max(axis=1) - p.min(axis=1)
        for d in [1, 6, 24]:
            out[f"spatial__pressure_diff{d}h"] = out["spatial__pressure_mean"].diff(d)
        for w in ROLL_WINDOWS:
            out[f"spatial__pressure_roll{w}h_mean"] = (
                out["spatial__pressure_mean"].rolling(w, min_periods=1).mean()
            )

    if all_wind_u:
        wu = pd.concat(all_wind_u, axis=1)
        wv = pd.concat(all_wind_v, axis=1)
        out["spatial__wind_u_mean"] = wu.mean(axis=1)
        out["spatial__wind_v_mean"] = wv.mean(axis=1)
        # South wind index: positive = southerly wind, primary surge driver
        out["spatial__south_wind_index"] = out["spatial__wind_v_mean"].clip(lower=0)
        for lag in LAG_STEPS:
            out[f"spatial__wind_u_lag{lag}h"] = out["spatial__wind_u_mean"].shift(lag)
            out[f"spatial__wind_v_lag{lag}h"] = out["spatial__wind_v_mean"].shift(lag)
        for w in ROLL_WINDOWS:
            out[f"spatial__wind_u_roll{w}h_mean"] = (
                out["spatial__wind_u_mean"].rolling(w, min_periods=1).mean()
            )
            out[f"spatial__wind_v_roll{w}h_mean"] = (
                out["spatial__wind_v_mean"].rolling(w, min_periods=1).mean()
            )

    return out


# ── Assemble flat dataset ─────────────────────────────────────────────────────

def assemble_flat(index: pd.DatetimeIndex, target: pd.Series,
                  cleaned_dir: Path) -> pd.DataFrame:
    feat = tide_features(index)
    feat = feat.join(weather_features(index, cleaned_dir))

    # Targets
    target_reindexed = target.reindex(index)
    for h in HORIZONS:
        feat[f"target_t+{h}h"] = target_reindexed.shift(-h)

    feat = feat.reset_index().rename(columns={"index": "datetime"})
    feat.insert(0, "datetime", feat.pop("datetime"))

    target_cols = [f"target_t+{h}h" for h in HORIZONS]

    return feat.dropna(subset=target_cols).reset_index(drop=True)


# ── Build sequence tensors ────────────────────────────────────────────────────

def build_sequences(flat: pd.DataFrame) -> dict:
    """
    Sliding window → (N, LOOKBACK_H, F) input tensor, (N, 3) target tensor.

    X[i] = features for hours [i, i+LOOKBACK_H)
    y[i] = [tide at t+1h, t+6h, t+24h] for the prediction point i+LOOKBACK_H
    """
    target_cols  = [f"target_t+{h}h" for h in HORIZONS]
    feature_cols = [c for c in flat.columns
                    if c != "datetime" and c not in target_cols
                    and ('roll' not in c or 'roll1h' in c)
                    and ('lag' not in c or 'lag1h' in c)]

    print(feature_cols)

    flat_clean   = flat.dropna(subset=feature_cols).reset_index(drop=True)
    feat_arr     = flat_clean[feature_cols].values.astype(np.float32)
    target_arr   = flat_clean[target_cols].values.astype(np.float32)
    times        = flat_clean["datetime"].values

    N = len(flat_clean) - LOOKBACK_H
    if N <= 0:
        raise ValueError(f"Not enough rows ({len(flat_clean)}) for lookback {LOOKBACK_H}h")

    X = np.stack([feat_arr[i: i + LOOKBACK_H] for i in range(N)])
    y = target_arr[LOOKBACK_H:]
    t = times[LOOKBACK_H:]

    print(f"[sequence] X: {X.shape}  y: {y.shape}  features: {len(feature_cols)}")
    return {"X": X, "y": y, "times": t, "feature_names": np.array(feature_cols)}


def split_dataset(flat: pd.DataFrame,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train/val/test split:
      - 2024: all goes to train (reconstructed sun tide)
      - 2025+2026: randomly sampled into train/val/test
                   preserving temporal blocks to avoid leakage

    Sampling is block-based (weekly blocks) to preserve
    short-term autocorrelation within each split.
    """
    rng = np.random.default_rng(random_state)

    df = flat.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    train_2024 = df[df["datetime"].dt.year == 2024].copy()
    rest       = df[df["datetime"].dt.year >= 2025].copy()
    # Convert flood dates to week block ids
    flood_dates = pd.to_datetime(FLOOD_EVENTS, format="%d/%m/%Y")

    flood_blocks = set(
        flood_dates.to_series().apply(
            lambda d: f"{d.isocalendar().year}_{d.isocalendar().week:02d}"
        )
    )
    # ── Block sampling on rest (weekly blocks) ────────────────────────────────
    rest["_week_block"] = (
        rest["datetime"].dt.isocalendar().year.astype(str) + "_" +
        rest["datetime"].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    blocks = rest["_week_block"].unique()
    rng.shuffle(blocks)

    # Remove flood blocks from random allocation
    available_blocks = [b for b in blocks if b not in flood_blocks]

    n_val = int(len(blocks) * val_ratio)
    n_test = int(len(blocks) * test_ratio)

    val_blocks = set(available_blocks[:n_val])
    test_blocks = set(available_blocks[n_val:n_val + n_test])
    train_blocks = set(available_blocks[n_val + n_test:])

    # Force flood weeks into test
    test_blocks.update(flood_blocks)
    # Safety: ensure flood weeks are not elsewhere
    train_blocks.difference_update(flood_blocks)
    val_blocks.difference_update(flood_blocks)

    rest_train = rest[rest["_week_block"].isin(train_blocks)].copy()
    val        = rest[rest["_week_block"].isin(val_blocks)].copy()
    test       = rest[rest["_week_block"].isin(test_blocks)].copy()

    train = pd.concat([train_2024, rest_train], ignore_index=True)

    # Drop helper column
    for d in [train, val, test]:
        d.drop(columns=["_week_block"], inplace=True, errors="ignore")

    train = train.sort_values("datetime").reset_index(drop=True)
    val   = val.sort_values("datetime").reset_index(drop=True)
    test  = test.sort_values("datetime").reset_index(drop=True)

    # ── Stats check ───────────────────────────────────────────────────────────
    print("\n[split summary]")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        months = sorted(d["datetime"].dt.month.unique())
        target = d["target_t+1h"]
        print(f"  {name:5s}  rows={len(d):6d}  "
              f"mean={target.mean():6.1f}  std={target.std():5.1f}  "
              f"months={months}")

    return train, val, test



def build_and_split(flat: pd.DataFrame, val_ratio=0.20, test_ratio=0.10):
    # Sort first — critical
    flat = flat.sort_values("datetime").reset_index(drop=True)

    seq = build_sequences(flat)  # build on full contiguous data
    X, y, times = seq["X"], seq["y"], seq["times"]

    N = len(X)
    n_test = int(N * test_ratio)
    n_val  = int(N * val_ratio)
    n_train = N - n_val - n_test

    return (
        (X[:n_train],            y[:n_train]),
        (X[n_train:n_train+n_val], y[n_train:n_train+n_val]),
        (X[n_train+n_val:],      y[n_train+n_val:]),
    )

def split_chronological(
    flat: pd.DataFrame,
    flood_events: list[str],
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    block_days: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/val/test split.
    Flood event blocks (±block_days window) are always forced into test.
    """
    flat = flat.sort_values("datetime").reset_index(drop=True)

    flood_dates = pd.to_datetime(flood_events, format="%d/%m/%Y")

    # Block ID: days since epoch // block_days
    flat["_block"] = (
        flat["datetime"].dt.floor("D").astype(np.int64) // (86400 * 1e9 * block_days)
    ).astype(int)

    flood_blocks = set()
    for d in flood_dates:
        block_id = int(pd.Timestamp(d).floor("D").value // (86400 * 1e9 * block_days))
        flood_blocks.add(block_id)

    flood_mask = flat["_block"].isin(flood_blocks)
    flood_df   = flat[flood_mask].copy()
    rest_df    = flat[~flood_mask].copy().reset_index(drop=True)

    N       = len(rest_df)
    n_test  = int(N * test_ratio)
    n_val   = int(N * val_ratio)
    n_train = N - n_val - n_test

    train = rest_df.iloc[:n_train].drop(columns=["_block"])
    val   = rest_df.iloc[n_train:n_train + n_val].drop(columns=["_block"])
    test  = (
        pd.concat([rest_df.iloc[n_train + n_val:], flood_df])
        .sort_values("datetime")
        .reset_index(drop=True)
        .drop(columns=["_block"])
    )

    for d in flood_dates:
        in_test = (pd.to_datetime(test["datetime"]).dt.date == d.date()).any()
        print(f"  flood {d.date()} → {'test ✓' if in_test else 'MISSING ✗'}")

    print(f"\n[split] train={len(train)}  val={len(val)}  test={len(test)}")
    return train, val, test

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] target station 2951")
    target = load_target(CLEANED_DIR)
    index  = build_index(target)

    # ── Flat ──────────────────────────────────────────────────────────────────
    print("\n── Flat feature set ──")
    flat = assemble_flat(index, target, CLEANED_DIR)

    target_cols  = [f"target_t+{h}h" for h in HORIZONS]
    feature_cols = [c for c in flat.columns
                    if c != "datetime" and c not in target_cols]

    missing   = flat[feature_cols].isna().mean()
    drop_cols = missing[missing > 0.4].index.tolist()
    flat      = flat.drop(columns=drop_cols)

    fill_cols = [c for c in feature_cols if c not in drop_cols]
    flat[fill_cols] = flat[fill_cols].fillna(
        flat[fill_cols].rolling(window=4, min_periods=1).mean()
    )

    print(f"  rows:         {len(flat)}")
    print(f"  features:     {len([c for c in flat.columns if c not in target_cols and c != 'datetime'])}")
    print(f"  dropped cols: {len(drop_cols)}")

    flat.to_parquet(FEATURES_DIR / "ml_features_flat.parquet", index=False)
    print("[saved] ml_features_flat.parquet")

    flat, report = remove_redundant_features(flat)

    for reason, cols in report.items():
        print(f"\n{reason}: {len(cols)}")
        if reason != "high_correlation":
            for c in cols:
                print(f"  - {c}")
    for item in report["high_correlation"]:
        print(f"Removed {item['removed']} (corr={item['correlation']:.3f}) with {item['kept']}")

    train, val, test = split_chronological(flat, FLOOD_EVENTS)

    train.to_parquet(FEATURES_DIR / "split_train.parquet", index=False)
    val.to_parquet(FEATURES_DIR / "split_val.parquet",     index=False)
    test.to_parquet(FEATURES_DIR / "split_test.parquet",   index=False)
    print("[saved] split_train / split_val / split_test")

    # ── Sequence ──────────────────────────────────────────────────────────────
    print("\n── Sequence tensors ──")
    seq = build_sequences(train)
    np.savez_compressed(
        FEATURES_DIR / "ml_features_sequence_tr.npz",
        X=seq["X"],
        y=seq["y"],
        times=seq["times"].astype(str),
        feature_names=seq["feature_names"],
    )
    print(f"[saved] ml_features_sequence_tr.npz")
    print(f"\n[done]  X={seq['X'].shape}  y={seq['y'].shape}")

    seq = build_sequences(val)
    np.savez_compressed(
        FEATURES_DIR / "ml_features_sequence_val.npz",
        X=seq["X"],
        y=seq["y"],
        times=seq["times"].astype(str),
        feature_names=seq["feature_names"],
    )
    print(f"[saved] ml_features_sequence_val.npz")
    print(f"\n[done]  X={seq['X'].shape}  y={seq['y'].shape}")

    seq = build_sequences(test)
    np.savez_compressed(
        FEATURES_DIR / "ml_features_sequence_te.npz",
        X=seq["X"],
        y=seq["y"],
        times=seq["times"].astype(str),
        feature_names=seq["feature_names"],
    )
    print(f"[saved] ml_features_sequence_te.npz")
    print(f"\n[done]  X={seq['X'].shape}  y={seq['y'].shape}")
    flood_dates = pd.to_datetime(FLOOD_EVENTS, format="%d/%m/%Y")

    # get observed tide at each flood event
    target = load_target(CLEANED_DIR)
    flood_levels = []
    for d in flood_dates:
        # get max tide in ±12h window around the event
        window = target[
            (target.index >= d - pd.Timedelta(hours=12)) &
            (target.index <= d + pd.Timedelta(hours=12))
        ]
        if len(window):
            flood_levels.append(window.max())

    flood_levels = pd.Series(flood_levels)
    print(flood_levels.describe())
    print(f"\nsuggested threshold: {flood_levels.min():.1f} cm  (min of known floods)")


if __name__ == "__main__":
    main()