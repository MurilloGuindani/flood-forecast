# claude-sonnet-4-6

"""
Flood threshold analysis.

For each known flood event, extracts:
  - max observed tide in ±window around event datetime
  - logistic regression to derive flood probability thresholds
  - multivariate model isolating independent effect of precipitation

Outputs logistic curve plot and JSON results.
Prints days where tide exceeded each probability threshold (outlier inspection).
"""

from __future__ import annotations
from pathlib import Path

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR  = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR    = PROJECT_ROOT / "data" / "analysis" / "plots"
OUT_DIR      = PLOTS_DIR
TARGET_STATION_GLOB = "2951_*.parquet"
PRECIP_FILE         = CLEANED_DIR / "inmet_A806_A806_FLORIANOPOLIS.parquet"

# Precipitation accumulation windows (hours before event)
PRECIP_WINDOWS_H = [24, 48, 72]
# Cap for precipitation outlier winsorizing (mm)
PRECIP_CAP_MM = 80.0

# Flood events with exact datetimes (date, time, type)
FLOOD_EVENTS = [
    ("15/05/2026", "02:10", "Partial Flooding"),
    ("11/05/2026", "11:20", "Partial Flooding"),
    ("04/01/2026", "14:54", "Partial Flooding"),
    ("29/12/2025", "20:11", "Flooding"),
    ("11/12/2025", "14:56", "Partial Flooding"),
    ("29/07/2025", "06:15", "Flooding"),
    ("24/06/2025", "14:29", "Partial Flooding"),
    ("29/05/2025", "17:17", "Partial Flooding"),
    ("29/04/2025", "18:00", "Flooding"),
    ("28/04/2025", "18:00", "Flooding"),
    ("28/04/2025", "15:41", "Partial Flooding"),
    ("30/03/2025", "16:18", "Partial Flooding"),
    ("14/03/2025", "15:58", "Partial Flooding"),
    ("29/01/2025", "00:00", "Flooding"),
    ("29/01/2025", "12:00", "Partial Flooding"),
    ("17/01/2025", "07:00", "Partial Flooding"),
    ("16/01/2025", "06:00", "Partial Flooding"),
    ("16/01/2025", "06:10", "Flooding"),
    ("25/05/2024", "16:00", "Partial Flooding"),
    ("15/04/2024", "08:30", "Partial Flooding"),
]

FLOOD_WINDOW_H   = 6
FLOOD_BLACKOUT_H = 72
SUBSAMPLE_FRAC   = 1.0
AGGREGATION      = "daily_max"
PROB_CUTOFFS     = [0.50, 0.75, 0.90, 0.95]
# ─────────────────────────────────────────────────────────────────────────────


def load_target(cleaned_dir: Path) -> pd.Series:
    matches = sorted(cleaned_dir.glob(TARGET_STATION_GLOB))
    if not matches:
        raise FileNotFoundError(f"Target station not found: {TARGET_STATION_GLOB}")
    df = pd.read_parquet(matches[0], columns=["datetime", "tide_level_cm"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index().resample("6h").mean()
    return df["tide_level_cm"]


def load_precip(precip_file: Path) -> pd.Series:
    """Load hourly precipitation, set datetime column as index, return as Series."""
    df = pd.read_parquet(precip_file, columns=["datetime", "precipitation_mm"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    return df["precipitation_mm"].resample("1h").sum()


def build_precip_windows(precip: pd.Series, index: pd.DatetimeIndex,
                          windows_h: list[int]) -> pd.DataFrame:
    """For each timestamp in index, compute accumulated precip over past N hours."""
    result = {}
    for w in windows_h:
        rolled = precip.rolling(window=w, min_periods=1).sum()
        result[f"precip_{w}h"] = rolled.reindex(index, method="nearest")
    return pd.DataFrame(result, index=index)


def parse_flood_events(events: list[tuple]) -> list[pd.Timestamp]:
    timestamps = []
    for date_str, time_str, _ in events:
        dt = pd.to_datetime(f"{date_str} {time_str}", format="%d/%m/%Y %H:%M")
        timestamps.append(dt)
    return timestamps


def build_flood_labels(index: pd.DatetimeIndex, flood_timestamps: list[pd.Timestamp],
                        window_h: int) -> pd.Series:
    label = pd.Series(0, index=index, dtype=int)
    for ts in flood_timestamps:
        mask = (index >= ts - pd.Timedelta(hours=window_h)) & \
               (index <= ts + pd.Timedelta(hours=window_h))
        label.loc[mask] = 1
    return label


def print_outliers(df: pd.DataFrame, thresholds: dict) -> None:
    return
    if hasattr(df.index, 'date'):
        flood_dates = set(df.index[df["flood"] == 1].normalize())
    else:
        flood_dates = set(df[df["flood"] == 1].index)

    print("\n" + "=" * 60)
    print("OUTLIER INSPECTION — high-tide non-flood days")
    print("=" * 60)
    for label, tide_cm in sorted(thresholds.items(), key=lambda x: x[1]):
        above = df[df["tide_level_cm"] >= tide_cm].copy()
        above = above[~above.index.normalize().isin(flood_dates)]
        if above.empty:
            print(f"\n[{label} >= {tide_cm:.0f}cm]  No non-flood days above threshold.")
            continue
        daily = above["tide_level_cm"].groupby(above.index.normalize()).max().sort_values(ascending=False)
        print(f"\n[{label} >= {tide_cm:.0f}cm]  {len(daily)} non-flood day(s):")
        for d, v in daily.items():
            print(f"  {d.date()}  max_tide={v:.1f}cm")


def fit_and_report(y, X, label: str) -> sm.GLM:
    """Fit balanced-weight logistic regression and print summary."""
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    w_pos = len(y) / (2 * n_pos)
    w_neg = len(y) / (2 * n_neg)
    weights = np.where(y == 1, w_pos, w_neg)
    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()
    print(f"\n{'─'*60}")
    print(f"MODEL: {label}")
    print(f"{'─'*60}")
    print(result.summary())
    return result


def main() -> None:
    target = load_target(CLEANED_DIR)
    flood_timestamps = parse_flood_events(FLOOD_EVENTS)
    flood_label = build_flood_labels(target.index, flood_timestamps, FLOOD_WINDOW_H)

    # Load and align precipitation
    precip_available = PRECIP_FILE.exists()
    if precip_available:
        precip = load_precip(PRECIP_FILE)
        precip_windows = build_precip_windows(precip, target.index, PRECIP_WINDOWS_H)
    else:
        print("[warn] precipitation file not found — skipping multivariate model")

    df = pd.DataFrame({"tide_level_cm": target, "flood": flood_label})
    if precip_available:
        df = df.join(precip_windows)
    df = df.dropna(subset=["tide_level_cm"])

    # Blackout ±72h around flood events
    blackout = pd.Series(False, index=target.index)
    for ts in flood_timestamps:
        mask = (target.index >= ts - pd.Timedelta(hours=FLOOD_BLACKOUT_H)) & \
               (target.index <= ts + pd.Timedelta(hours=FLOOD_BLACKOUT_H))
        blackout.loc[mask] = True

    flood_rows    = df[df["flood"] == 1]
    non_flood_all = df[(df["flood"] == 0) & ~blackout]
    non_flood_sub = non_flood_all.sample(frac=SUBSAMPLE_FRAC, random_state=42)
    df = pd.concat([flood_rows, non_flood_sub]).sort_index()

    if AGGREGATION == "daily_max":
        daily = df.copy()
        daily["date"] = daily.index.date
        agg_dict = {"tide_level_cm": "max", "flood": "max"}
        if precip_available:
            for w in PRECIP_WINDOWS_H:
                agg_dict[f"precip_{w}h"] = "max"
        df = daily.groupby("date").agg(agg_dict)
        df.index = pd.to_datetime(df.index)

    # ── Winsorize precipitation outlier ───────────────────────────────────
    if precip_available:
        for w in PRECIP_WINDOWS_H:
            col = f"precip_{w}h"
            n_capped = (df[col] > PRECIP_CAP_MM).sum()
            if n_capped:
                print(f"[precip] winsorizing {n_capped} value(s) > {PRECIP_CAP_MM}mm in {col}")
            df[col] = df[col].clip(upper=PRECIP_CAP_MM)

    n_total = len(df)
    n_flood = int(df["flood"].sum())
    print(f"[data] rows={n_total}  flood-positive={n_flood}  "
          f"({100 * n_flood / n_total:.2f}%)  aggregation={AGGREGATION}  "
          f"window=±{FLOOD_WINDOW_H}h")

    if n_flood < 5:
        print("[warn] very few positive labels — estimates may be unstable")

    y = df["flood"]

    # ── Model 1: tide only (baseline) ─────────────────────────────────────
    X_tide = sm.add_constant(df["tide_level_cm"])
    result_tide = fit_and_report(y, X_tide, "Tide only (baseline)")

    coef      = result_tide.params["tide_level_cm"]
    ci_low, ci_high = result_tide.conf_int().loc["tide_level_cm"]
    pval      = result_tide.pvalues["tide_level_cm"]
    intercept = result_tide.params["const"]

    print(f"\n[coefficient] tide_level_cm: {coef:.4f}  "
          f"95% CI [{ci_low:.4f}, {ci_high:.4f}]  p={pval:.2e}")
    print(f"[odds ratio]  per 1cm:  {np.exp(coef):.4f}")
    print(f"[odds ratio]  per 10cm: {np.exp(coef * 10):.4f}")

    # ── Thresholds from tide-only model ───────────────────────────────────
    thresholds = {}
    print("\n[thresholds — tide only]")
    for p in PROB_CUTOFFS:
        logit_p = np.log(p / (1 - p))
        tide_at_p = (logit_p - intercept) / coef
        thresholds[f"p={p}"] = round(float(tide_at_p), 2)
        print(f"  P(flood)={p:.2f}  ->  tide_level_cm ≈ {tide_at_p:.1f}")

    p90 = float(df["tide_level_cm"].quantile(0.90))
    print(f"\n[90th percentile] tide_level_cm = {p90:.1f}cm")
    thresholds["p90_percentile"] = round(p90, 2)

    # ── Model 2: tide + precipitation (multivariate) ───────────────────────
    if precip_available:
        # Choose best precipitation window by AIC
        best_aic, best_w, best_result_mv = np.inf, None, None
        for w in PRECIP_WINDOWS_H:
            col = f"precip_{w}h"
            sub = df[["tide_level_cm", col]].dropna()
            y_sub = y.loc[sub.index]
            X_mv = sm.add_constant(sub)
            n_pos_s = int(y_sub.sum()); n_neg_s = len(y_sub) - n_pos_s
            wts = np.where(y_sub == 1, len(y_sub)/(2*n_pos_s), len(y_sub)/(2*n_neg_s))
            r = sm.GLM(y_sub, X_mv, family=sm.families.Binomial(), freq_weights=wts).fit()
            print(f"  [AIC] tide + precip_{w}h: {r.aic:.1f}")
            if r.aic < best_aic:
                best_aic, best_w, best_result_mv = r.aic, w, r

        print(f"\n[best precipitation window] precip_{best_w}h  (AIC={best_aic:.1f})")
        result_mv = fit_and_report(
            y.loc[df[["tide_level_cm", f"precip_{best_w}h"]].dropna().index],
            sm.add_constant(df[["tide_level_cm", f"precip_{best_w}h"]].dropna()),
            f"Tide + precip_{best_w}h (multivariate)"
        )

        coef_tide_mv  = result_mv.params["tide_level_cm"]
        coef_rain_mv  = result_mv.params[f"precip_{best_w}h"]
        pval_tide_mv  = result_mv.pvalues["tide_level_cm"]
        pval_rain_mv  = result_mv.pvalues[f"precip_{best_w}h"]

        print(f"\n[isolation of rain effect]")
        print(f"  tide_level_cm coef (tide-only):       {coef:.4f}  p={pval:.2e}")
        print(f"  tide_level_cm coef (w/ rain control): {coef_tide_mv:.4f}  p={pval_tide_mv:.2e}")
        delta = coef_tide_mv - coef
        print(f"  change in tide coef after adding rain: {delta:+.4f}  "
              f"({'confounding' if abs(delta)/abs(coef) > 0.10 else 'minimal confounding'})")
        print(f"  precip_{best_w}h coef: {coef_rain_mv:.4f}  p={pval_rain_mv:.2e}  "
              f"OR per 10mm: {np.exp(coef_rain_mv*10):.3f}")

    # ── Outlier inspection ─────────────────────────────────────────────────
    print_outliers(df, thresholds)

    # ── Plot ───────────────────────────────────────────────────────────────
    tide_range = np.linspace(df["tide_level_cm"].min() - 10,
                              df["tide_level_cm"].max() + 10, 300)
    X_range = sm.add_constant(pd.DataFrame({"tide_level_cm": tide_range}))
    probs = result_tide.predict(X_range)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tide_range, probs, color="#1f77b4", label="P(flood | tide level)")
    ax.scatter(df["tide_level_cm"], df["flood"], alpha=0.3, color="black",
               s=15, label="observed")
    for p, t in thresholds.items():
        ax.axvline(t, linestyle="--", color="gray", alpha=0.6)
        ax.text(t, 0.02, f"{p}\n{t:.0f}cm", rotation=90, fontsize=8,
                ha="right", va="bottom")
    ax.set_xlabel("Tide level (cm)")
    ax.set_ylabel("P(flood)")
    ax.set_title(f"Flood probability vs tide level ({AGGREGATION}, ±{FLOOD_WINDOW_H}h window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "logistic_flood_curve.png", dpi=150)
    print(f"\n[saved] {OUT_DIR / 'logistic_flood_curve.png'}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    results = {
        "aggregation": AGGREGATION,
        "flood_window_hours": FLOOD_WINDOW_H,
        "n_total": n_total,
        "n_flood": n_flood,
        "coefficient_tide_level_cm": float(coef),
        "coefficient_ci95": [float(ci_low), float(ci_high)],
        "p_value": float(pval),
        "intercept": float(intercept),
        "odds_ratio_per_cm": float(np.exp(coef)),
        "odds_ratio_per_10cm": float(np.exp(coef * 10)),
        "thresholds_by_probability": thresholds,
    }
    if precip_available:
        results["multivariate"] = {
            "best_precip_window_h": best_w,
            "tide_coef_with_rain": float(coef_tide_mv),
            "rain_coef": float(coef_rain_mv),
            "rain_pvalue": float(pval_rain_mv),
            "rain_OR_per_10mm": float(np.exp(coef_rain_mv * 10)),
            "tide_coef_change_pct": float(100 * delta / abs(coef)),
        }
    with open(OUT_DIR / "logistic_flood_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'logistic_flood_results.json'}")


if __name__ == "__main__":
    main()


# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR  = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR    = PROJECT_ROOT / "data" / "analysis" / "plots"
OUT_DIR      = PLOTS_DIR
TARGET_STATION_GLOB = "2951_*.parquet"

# Flood events with exact datetimes (date, time, type)
FLOOD_EVENTS = [
    ("15/05/2026", "02:10", "Partial Flooding"),
    ("11/05/2026", "11:20", "Partial Flooding"),
    ("04/01/2026", "14:54", "Partial Flooding"),
    ("29/12/2025", "20:11", "Flooding"),
    ("11/12/2025", "14:56", "Partial Flooding"),
    ("29/07/2025", "06:15", "Flooding"),
    ("24/06/2025", "14:29", "Partial Flooding"),
    ("29/05/2025", "17:17", "Partial Flooding"),
    ("29/04/2025", "18:00", "Flooding"),
    ("28/04/2025", "18:00", "Flooding"),
    ("28/04/2025", "15:41", "Partial Flooding"),
    ("30/03/2025", "16:18", "Partial Flooding"),
    ("14/03/2025", "15:58", "Partial Flooding"),
    ("29/01/2025", "00:00", "Flooding"),
    ("29/01/2025", "12:00", "Partial Flooding"),
    ("17/01/2025", "07:00", "Partial Flooding"),
    ("16/01/2025", "06:00", "Partial Flooding"),
    ("16/01/2025", "06:10", "Flooding"),
    ("25/05/2024", "16:00", "Partial Flooding"),
    ("15/04/2024", "08:30", "Partial Flooding"),
]

# Tighter window: ±3h around exact event time
FLOOD_WINDOW_H = 6
FLOOD_BLACKOUT_H = 72  # hours to exclude around each event (not used for labeling)
SUBSAMPLE_FRAC = 0.30   # fraction of non-flood rows to keep

# Aggregation: "hourly" or "daily_max"
AGGREGATION = "hourly"

# Probability cutoffs for threshold derivation
PROB_CUTOFFS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
# ─────────────────────────────────────────────────────────────────────────────


def load_target(cleaned_dir: Path) -> pd.Series:
    matches = sorted(cleaned_dir.glob(TARGET_STATION_GLOB))
    if not matches:
        raise FileNotFoundError(f"Target station not found: {TARGET_STATION_GLOB}")
    df = pd.read_parquet(matches[0], columns=["datetime", "tide_level_cm"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index().resample("6h").mean()
    return df["tide_level_cm"]


def parse_flood_events(events: list[tuple]) -> list[pd.Timestamp]:
    """Parse (date, time, type) tuples into Timestamps."""
    timestamps = []
    for date_str, time_str, _ in events:
        dt = pd.to_datetime(f"{date_str} {time_str}", format="%d/%m/%Y %H:%M")
        timestamps.append(dt)
    return timestamps


def build_flood_labels(index: pd.DatetimeIndex, flood_timestamps: list[pd.Timestamp],
                        window_h: int) -> pd.Series:
    """Binary label: 1 if within ±window_h hours of a flood event timestamp."""
    label = pd.Series(0, index=index, dtype=int)
    for ts in flood_timestamps:
        mask = (index >= ts - pd.Timedelta(hours=window_h)) & \
               (index <= ts + pd.Timedelta(hours=window_h))
        label.loc[mask] = 1
    return label


def print_outliers(df: pd.DataFrame, thresholds: dict) -> None:
    return
    """Print dates where tide exceeded each threshold with no flood label on that date."""
    # All dates that have at least one flood-positive sample
    if hasattr(df.index, 'date'):
        flood_dates = set(df.index[df["flood"] == 1].normalize())
    else:
        flood_dates = set(df[df["flood"] == 1].index)

    print("\n" + "=" * 60)
    print("OUTLIER INSPECTION — high-tide non-flood days")
    print("=" * 60)
    for label, tide_cm in sorted(thresholds.items(), key=lambda x: x[1]):
        above = df[df["tide_level_cm"] >= tide_cm].copy()
        # Exclude any date that has a flood label anywhere
        above = above[~above.index.normalize().isin(flood_dates)]
        if above.empty:
            print(f"\n[{label} >= {tide_cm:.0f}cm]  No non-flood days above threshold.")
            continue
        daily = above["tide_level_cm"].groupby(above.index.normalize()).max().sort_values(ascending=False)
        print(f"\n[{label} >= {tide_cm:.0f}cm]  {len(daily)} non-flood day(s):")
        for d, v in daily.items():
            print(f"  {d.date()}  max_tide={v:.1f}cm")


def main() -> None:
    target = load_target(CLEANED_DIR)
    flood_timestamps = parse_flood_events(FLOOD_EVENTS)
    flood_label = build_flood_labels(target.index, flood_timestamps, FLOOD_WINDOW_H)

    df = pd.DataFrame({"tide_level_cm": target, "flood": flood_label})
    df = df.dropna(subset=["tide_level_cm"])

    # Mark rows within ±72h of any flood event (includes label window + buffer)
    blackout = pd.Series(False, index=target.index)
    for ts in flood_timestamps:
        mask = (target.index >= ts - pd.Timedelta(hours=FLOOD_BLACKOUT_H)) & \
               (target.index <= ts + pd.Timedelta(hours=FLOOD_BLACKOUT_H))
        blackout.loc[mask] = True

    # Keep: flood-positive rows + non-blackout non-flood rows (subsampled)
    flood_rows    = df[df["flood"] == 1]
    non_flood_all = df[(df["flood"] == 0) & ~blackout]
    non_flood_sub = non_flood_all.sample(frac=SUBSAMPLE_FRAC, random_state=42)
    df = pd.concat([flood_rows, non_flood_sub]).sort_index()

    if AGGREGATION == "daily_max":
        daily = df.copy()
        daily["date"] = daily.index.date
        df = daily.groupby("date").agg(
            tide_level_cm=("tide_level_cm", "max"),
            flood=("flood", "max"),
        )
        df.index = pd.to_datetime(df.index)

    n_total = len(df)
    n_flood = int(df["flood"].sum())
    print(f"[data] rows={n_total}  flood-positive={n_flood}  "
          f"({100 * n_flood / n_total:.2f}%)  aggregation={AGGREGATION}  "
          f"window=±{FLOOD_WINDOW_H}h")

    if n_flood < 5:
        print("[warn] very few positive labels — estimates may be unstable")

    # ── Logistic regression ────────────────────────────────────────────────
    X = sm.add_constant(df["tide_level_cm"])
    y = df["flood"]

    n_pos, n_neg = y.sum(), len(y) - y.sum()
    w_pos = len(y) / (2 * n_pos)
    w_neg = len(y) / (2 * n_neg)
    weights = np.where(y == 1, w_pos, w_neg)

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()
    print(result.summary())

    coef      = result.params["tide_level_cm"]
    ci_low, ci_high = result.conf_int().loc["tide_level_cm"]
    pval      = result.pvalues["tide_level_cm"]
    intercept = result.params["const"]

    print(f"\n[coefficient] tide_level_cm: {coef:.4f}  "
          f"95% CI [{ci_low:.4f}, {ci_high:.4f}]  p={pval:.2e}")
    print(f"[odds ratio]  per 1cm:  {np.exp(coef):.4f}")
    print(f"[odds ratio]  per 10cm: {np.exp(coef * 10):.4f}")

    # ── Thresholds ─────────────────────────────────────────────────────────
    thresholds = {}
    for p in PROB_CUTOFFS:
        logit_p = np.log(p / (1 - p))
        tide_at_p = (logit_p - intercept) / coef
        thresholds[f"p={p}"] = round(float(tide_at_p), 2)
        print(f"  P(flood)={p:.2f}  ->  tide_level_cm ≈ {tide_at_p:.1f}")

    # ── Outlier inspection ─────────────────────────────────────────────────
    p90 = float(df["tide_level_cm"].quantile(0.90))
    print(f"\n[90th percentile] tide_level_cm = {p90:.1f}cm")
    thresholds["p90_percentile"] = round(p90, 2)
    print_outliers(df, thresholds)

    # ── Plot ───────────────────────────────────────────────────────────────
    tide_range = np.linspace(df["tide_level_cm"].min() - 10,
                              df["tide_level_cm"].max() + 10, 300)
    X_range = sm.add_constant(pd.DataFrame({"tide_level_cm": tide_range}))
    probs = result.predict(X_range)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tide_range, probs, color="#1f77b4", label="P(flood | tide level)")
    ax.scatter(df["tide_level_cm"], df["flood"], alpha=0.3, color="black",
               s=15, label="observed")
    for p, t in thresholds.items():
        ax.axvline(t, linestyle="--", color="gray", alpha=0.6)
        ax.text(t, 0.02, f"{p}\n{t:.0f}cm", rotation=90, fontsize=8,
                ha="right", va="bottom")
    ax.set_xlabel("Tide level (cm)")
    ax.set_ylabel("P(flood)")
    ax.set_title(f"Flood probability vs tide level ({AGGREGATION}, ±{FLOOD_WINDOW_H}h window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "logistic_flood_curve.png", dpi=150)
    print(f"\n[saved] {OUT_DIR / 'logistic_flood_curve.png'}")

    # ── Save JSON ──────────────────────────────────────────────────────────
    results = {
        "aggregation": AGGREGATION,
        "flood_window_hours": FLOOD_WINDOW_H,
        "n_total": n_total,
        "n_flood": n_flood,
        "coefficient_tide_level_cm": float(coef),
        "coefficient_ci95": [float(ci_low), float(ci_high)],
        "p_value": float(pval),
        "intercept": float(intercept),
        "odds_ratio_per_cm": float(np.exp(coef)),
        "odds_ratio_per_10cm": float(np.exp(coef * 10)),
        "thresholds_by_probability": thresholds,
    }
    with open(OUT_DIR / "logistic_flood_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'logistic_flood_results.json'}")


if __name__ == "__main__":
    main()