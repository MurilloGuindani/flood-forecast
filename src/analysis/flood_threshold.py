# claude-sonnet-4-6

"""
Flood threshold analysis.

For each known flood event, extracts:
  - max observed tide in ±6h window
  - accumulated precipitation in 24h, 48h, 72h windows before event

Outputs boxplots comparing flood vs non-flood distributions.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import json
from pathlib import Path

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
OUT_DIR =PLOTS_DIR
TARGET_STATION_GLOB = "2951_*.parquet"
PRECIP_FILE         = CLEANED_DIR / "inmet_A806_A806_FLORIANOPOLIS.parquet"

PRECIP_WINDOWS_H = [6, 12, 24]  # accumulation windows before event
FLOOD_WINDOW_H = 12

FLOOD_EVENTS = ['29/07/2025',
                '29/01/2025',
                '29/12/2025',#
                '29/04/2025',
                '11/05/2026',#
                '14/03/2025',
                '24/06/2025',#
                '29/05/2025',#
                '04/01/2026',
                '30/03/2025',#
                '28/04/2025',
                '16/01/2025',#
                '11/12/2025',#
                '17/01/2025',
                '25/05/2024']
# ─────────────────────────────────────────────────────────────────────────────


# Aggregation level: "hourly" (every hour gets a label) or
# "daily_max" (one row per day, using the day's max tide level).
# daily_max is recommended: floods are day-level events and this avoids
# pseudo-replication / autocorrelation inflating significance.
AGGREGATION = "daily_max"

# Probability cutoffs to report as candidate operational thresholds
PROB_CUTOFFS = [0.10, 0.25, 0.50, 0.75]


# ── Load data ─────────────────────────────────────────────────────────────

def load_target(cleaned_dir: Path) -> pd.Series:
    matches = sorted(cleaned_dir.glob(TARGET_STATION_GLOB))
    if not matches:
        raise FileNotFoundError(f"Target station not found: {TARGET_STATION_GLOB}")
    df = pd.read_parquet(matches[0], columns=["datetime", "tide_level_cm"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index().resample("1h").mean()
    return df["tide_level_cm"]


def build_flood_labels(index: pd.DatetimeIndex, flood_events: list[str],
                        window_h: int) -> pd.Series:
    """Binary label per timestamp: 1 if within ±window_h of a flood date."""
    flood_dates = pd.to_datetime(flood_events, format="%d/%m/%Y")
    label = pd.Series(0, index=index, dtype=int)
    for d in flood_dates:
        mask = (index >= d - pd.Timedelta(hours=window_h)) & \
               (index <= d + pd.Timedelta(days=1) + pd.Timedelta(hours=window_h))
        label.loc[mask] = 1
    return label


# ── Main analysis ────────────────────────────────────────────────────────

def main() -> None:
    target = load_target(CLEANED_DIR)
    flood_label = build_flood_labels(target.index, FLOOD_EVENTS, FLOOD_WINDOW_H)

    df = pd.DataFrame({"tide_level_cm": target, "flood": flood_label})
    df = df.dropna(subset=["tide_level_cm"])

    if AGGREGATION == "daily_max":
        daily = df.copy()
        daily["date"] = daily.index.date
        agg = daily.groupby("date").agg(
            tide_level_cm=("tide_level_cm", "max"),
            flood=("flood", "max"),
        ).reset_index(drop=True)
        df = agg

    n_total = len(df)
    n_flood = int(df["flood"].sum())
    print(f"[data] rows={n_total}  flood-positive={n_flood}  "
          f"({100 * n_flood / n_total:.2f}%)  aggregation={AGGREGATION}")

    if n_flood < 5:
        print("[warn] very few positive labels — coefficient estimates will be unstable")

    # ── Fit class-weighted logistic regression ─────────────────────────────
    X = sm.add_constant(df["tide_level_cm"])
    y = df["flood"]

    # statsmodels GLM doesn't support class_weight directly; emulate
    # "balanced" weighting via observation weights (freq_weights approx).
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    w_pos = len(y) / (2 * n_pos)
    w_neg = len(y) / (2 * n_neg)
    weights = np.where(y == 1, w_pos, w_neg)

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=weights)
    result = model.fit()

    print(result.summary())

    coef = result.params["tide_level_cm"]
    ci_low, ci_high = result.conf_int().loc["tide_level_cm"]
    pval = result.pvalues["tide_level_cm"]
    intercept = result.params["const"]

    print(f"\n[coefficient] tide_level_cm: {coef:.4f}  "
          f"95% CI [{ci_low:.4f}, {ci_high:.4f}]  p={pval:.2e}")

    # Odds ratio per cm and per 10cm
    print(f"[odds ratio]  per 1cm:  {np.exp(coef):.4f}")
    print(f"[odds ratio]  per 10cm: {np.exp(coef * 10):.4f}")

    # ── Derive thresholds from probability cutoffs ──────────────────────────
    # logit(p) = intercept + coef * tide  =>  tide = (logit(p) - intercept) / coef
    thresholds = {}
    for p in PROB_CUTOFFS:
        logit_p = np.log(p / (1 - p))
        tide_at_p = (logit_p - intercept) / coef
        thresholds[f"p={p}"] = round(float(tide_at_p), 2)
        print(f"  P(flood)={p:.2f}  ->  tide_level_cm ≈ {tide_at_p:.1f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    tide_range = np.linspace(df["tide_level_cm"].min() - 10,
                              df["tide_level_cm"].max() + 10, 200)
    X_range = sm.add_constant(pd.DataFrame({"tide_level_cm": tide_range}))
    probs = result.predict(X_range)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tide_range, probs, color="#1f77b4", label="P(flood | tide level)")
    ax.scatter(df["tide_level_cm"], df["flood"], alpha=0.3, color="black",
               s=15, label="observed (jittered)")
    for p, t in thresholds.items():
        ax.axvline(t, linestyle="--", color="gray", alpha=0.6)
        ax.text(t, 0.02, f"{p}\n{t:.0f}cm", rotation=90, fontsize=8,
                ha="right", va="bottom")
    ax.set_xlabel("Tide level (cm)")
    ax.set_ylabel("P(flood)")
    ax.set_title(f"Flood probability vs tide level ({AGGREGATION})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "logistic_flood_curve.png", dpi=150)
    print(f"\n[saved] {OUT_DIR / 'logistic_flood_curve.png'}")

    # ── Save results ────────────────────────────────────────────────────
    results = {
        "aggregation": AGGREGATION,
        "n_total": n_total,
        "n_flood": n_flood,
        "flood_window_hours": FLOOD_WINDOW_H,
        "coefficient_tide_level_cm": float(coef),
        "coefficient_ci95": [float(ci_low), float(ci_high)],
        "p_value": float(pval),
        "intercept": float(intercept),
        "odds_ratio_per_cm": float(np.exp(coef)),
        "odds_ratio_per_10cm": float(np.exp(coef * 10)),
        "thresholds_by_probability": thresholds,
        "interpretation": (
            "p < 0.05 indicates a statistically significant association "
            "between tide level and flood occurrence. The odds ratio per cm "
            "shows how much flood odds multiply for each 1cm increase in "
            "tide level. Thresholds are model-implied tide levels at which "
            "predicted flood probability reaches the given cutoff — these "
            "are NOT guarantees, just the regression's estimated risk curve."
        ),
    }
    with open(OUT_DIR / "logistic_flood_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {OUT_DIR / 'logistic_flood_results.json'}")


if __name__ == "__main__":
    main()
