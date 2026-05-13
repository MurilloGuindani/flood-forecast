"""
Tide table transformation pipeline.

Steps:
  1. Load tide_table.parquet, convert m → cm
  2. Cubic-spline interpolate to hourly grid
  3. Align to station 15 datetime index
  4. Version A: remove yearly low-frequency component (sun tide) via low-cut filter
  5. Version B: reconstruct 2024 sun-tide from 2025-2026 observations and add it back

Outputs (all in data/features/):
  - tide_hourly_cm.parquet          raw hourly in cm
  - tide_aligned.parquet            aligned to station 15 index
  - tide_no_sun.parquet             Version A: sun tide removed
  - tide_with_reconstructed_sun.parquet  Version B: 2024 sun tide restored
"""

# claude-sonnet-4-20250514

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "cleaned"
FEATURES_DIR = PROJECT_ROOT / "data" / "processed"

TIDE_PARQUET = PROCESSED_DIR / "tide_table.parquet"

# Station 15 parquet — used as the alignment target datetime index
# Adjust filename to match the actual station 15 parquet
STATION_15_GLOB = "2951_*"   # change if station 15 has a different prefix

# Sun-tide period: ~365.25 days — we use a low-pass cutoff slightly above this
# to capture the annual + semi-annual signal
SUN_TIDE_CUTOFF_DAYS = 60    # remove everything slower than 60 days
# (captures annual, semi-annual, seasonal)

SAMPLE_RATE_H = 1            # hourly data
# ─────────────────────────────────────────────────────────────────────────────


# ── 1. Load & convert ─────────────────────────────────────────────────────────

def load_tide(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("datetime").drop_duplicates(
        "datetime").reset_index(drop=True)
    df["tide_level_cm"] = df["tide_level_m"] * 100
    return df


# ── 2. Cubic-spline interpolation to hourly ───────────────────────────────────

def interpolate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    x = df["datetime"].astype("int64").values / 1e9
    y = df["tide_level_cm"].values

    cs = CubicSpline(x, y)

    hourly_dt = pd.date_range(
        start=df["datetime"].min().floor("h"),
        end=df["datetime"].max().ceil("h"),
        freq="1h",
    )
    x_h = hourly_dt.astype("int64").values / 1e9

    return pd.DataFrame({"datetime": hourly_dt, "tide_cm": cs(x_h)})


# ── 3. Align to station 15 ────────────────────────────────────────────────────

def align_to_station(tide_hourly: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    matches = sorted(processed_dir.glob(STATION_15_GLOB + ".parquet"))
    if not matches:
        print(
            f"[warn] no station 15 parquet found matching '{STATION_15_GLOB}' — skipping alignment")
        return tide_hourly

    station_df = pd.read_parquet(matches[0])
    station_dt = pd.to_datetime(
        station_df["datetime"]).drop_duplicates().sort_values()

    # Reindex tide to station timestamps via nearest-neighbour (within 30 min)
    tide_indexed = tide_hourly.set_index("datetime")
    aligned = tide_indexed.reindex(
        station_dt, method="nearest", tolerance="30min")
    aligned = aligned.reset_index().rename(columns={"index": "datetime"})
    aligned = aligned.dropna(subset=["tide_cm"]).reset_index(drop=True)

    print(
        f"[align] station 15 rows: {len(station_dt)}  aligned tide rows: {len(aligned)}")
    return aligned


# ── 4. Low-pass / high-pass Butterworth filter ────────────────────────────────

def butter_lowpass(data: np.ndarray, cutoff_hours: float, fs: float = 1.0, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    normal_cutoff = (1 / cutoff_hours) / nyq
    normal_cutoff = min(normal_cutoff, 0.999)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def extract_sun_tide(tide_cm: np.ndarray, cutoff_days: float = SUN_TIDE_CUTOFF_DAYS) -> np.ndarray:
    """Extract the low-frequency yearly component (sun tide)."""
    cutoff_hours = cutoff_days * 24
    return butter_lowpass(tide_cm, cutoff_hours, fs=SAMPLE_RATE_H)


# ── 5. Reconstruct 2024 sun tide from 2025–2026 ───────────────────────────────

def reconstruct_2024_sun_tide(tide_hourly: pd.DataFrame) -> pd.DataFrame:
    df = tide_hourly.copy()
    df["year"] = df["datetime"].dt.year

    # Extract sun tide from each year separately
    sun_all = extract_sun_tide(df["tide_cm"].values)
    df["sun_tide"] = sun_all

    # Build mean annual profile from 2025-2026 only
    mask_ref = df["year"].isin([2025, 2026])
    ref = df[mask_ref].copy()
    ref["hoy"] = (ref["datetime"].dt.day_of_year - 1) * \
        24 + ref["datetime"].dt.hour
    annual_profile = ref.groupby("hoy")["sun_tide"].mean()

    # For 2024: replace its extracted sun tide with the reconstructed one
    mask_2024 = df["year"] == 2024
    hoy_2024 = ((df.loc[mask_2024, "datetime"].dt.day_of_year - 1) * 24
                + df.loc[mask_2024, "datetime"].dt.hour)

    reconstructed = hoy_2024.map(annual_profile).fillna(method="ffill")

    result = df.copy()
    # Remove existing 2024 sun tide, add reconstructed
    result.loc[mask_2024, "tide_cm"] = (
        df.loc[mask_2024, "tide_cm"]
        # remove original weak sun tide
        - df.loc[mask_2024, "sun_tide"]
        + reconstructed.values                    # add reconstructed from 2025-2026
    )
    result["reconstructed_sun_cm"] = 0.0
    result.loc[mask_2024, "reconstructed_sun_cm"] = reconstructed.values

    return result

# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_versions(hourly: pd.DataFrame, no_sun: pd.DataFrame, reconstructed: pd.DataFrame,
                  out_dir: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(22, 18), sharex=True)
    fig.suptitle("Tide processing versions", fontsize=14, fontweight="bold")

    for ax, df, title, col in zip(
        axes[:3],
        [hourly, no_sun, reconstructed],
        ["Raw hourly (cm)", "Version A — sun tide removed",
         "Version B — 2024 sun tide reconstructed"],
        ["tide_cm", "tide_cm", "tide_cm"],
    ):
        ax.plot(df["datetime"], df[col], linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("cm")
        ax.grid(True, alpha=0.4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Solar tide panel
    ax4 = axes[3]
    # Original extracted sun tide (all years)
    sun_original = extract_sun_tide(hourly["tide_cm"].values)
    ax4.plot(hourly["datetime"], sun_original, linewidth=1.2,
             label="Extracted (all years)", color="#4C72B0")

    # Reconstructed 2024 sun tide
    if "reconstructed_sun_cm" in reconstructed.columns:
        mask = reconstructed["datetime"].dt.year == 2024
        ax4.plot(
            reconstructed.loc[mask, "datetime"],
            reconstructed.loc[mask, "reconstructed_sun_cm"],
            linewidth=1.2, linestyle="--", label="Reconstructed 2024", color="red"
        )

    ax4.axvline(pd.Timestamp("2025-01-01"),
                color="gray", linestyle=":", linewidth=1)
    ax4.set_title("Solar tide component", fontsize=11)
    ax4.set_ylabel("cm")
    ax4.grid(True, alpha=0.4)
    ax4.legend(fontsize=9)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    out = out_dir / "tide_versions.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] {out}")

# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("[load] tide_table.parquet")
    raw = load_tide(TIDE_PARQUET)
    print(
        f"       {len(raw)} observations, {raw['datetime'].min()} → {raw['datetime'].max()}")

    # 2. Interpolate
    print("[interpolate] cubic spline → hourly")
    hourly = interpolate_hourly(raw)
    hourly.to_parquet(FEATURES_DIR / "tide_hourly_cm.parquet", index=False)
    print(f"[saved] tide_hourly_cm.parquet  ({len(hourly)} rows)")

    # 3. Align to station 15
    print("[align] to station 15 datetime index")
    aligned = align_to_station(hourly, PROCESSED_DIR)
    aligned.to_parquet(FEATURES_DIR / "tide_aligned.parquet", index=False)
    print(f"[saved] tide_aligned.parquet  ({len(aligned)} rows)")

    # 4. Version A — remove sun tide
    print("[filter] extracting and removing sun tide")
    tide_values = hourly["tide_cm"].values
    sun_tide = extract_sun_tide(tide_values)
    no_sun = hourly.copy()
    no_sun["sun_tide_cm"] = sun_tide
    no_sun["tide_cm"] = tide_values - sun_tide
    no_sun.to_parquet(FEATURES_DIR / "tide_no_sun.parquet", index=False)
    print(f"[saved] tide_no_sun.parquet")

    # 5. Version B
    print("[reconstruct] estimating 2024 sun tide from 2025–2026")
    with_sun = reconstruct_2024_sun_tide(hourly)
    with_sun.to_parquet(
        FEATURES_DIR / "tide_with_reconstructed_sun.parquet", index=False)
    print(f"[saved] tide_with_reconstructed_sun.parquet")

    # 6. Plot
    plot_versions(hourly, no_sun, with_sun, FEATURES_DIR)
    print("[done]")


if __name__ == "__main__":
    main()
