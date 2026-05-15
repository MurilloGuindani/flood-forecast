"""
EDA report generator for EPAGRI weather stations.
Reads processed parquets, produces descriptive stats + plots per station,
and compiles everything into a single PDF report.

Output: data/reports/eda_report.pdf
"""

# claude-sonnet-4-20250514

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy.stats import boxcox, spearmanr
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "cleaned"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
OUTPUT_PDF = REPORTS_DIR / "eda_report.pdf"

STATION_COLS = {"station_name", "station_city", "station_lat",
                "station_lon", "station_alt_m", "station_id"}
TIME_COL = "datetime"
# ─────────────────────────────────────────────────────────────────────────────


def features_table_page(pdf: PdfPages, station_meta: list[dict], dfs: list[pd.DataFrame]) -> None:
    all_features = sorted({col for df in dfs for col in numeric_cols(df)})
    
    # Sort stations by name
    paired = sorted(zip(station_meta, dfs), key=lambda x: x[0]["name"])
    station_meta, dfs = zip(*paired)
    station_names = [m["name"] for m in station_meta]

    # Build presence + missing % matrix
    matrix = []
    cell_colors = []
    for df in dfs:
        row = []
        row_colors = []
        for feature in all_features:
            if feature not in df.columns:
                row.append("")
                row_colors.append("white")
            else:
                missing_pct = df[feature].isna().mean() * 100
                if missing_pct < 1:
                    row.append("✓")
                    row_colors.append("#4C72B0")
                elif missing_pct < 5:
                    row.append("⚠")
                    row_colors.append("#FFD700")
                else:
                    row.append("✗")
                    row_colors.append("#D9534F")
        matrix.append(row)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(
        figsize=(max(6, len(all_features) * 0.8), max(6, len(station_names) * 0.5 + 4)))

    ax.axis("off")
    ax.set_title("Feature Presence per Station", fontsize=13, fontweight="bold", pad=12)

    tbl = ax.table(
    cellText=matrix,
    rowLabels=station_names,
    colLabels=all_features,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(0.9, 1.1)
    tbl.auto_set_column_width(list(range(len(all_features))))

    for (row, col), cell in tbl.get_celld().items():
        if col == -1:  # row labels (features)
            cell.set_text_props(fontsize=7)
        if row == 0:
            cell.set_text_props(rotation=70, ha="left", fontsize=7)
            cell.set_height(0.15)
        if row > 0 and col >= 0:
            color = cell_colors[row - 1][col]
            cell.set_facecolor(color)
            text_color = "white" if color in ("#4C72B0", "#D9534F") else "black"
            cell.set_text_props(color=text_color, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#4C72B0", label="< 1% missing"),
        Patch(facecolor="#FFD700", label="1–5% missing"),
        Patch(facecolor="#D9534F", label="> 5% missing"),
    ]

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.legend(handles=legend, loc="lower center", fontsize=8,
            ncol=3, bbox_to_anchor=(0.5, 0.0), framealpha=0.8)
    fig.subplots_adjust(bottom=0.15, top=0.95)
    ax.set_ylim(-0.5, len(station_names) + 0.5)
    pdf.savefig(fig)
    plt.close(fig)


def _mic_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Approximate MIC via mutual information from sklearn."""
    from sklearn.feature_selection import mutual_info_regression
    cols = data.columns.tolist()
    n = len(cols)
    mic = np.zeros((n, n))
    for i in range(n):
        mi = mutual_info_regression(
            data.values, data.iloc[:, i].values, random_state=42)
        mic[i, :] = mi
    # Normalise to [0, 1]
    mic_max = mic.max()
    if mic_max > 0:
        mic /= mic_max
    np.fill_diagonal(mic, 1.0)
    return pd.DataFrame(mic, index=cols, columns=cols)


def numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns excluding station metadata."""
    return [
        c for c in df.select_dtypes(include="number").columns
        if c not in STATION_COLS
    ]


def cover_page(pdf: PdfPages, title: str, station_files: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.5, 0.72, title, ha="center", va="center",
            fontsize=24, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.62, f"{len(station_files)} stations processed",
            ha="center", va="center", fontsize=14, color="#555555",
            transform=ax.transAxes)
    for i, f in enumerate(sorted(station_files)):
        ax.text(0.5, 0.52 - i * 0.04, f.stem, ha="center", va="center",
                fontsize=10, color="#333333", transform=ax.transAxes)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def stats_page(pdf: PdfPages, df: pd.DataFrame, station_name: str) -> None:
    cols = numeric_cols(df)
    stats = df[cols].describe().T
    stats["missing"] = df[cols].isna().sum()
    stats["missing_%"] = (df[cols].isna().mean() * 100).round(2)

    fig, ax = plt.subplots(figsize=(14, max(4, len(cols) * 0.5 + 2)))
    ax.axis("off")
    ax.set_title(f"{station_name} — Descriptive Statistics", fontsize=13,
                 fontweight="bold", pad=12)

    col_labels = ["count", "mean", "std", "min", "25%",
                  "50%", "75%", "max", "missing", "missing_%"]
    table_data = [[f"{stats.loc[r, c]:.3g}" if pd.notna(stats.loc[r, c]) else "—"
                   for c in col_labels] for r in stats.index]

    tbl = ax.table(
        cellText=table_data,
        rowLabels=stats.index.tolist(),
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 1.4)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def histogram_page(pdf: PdfPages, df: pd.DataFrame, station_name: str) -> None:
    cols = numeric_cols(df)
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    fig.suptitle(f"{station_name} — Histograms",
                 fontsize=13, fontweight="bold")
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=40, color="#4C72B0",
                     edgecolor="white", linewidth=0.4)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def precipitation_transformations_page(
    pdf: PdfPages,
    df: pd.DataFrame,
    station_name: str
) -> None:

    col = "precipitation_mm"

    data = df[col].dropna().values

    # Keep original positive values for transforms that require positivity
    positive_data = data[data > 0]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"{station_name} — Precipitation Transformations",
        fontsize=13,
        fontweight="bold"
    )

    axes = axes.flatten()

    # ------------------------------------------------------------------
    # 1. Original
    # ------------------------------------------------------------------
    axes[0].hist(
        data,
        bins=40,
        edgecolor="white",
        linewidth=0.4
    )
    axes[0].set_title("Original")

    # ------------------------------------------------------------------
    # 2. log1p
    # ------------------------------------------------------------------
    log_data = np.log1p(data)

    axes[1].hist(
        log_data,
        bins=40,
        edgecolor="white",
        linewidth=0.4
    )
    axes[1].set_title("log1p")

    # ------------------------------------------------------------------
    # 3. sqrt
    # ------------------------------------------------------------------
    sqrt_data = np.sqrt(data)

    axes[2].hist(
        sqrt_data,
        bins=40,
        edgecolor="white",
        linewidth=0.4
    )
    axes[2].set_title("Square Root")

    # ------------------------------------------------------------------
    # 4. Box-Cox
    # ------------------------------------------------------------------
    if len(positive_data) > 0:
        boxcox_data, lam = boxcox(positive_data)

        axes[3].hist(
            boxcox_data,
            bins=40,
            edgecolor="white",
            linewidth=0.4
        )
        axes[3].set_title(f"Box-Cox (λ={lam:.2f})")
    else:
        axes[3].set_visible(False)

    # ------------------------------------------------------------------
    # 5. Yeo-Johnson
    # ------------------------------------------------------------------
    pt_yj = PowerTransformer(method="yeo-johnson")

    yj_data = pt_yj.fit_transform(
        data.reshape(-1, 1)
    ).flatten()

    axes[4].hist(
        yj_data,
        bins=40,
        edgecolor="white",
        linewidth=0.4
    )
    axes[4].set_title("Yeo-Johnson")

    # ------------------------------------------------------------------
    # 6. Quantile -> Normal
    # ------------------------------------------------------------------
    qt = QuantileTransformer(
        output_distribution="normal",
        random_state=42
    )

    qt_data = qt.fit_transform(
        data.reshape(-1, 1)
    ).flatten()

    axes[5].hist(
        qt_data,
        bins=40,
        edgecolor="white",
        linewidth=0.4
    )
    axes[5].set_title("Quantile → Normal")

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    for ax in axes:
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf.savefig(fig)
    plt.close(fig)


def boxplot_page(pdf: PdfPages, df: pd.DataFrame, station_name: str) -> None:
    cols = numeric_cols(df)

    n_cols = 2
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 4 * n_rows)
    )

    fig.suptitle(
        f"{station_name} — Boxplots",
        fontsize=13,
        fontweight="bold"
    )

    axes = np.array(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))

    for ax, col, color in zip(axes, cols, colors):

        values = df[col].dropna().values

        # Main boxplot without default fliers
        bp = ax.boxplot(
            values,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5)
        )

        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.7)

        # ------------------------------------------------------------
        # Detect outliers using IQR rule
        # ------------------------------------------------------------
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = values[
            (values < lower) | (values > upper)
        ]

        # ------------------------------------------------------------
        # Scatter outliers with horizontal jitter
        # ------------------------------------------------------------
        if len(outliers) > 0:
            jitter = np.random.normal(
                loc=1,
                scale=0.01,
                size=len(outliers)
            )

            ax.scatter(
                jitter,
                outliers,
                color=color,
                edgecolor="black",
                alpha=0.6,
                s=12
            )

        ax.set_title(col, fontsize=10)
        ax.tick_params(axis="x", labelbottom=False)

    # Remove unused axes
    for ax in axes[len(cols):]:
        fig.delaxes(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf.savefig(fig)
    plt.close(fig)


def _build_geodataframe(station_meta: list[dict]):
    valid = [s for s in station_meta if not (
        np.isnan(s["lat"]) or np.isnan(s["lon"]))]
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=[Point(s["lon"], s["lat"]) for s in valid],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    return gdf


def _draw_map(ax, gdf, highlight_idx=None):
    others = gdf[gdf.index !=
                 highlight_idx] if highlight_idx is not None else gdf
    highlight = gdf[gdf.index ==
                    highlight_idx] if highlight_idx is not None else None

    if not others.empty:
        others.plot(ax=ax, color="#4C72B0", markersize=40, zorder=3, alpha=0.8)
    if highlight is not None and not highlight.empty:
        highlight.plot(ax=ax, color="red", markersize=80, marker="*", zorder=4)

    for _, row in gdf.iterrows():
        ax.annotate(row["name"], xy=(row.geometry.x, row.geometry.y),
                    xytext=(4, 4), textcoords="offset points", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, lw=0))
    try:
        ctx.add_basemap(
            ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto')
    except Exception:
        ax.set_facecolor("#cde")


    # Expand bounds
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    xpad = (x2 - x1) * 0.5
    ypad = (y2 - y1) * 0.1
    ax.set_xlim(x1 - xpad, x2 + xpad)
    ax.set_ylim(y1 - ypad, y2 + ypad)

    ax.set_axis_off()


def correlation_page(pdf: PdfPages, df: pd.DataFrame, station_name: str) -> None:

    cols = numeric_cols(df)
    if len(cols) < 2:
        return

    data = df[cols].dropna()

    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")

    mic_df = _mic_matrix(data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{station_name} — Correlation Matrices",
                 fontsize=13, fontweight="bold")

    for ax, matrix, title in zip(axes,
                                 [pearson, spearman, mic_df],
                                 ["Pearson", "Spearman", "MIC"]):
        im = ax.imshow(matrix.values, vmin=-1, vmax=1,
                       cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title(title, fontsize=11)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{matrix.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=5, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def overview_map_page(pdf: PdfPages, station_meta: list[dict]) -> None:
    gdf = _build_geodataframe(station_meta)
    if gdf.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.suptitle("All Stations — Overview Map", fontsize=14, fontweight="bold")
    _draw_map(ax, gdf)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def station_map_page(pdf: PdfPages, station_meta: list[dict], current_idx: int, station_name: str) -> None:
    gdf = _build_geodataframe(station_meta)
    if gdf.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f"{station_name} — Location", fontsize=13, fontweight="bold")
    _draw_map(ax, gdf, highlight_idx=current_idx)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def violin_page(pdf: PdfPages, df: pd.DataFrame, station_name: str) -> None:
    cols = numeric_cols(df)
    # Violin needs variance — skip constant or near-empty columns
    valid = [c for c in cols if df[c].dropna(
    ).std() > 0 and df[c].dropna().shape[0] > 10]
    if not valid:
        return

    ncols = 3
    nrows = int(np.ceil(len(valid) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    fig.suptitle(f"{station_name} — Violin Plots",
                 fontsize=13, fontweight="bold")
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid):
        data = df[col].dropna().values
        parts = axes[i].violinplot(data, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.7)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xticks([])
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def timeseries_page(pdf: PdfPages, df: pd.DataFrame, station_name: str) -> None:
    if TIME_COL not in df.columns:
        return

    cols = numeric_cols(df)
    ncols = 2
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    fig.suptitle(f"{station_name} — Time Series (missing values in red)",
                 fontsize=13, fontweight="bold")
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        series = df.set_index(TIME_COL)[col]

        # Plot valid data
        valid = series.dropna()
        ax.plot(valid.index, valid.values, color="#4C72B0",
                linewidth=0.6, label="valid")

        # Highlight missing value positions as red ticks on x-axis
        missing_idx = series[series.isna()].index
        if len(missing_idx):
            ax.scatter(missing_idx, [series.min()] * len(missing_idx),
                       color="red", s=4, zorder=5, label="missing")

        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        if len(missing_idx):
            ax.legend(fontsize=6, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def station_header_page(pdf: PdfPages, df: pd.DataFrame, path: Path) -> None:
    name = df["station_name"].iloc[0] if "station_name" in df.columns else path.stem
    city = df["station_city"].iloc[0] if "station_city" in df.columns else "—"
    lat = df["station_lat"].iloc[0] if "station_lat" in df.columns else "—"
    lon = df["station_lon"].iloc[0] if "station_lon" in df.columns else "—"
    alt = df["station_alt_m"].iloc[0] if "station_alt_m" in df.columns else "—"

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    ax.text(0.05, 0.7, name, fontsize=18,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.4,
            f"City: {city}   |   Lat: {lat}   Lon: {lon}   Alt: {alt} m   |   Rows: {len(df)}",
            fontsize=11, color="#555555", transform=ax.transAxes)
    fig.patch.set_facecolor("#f0f4f8")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def generate_report() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    parquets = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not parquets:
        print(f"[warn] no parquet files found in {PROCESSED_DIR}")
        return

    print(f"[info] {len(parquets)} stations found")

    with PdfPages(OUTPUT_PDF) as pdf:
        cover_page(pdf, "EPAGRI Weather Stations — EDA Report", parquets)
        station_meta = []
        dfs = []
        for path in parquets:
            df = pd.read_parquet(path)
            dfs.append(df)
            station_meta.append({
                "name": df["station_name"].iloc[0] if "station_name" in df.columns else path.stem,
                "lat":  float(df["station_lat"].iloc[0]) if "station_lat" in df.columns else float("nan"),
                "lon":  float(df["station_lon"].iloc[0]) if "station_lon" in df.columns else float("nan"),
            })

        # then inside PdfPages, after cover_page:
        overview_map_page(pdf, station_meta)
        features_table_page(pdf, station_meta, dfs)
        for idx, (path, df) in enumerate(zip(parquets, dfs)):
            df = pd.read_parquet(path)
            name = df["station_name"].iloc[0] if "station_name" in df.columns else path.stem
            print(f"[station] {name}")

            station_header_page(pdf, df, path)
            station_map_page(pdf, station_meta, idx, name)
            stats_page(pdf, df, name)
            histogram_page(pdf, df, name)
            correlation_page(pdf, df, name)
            # precipitation_transformations_page(pdf, df, name)
            boxplot_page(pdf, df, name)
            # violin_page(pdf, df, name)
            timeseries_page(pdf, df, name)

    print(f"[done] report saved to {OUTPUT_PDF}")


if __name__ == "__main__":
    generate_report()
