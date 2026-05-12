"""
EPAGRI weather data preprocessor.
Reads raw CSVs from data/raw/, parses metadata, cleans and translates
columns, and saves one parquet per station to data/cleaned/.
"""

# claude-sonnet-4-20250514

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data"
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "cleaned"

NA_VALUES = ["9999.9", "9999", "9999,9", ""]

COLUMN_MAP = {
    "Codigo":               "station_id",
    "Data Horario":         "datetime",
    "TempArInst(C)":        "temp_air_inst_c",
    "TempMin(C)":           "temp_min_c",
    "TempMax(C)":           "temp_max_c",
    "Vel.MediaVento(m/s)":  "wind_speed_avg_ms",
    "Dir.MediaVento(Graus)": "wind_dir_avg_deg",
    "VentoMax(m/s)":        "wind_speed_max_ms",
    "PressaoAtm(mB)":       "atm_pressure_mb",
    "UmidRelMedia(%)":      "rel_humidity_avg_pct",
    "Precipitacao(mm)":     "precipitation_mm",
    "Altura da Maré(cm)":   "tide_level_cm"
}
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StationMeta:
    station_id: int
    name: str
    city: str
    lat: float
    lon: float
    altitude_m: float


def slugify(text: str) -> str:
    """Lowercase ASCII slug for filenames."""
    text = unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def parse_metadata(line: str) -> StationMeta:
    """Parse the first CSV row: id, full_name, city, lat, lon, altitude."""
    parts = [p.strip() for p in line.split(",")]
    # parts[1] is 'City - Description' or just description
    name = parts[1] if len(parts) > 1 else "unknown"
    city = parts[2] if len(parts) > 2 else "unknown"
    lat = float(parts[3]) if len(parts) > 3 and parts[3] else float("nan")
    lon = float(parts[4]) if len(parts) > 4 and parts[4] else float("nan")
    alt = float(parts[5]) if len(parts) > 5 and parts[5] else float("nan")
    return StationMeta(
        station_id=int(parts[0]),
        name=name,
        city=city,
        lat=lat,
        lon=lon,
        altitude_m=alt,
    )


def fix_header_line(header: str) -> str:
    """
    The header row is missing a trailing comma that the data rows have,
    which causes pandas to misalign columns. Add it back so column count matches.
    """
    data_col_count = len(header.split(",")) + \
        1  # one extra for the trailing comma in data
    return header + ","


def load_csv(file_path: Path) -> tuple[StationMeta, pd.DataFrame]:
    with open(file_path, "r", encoding="cp1252") as f:
        meta_line = f.readline().rstrip("\n")
        header_line = f.readline().rstrip("\n")
        data_text = f.read()

    meta = parse_metadata(meta_line)

    # Fix missing trailing comma on header
    fixed_header = fix_header_line(header_line)

    # Reconstruct full CSV text with fixed header
    full_text = fixed_header + "\n" + data_text

    df = pd.read_csv(
        pd.io.common.StringIO(full_text),
        header=0,
        na_values=NA_VALUES,
        parse_dates=["Data Horario"],
        date_format="%d/%m/%Y %H:%M:%S",
        engine="python",
        encoding=None,  # already decoded
    )

    return meta, df


def clean(df: pd.DataFrame, meta: StationMeta) -> pd.DataFrame:
    # Drop unnamed/trailing columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Reset index if Codigo was absorbed as index
    if df.index.name == "Codigo":
        df = df.reset_index()

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    # Translate columns
    df = df.rename(
        columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # Drop rows where ALL numeric columns are NaN (completely empty readings)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        df = df.dropna(subset=numeric_cols, how="all")

    # Add station metadata as columns
    station_slug = slugify(meta.name)
    df["station_name"] = meta.name
    df["station_city"] = meta.city
    df["station_lat"] = meta.lat
    df["station_lon"] = meta.lon
    df["station_alt_m"] = meta.altitude_m
    station_cols = ["station_name", "station_city",
                    "station_lat", "station_lon", "station_alt_m"]
    df = df[station_cols + [c for c in df.columns if c not in station_cols]]

    # Sort by time
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    return df


def process_all() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"[warn] no CSV files found in {RAW_DIR}")
        return

    print(f"[info] found {len(csv_files)} CSV files")

    for path in csv_files:
        try:
            meta, raw_df = load_csv(path)
            df = clean(raw_df, meta)

            out_name = f"{meta.station_id}_{slugify(meta.name)}.parquet"
            out_path = PROCESSED_DIR / out_name
            df.to_parquet(out_path, index=False)

            print(
                f"[ok] {path.name} -> {out_name}  ({len(df)} rows, {len(df.columns)} cols)")
        except Exception as e:
            print(f"[error] {path.name}: {e}")


if __name__ == "__main__":
    process_all()
