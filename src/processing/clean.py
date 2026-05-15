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
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data"
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "cleaned"

NA_VALUES = ["9999.9", "9999", "9999,9", ""]
MONTHS_PT = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Marco": 3,
    "Abril": 4, "Maio": 5, "Junho": 6, "Julho": 7,
    "Agosto": 8, "Setembro": 9, "Outubro": 10,
    "Novembro": 11, "Dezembro": 12,
}

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
INMET_COLUMN_MAP = {
    "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)":                       "precipitation_mm",
    "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)":  "atm_pressure_mb",
    "PRESSÃO ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)":        "atm_pressure_max_mb",
    "PRESSÃO ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)":       "atm_pressure_min_mb",
    "RADIACAO GLOBAL (Kj/m²)":                                "solar_radiation_kjm2",
    "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)":           "temp_air_inst_c",
    "TEMPERATURA DO PONTO DE ORVALHO (°C)":                   "dew_point_c",
    "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)":             "temp_max_c",
    "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)":             "temp_min_c",
    "TEMPERATURA ORVALHO MAX. NA HORA ANT. (AUT) (°C)":       "dew_point_max_c",
    "TEMPERATURA ORVALHO MIN. NA HORA ANT. (AUT) (°C)":       "dew_point_min_c",
    "UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)":               "rel_humidity_max_pct",
    "UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)":               "rel_humidity_min_pct",
    "UMIDADE RELATIVA DO AR, HORARIA (%)":                    "rel_humidity_avg_pct",
    "VENTO, DIREÇÃO HORARIA (gr) (° (gr))":                   "wind_dir_avg_deg",
    "VENTO, RAJADA MAXIMA (m/s)":                             "wind_speed_max_ms",
    "VENTO, VELOCIDADE HORARIA (m/s)":                        "wind_speed_avg_ms",
}


@dataclass
class StationMeta:
    station_id: int
    name: str
    city: str
    lat: float
    lon: float
    altitude_m: float


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def parse_inmet_meta(path: Path) -> dict:
    meta = {}
    with open(path, encoding="latin1") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            key = parts[0].strip().rstrip(":")
            val = parts[1].strip().replace(",", ".")
            if key == "LATITUDE":
                meta["lat"] = float(val)
            elif key == "LONGITUDE":
                meta["lon"] = float(val)
            elif key == "ALTITUDE":
                meta["alt_m"] = float(val)
            elif key == "ESTACAO":
                meta["name"] = val
            elif key == "CODIGO (WMO)":
                meta["station_id"] = val
            elif key == "UF":
                meta["city"] = val
            if key == "Data":
                break
    return meta


def load_inmet_csv(path: Path) -> tuple[dict, pd.DataFrame]:
    meta = parse_inmet_meta(path)

    df = pd.read_csv(
        path,
        sep=";",
        skiprows=8,
        header=0,
        encoding="latin1",
        na_values=["", "-9999", "-9999.0"],
        decimal=",",
    )

    # Drop trailing empty column from trailing semicolon
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df.rename(columns={'Data': 'DATA (YYYY-MM-DD)'},
              inplace=True,  errors='ignore')
    df.rename(columns={'HORA (UTC)': 'Hora UTC'},
              inplace=True, errors='ignore')

    df.rename(columns={'RADIACAO GLOBAL (Kj/m²)': 'RADIACAO GLOBAL (KJ/m²)'},
              inplace=True, errors='ignore')

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df['DATA (YYYY-MM-DD)'] + " " +
        df["Hora UTC"].str.replace(" UTC", "", regex=False),
        format="mixed",         errors="coerce",
    )
    df = df.drop(columns=['DATA (YYYY-MM-DD)', "Hora UTC"])
    df = df.dropna(subset=["datetime"])

    # Rename columns
    df = df.rename(
        columns={k: v for k, v in INMET_COLUMN_MAP.items() if k in df.columns})

    # Add station metadata
    df["station_name"] = meta.get("name", path.stem)
    df["station_city"] = meta.get("city", "")
    df["station_lat"] = meta.get("lat", float("nan"))
    df["station_lon"] = meta.get("lon", float("nan"))
    df["station_alt_m"] = meta.get("alt_m", float("nan"))

    station_cols = ["station_name", "station_city",
                    "station_lat", "station_lon", "station_alt_m"]
    df = df[station_cols + [c for c in df.columns if c not in station_cols]]

    df = df.sort_values("datetime").reset_index(drop=True)
    return meta, df


def process_inmet(center_lat: float = None, center_lon: float = None, radius_km: float = 100.0) -> None:
    inmet_files = sorted(RAW_DIR.rglob("INMET_*.CSV"))
    if not inmet_files:
        print("[warn] no INMET CSV files found")
        return

    print(f"[inmet] found {len(inmet_files)} files")

    for path in inmet_files:
        meta, df = load_inmet_csv(path)

        # Distance filter
        if center_lat is not None and center_lon is not None:
            dist = haversine_km(center_lat, center_lon,
                                meta["lat"], meta["lon"])
            if dist > radius_km:
                print(f"[skip] {path.name} — {dist:.1f} km from center")

        station_slug = slugify(meta.get("name", path.stem))
        out_name = f"inmet_{meta.get('station_id', station_slug)}_{station_slug}.parquet"
        out_path = PROCESSED_DIR / out_name
        df.to_parquet(out_path, index=False)
        print(
            f"[ok] {path.name} -> {out_name}  ({len(df)} rows, {len(df.columns)} cols)")


def process_inmet(center_lat: float = None, center_lon: float = None, radius_km: float = 100.0) -> None:
    inmet_files = sorted(RAW_DIR.rglob("INMET_*.CSV"))
    if not inmet_files:
        print("[warn] no INMET CSV files found")
        return

    print(f"[inmet] found {len(inmet_files)} files")
    names = {}
    # Group by station ID (e.g. A898 from filename)
    station_files: dict[str, str, list[Path]] = {}
    for path in inmet_files:
        parts = path.stem.split("_")
        # INMET_S_SC_A898_CAMPOS_NOVOS_... → parts[3] is station ID
        station_id = parts[3] if len(parts) > 3 else path.stem
        names[station_id] = "_".join(parts[3:-3])
        station_files.setdefault(station_id, []).append(path)

    for station_id,  paths in station_files.items():
        dfs = []
        meta = None

        meta = parse_inmet_meta(paths[0])

        # Distance filter
        if center_lat is not None and center_lon is not None:
            dist = haversine_km(center_lat, center_lon,
                                meta["lat"], meta["lon"])

            if dist > radius_km:
                print(
                    f"[skip] {station_id} — {names[station_id]} — {dist:.1f} km from center")
                continue

        for path in sorted(paths):
            m, df = load_inmet_csv(path)
            if meta is None:
                meta = m

            dfs.append(df)

        combined = (pd.concat(dfs, ignore_index=True)
                    .drop_duplicates(subset=["datetime"])
                    .sort_values("datetime")
                    .reset_index(drop=True))

        out_name = f"inmet_{station_id}_{names[station_id]}.parquet"
        out_path = PROCESSED_DIR / out_name
        combined.to_parquet(out_path, index=False)
        print(
            f"[ok] {station_id} ({len(paths)} files) -> {out_name}  ({len(combined)} rows, {len(combined.columns)} cols)")


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


def load_epagri_csv(file_path: Path) -> tuple[StationMeta, pd.DataFrame]:
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


def parse_tide_file(filepath: Path, year: int) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if "BEGIN" in text:
        text = text.split("BEGIN", 1)[1]

    lines = text.splitlines()
    records = []
    current_month = None
    current_day = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line in MONTHS_PT:
            current_month = MONTHS_PT[line]
            current_day = None
            i += 1
            continue
        if re.fullmatch(r"\d{2}", line) and i + 1 < len(lines):
            weekday = lines[i + 1].strip()
            if re.fullmatch(r"[A-ZÇÁÉÍÓÚ]{3}", weekday):
                current_day = int(line)
                i += 2
                continue
        m = re.fullmatch(r"\s*(\d{2})(\d{2})\s+(-?\d+\.\d+)\s*", line)
        if m and current_month and current_day:
            try:
                dt = pd.Timestamp(year=year, month=current_month,
                                  day=current_day, hour=int(m.group(1)),
                                  minute=int(m.group(2)))
                records.append(
                    {"datetime": dt, "tide_level_cm": float(m.group(3))*100})
            except ValueError:
                pass
        i += 1

    return (pd.DataFrame(records)
            .sort_values("datetime")
            .drop_duplicates(subset="datetime")
            .reset_index(drop=True))


# TODO: convert to cm
def process_tides() -> None:
    tide_table_year = {
        2024: RAW_DIR / "tide_table_2024.txt",
        2025: RAW_DIR / "tide_table_2025.txt",
        2026: RAW_DIR / "tide_table_2026.txt",
    }

    dfs = []
    for year, filepath in tide_table_year.items():
        if not filepath.exists():
            print(f"[warn] tide file not found: {filepath.name}")
            continue
        print(f"[tide] parsing {filepath.name}")
        dfs.append(parse_tide_file(filepath, year))

    if not dfs:
        print("[warn] no tide files parsed")
        return

    df = (pd.concat(dfs, ignore_index=True)
          .drop_duplicates()
          .sort_values("datetime")
          .reset_index(drop=True))
    

    out_path = PROCESSED_DIR / "tide_table.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[ok] tide_table.parquet  ({len(df)} rows)")


def process_all() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        print(f"[warn] no CSV files found in {RAW_DIR}")
        return

    print(f"[info] found {len(csv_files)} CSV files")

    for path in csv_files:
        try:
            meta, raw_df = load_epagri_csv(path)
            df = clean(raw_df, meta)
            out_name = f"{meta.station_id}_{slugify(meta.name)}.parquet"
            out_path = PROCESSED_DIR / out_name
            df.to_parquet(out_path, index=False)
            print(
                f"[ok] {path.name} -> {out_name}  ({len(df)} rows, {len(df.columns)} cols)")
        except Exception as e:
            print(f"[error] {path.name}: {e}")

    process_tides()

    # Florianópolis center — adjust or pass None to load all
    process_inmet(center_lat=-27.59, center_lon=-48.55, radius_km=120.0)


if __name__ == "__main__":
    process_all()
