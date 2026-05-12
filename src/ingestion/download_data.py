"""
Weather data downloader.
Streams archive into memory, extracts contents directly to data/raw/.
Supported formats: .zip, .tar.xz / .tar.gz / .tar.bz2
Folders are created automatically on first run.
"""

# claude-sonnet-4-20250514
import io
import tarfile
import zipfile
import requests
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_URL = "https://arquivos.ufsc.br/f/d5413e85d6004a5c8a71/?dl=1"
# src/ingestion/download_data.py -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data"
DIRS = {
    "raw":       BASE_DIR / "raw",
    "processed": BASE_DIR / "processed",
    "features":  BASE_DIR / "features",
}
# ─────────────────────────────────────────────────────────────────────────────


def create_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
        print(f"[dir] {path}")


def download_to_buffer(url: str) -> tuple[io.BytesIO, str]:
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        buf = io.BytesIO(r.content)
    print(f"[buffered] {buf.getbuffer().nbytes / 1024:.1f} KB")
    return buf, content_type


def extract(buf: io.BytesIO, dest_dir: Path, content_type: str) -> None:
    if zipfile.is_zipfile(buf):
        buf.seek(0)
        print(f"[extract] zip -> {dest_dir}")
        with zipfile.ZipFile(buf) as z:
            z.extractall(dest_dir)
    else:
        buf.seek(0)
        try:
            with tarfile.open(fileobj=buf, mode="r:*") as t:
                print(f"[extract] tar -> {dest_dir}")
                t.extractall(dest_dir)
        except tarfile.TarError as e:
            raise ValueError(f"Unsupported archive format: {e}")
    print("[done]")


def main() -> None:
    create_dirs()
    buf, content_type = download_to_buffer(DATA_URL)
    extract(buf, DIRS["raw"], content_type)


if __name__ == "__main__":
    main()
