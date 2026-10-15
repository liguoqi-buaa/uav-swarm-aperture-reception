from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen


def download_text(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def download_starlink_tle(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = download_text(url)
    output_path.write_text(text, encoding="utf-8")
    return output_path

