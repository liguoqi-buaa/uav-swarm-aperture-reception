from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TleRecord:
    name: str
    line1: str
    line2: str

    @property
    def catalog_number(self) -> str:
        return self.line1[2:7].strip()

    @property
    def epoch_year(self) -> int:
        year = int(self.line1[18:20])
        return 2000 + year if year < 57 else 1900 + year

    @property
    def epoch_day(self) -> float:
        return float(self.line1[20:32])

    @property
    def inclination_deg(self) -> float:
        return float(self.line2[8:16])

    @property
    def eccentricity(self) -> float:
        return float(f"0.{self.line2[26:33].strip()}")

    @property
    def mean_motion_rev_per_day(self) -> float:
        return float(self.line2[52:63])

    @property
    def mean_motion_derivative(self) -> float:
        raw = self.line1[33:43].strip()
        return float(raw) if raw else 0.0

    @property
    def epoch_datetime(self) -> datetime:
        from datetime import timedelta

        year = self.epoch_year
        day = self.epoch_day
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        return start + timedelta(days=day - 1.0)


def load_tle_records(path: Path) -> list[TleRecord]:
    text = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    records: list[TleRecord] = []
    i = 0
    while i < len(text):
        line = text[i]
        if line.startswith("1 ") and i + 1 < len(text) and text[i + 1].startswith("2 "):
            records.append(TleRecord(name="", line1=line, line2=text[i + 1].strip()))
            i += 2
            continue
        if i + 2 < len(text) and text[i + 1].startswith("1 ") and text[i + 2].startswith("2 "):
            records.append(TleRecord(name=line, line1=text[i + 1], line2=text[i + 2]))
            i += 3
            continue
        i += 1
    return records


def write_tle_archive(records: Iterable[TleRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if record.name:
            lines.append(record.name)
        lines.append(record.line1)
        lines.append(record.line2)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
