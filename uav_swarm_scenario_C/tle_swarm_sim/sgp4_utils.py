from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
from sgp4.api import Satrec, jday

from .tle import TleRecord

EARTH_RADIUS_M = 6378137.0
EARTH_ECC2 = 6.69437999014e-3
MU_EARTH = 398600.4418e9


@dataclass(frozen=True, slots=True)
class SamplePoint:
    timestamp: datetime
    ecef_m: np.ndarray
    lat_deg: float
    lon_deg: float
    alt_m: float
    range_m: float
    elevation_deg: float


def build_satrec(record: TleRecord) -> Satrec:
    return Satrec.twoline2rv(record.line1, record.line2)


def gmst_rad(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    year = dt.year
    month = dt.month
    day = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100)
    b = 2 - a + math.floor(a / 4)
    jd = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 1524.5
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def teme_to_ecef(r_teme_km: np.ndarray, dt: datetime) -> np.ndarray:
    theta = gmst_rad(dt)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ (r_teme_km * 1000.0)


def ecef_to_geodetic(ecef_m: np.ndarray) -> tuple[float, float, float]:
    x, y, z = ecef_m
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - EARTH_ECC2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = EARTH_RADIUS_M / math.sqrt(1 - EARTH_ECC2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - EARTH_ECC2 * n / (n + alt)))
    sin_lat = math.sin(lat)
    n = EARTH_RADIUS_M / math.sqrt(1 - EARTH_ECC2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)
    n = EARTH_RADIUS_M / math.sqrt(1 - EARTH_ECC2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1 - EARTH_ECC2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def range_and_elevation(sat_ecef_m: np.ndarray, site_lat_deg: float, site_lon_deg: float, site_alt_m: float):
    site_ecef = geodetic_to_ecef(site_lat_deg, site_lon_deg, site_alt_m)
    rho = sat_ecef_m - site_ecef
    lat = math.radians(site_lat_deg)
    lon = math.radians(site_lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)
    enu = np.array([
        -slon * rho[0] + clon * rho[1],
        -slat * clon * rho[0] - slat * slon * rho[1] + clat * rho[2],
        clat * clon * rho[0] + clat * slon * rho[1] + slat * rho[2],
    ])
    east, north, up = enu
    rng = float(np.linalg.norm(rho))
    elev = math.degrees(math.atan2(up, math.hypot(east, north)))
    return rng, elev


def propagate_record(
    record: TleRecord,
    start: datetime,
    duration_minutes: int,
    step_seconds: int,
    site_lat_deg: float,
    site_lon_deg: float,
    site_alt_m: float,
) -> list[SamplePoint]:
    sat = build_satrec(record)
    points: list[SamplePoint] = []
    total_steps = int(duration_minutes * 60 / step_seconds) + 1
    for step in range(total_steps):
        ts = start + timedelta(seconds=step * step_seconds)
        jd, fr = jday(ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second + ts.microsecond / 1e6)
        err, teme_pos_km, _vel = sat.sgp4(jd, fr)
        if err != 0:
            continue
        ecef_m = teme_to_ecef(np.array(teme_pos_km, dtype=float), ts)
        lat_deg, lon_deg, alt_m = ecef_to_geodetic(ecef_m)
        rng_m, elev_deg = range_and_elevation(ecef_m, site_lat_deg, site_lon_deg, site_alt_m)
        points.append(
            SamplePoint(
                timestamp=ts,
                ecef_m=ecef_m,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                alt_m=alt_m,
                range_m=rng_m,
                elevation_deg=elev_deg,
            )
        )
    return points


def satellite_ecef_at(record: TleRecord, dt: datetime) -> np.ndarray:
    sat = build_satrec(record)
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    err, teme_pos_km, _vel = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation failed with error code {err}")
    return teme_to_ecef(np.array(teme_pos_km, dtype=float), dt)


def mean_orbit_altitude_m(record: TleRecord) -> float:
    mm = record.mean_motion_rev_per_day
    n = mm * 2.0 * math.pi / 86400.0
    a = (MU_EARTH / (n * n)) ** (1.0 / 3.0)
    return a - EARTH_RADIUS_M
