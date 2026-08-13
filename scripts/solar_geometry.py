"""
Where the sun is (altitude + azimuth, degrees) at a given instant -- feeds the shadow
model. The point is to sample sun positions that actually match the window the user
picked: solstices/equinoxes for a year, mid-month days for a quarter, a few days for a
month, or a single day. Everything's in UTC to line up with ERA5.

Just stdlib math. Azimuth is clockwise from geographic north (0-360), same as penalties.py.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import List, Tuple

# altitude_deg, azimuth_deg_from_north, weight, hour_utc (integer 0..23)
WeightedPosition = Tuple[float, float, float, int]


def sun_altitude_azimuth_north(lat_deg: float, lon_deg: float, when_utc: datetime) -> Tuple[float, float]:
    """Sun altitude (deg) and azimuth from north clockwise (deg), when_utc must be timezone-aware UTC."""
    if when_utc.tzinfo is None:
        raise ValueError("when_utc must be timezone-aware (use UTC)")
    when_utc = when_utc.astimezone(timezone.utc)
    y, m, d = when_utc.year, when_utc.month, when_utc.day
    ut = when_utc.hour + when_utc.minute / 60.0 + when_utc.second / 3600.0

    y2, m2 = y, m
    if m2 <= 2:
        y2 -= 1
        m2 += 12
    A = y2 // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (y2 + 4716)) + int(30.6001 * (m2 + 1)) + d + ut / 24.0 + B - 1524.5

    n = jd - 2451545.0
    L = (280.460 + 0.9856474 * n) % 360
    if L < 0:
        L += 360
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lambda_sun = math.radians((L + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g)) % 360)
    epsilon = math.radians(23.439 - 0.0000004 * n)
    dec = math.asin(math.sin(epsilon) * math.sin(lambda_sun))
    alpha = math.atan2(math.cos(epsilon) * math.sin(lambda_sun), math.cos(lambda_sun))

    gmst = (280.46061837 + 360.98564736629 * n) % 360
    gmst_rad = math.radians(gmst)
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    H = gmst_rad + lon_rad - alpha
    while H > math.pi:
        H -= 2 * math.pi
    while H < -math.pi:
        H += 2 * math.pi

    alt = math.asin(
        math.sin(lat_rad) * math.sin(dec) + math.cos(lat_rad) * math.cos(dec) * math.cos(H)
    )
    az = math.atan2(
        -math.sin(H) * math.cos(dec),
        math.cos(lat_rad) * math.sin(dec) - math.sin(lat_rad) * math.cos(dec) * math.cos(H),
    )
    az_deg = math.degrees(az) % 360.0
    alt_deg = math.degrees(alt)
    return alt_deg, az_deg


def _normalize_weights(positions: List[WeightedPosition]) -> List[WeightedPosition]:
    total = sum(w for _, _, w, _ in positions)
    if total <= 0:
        return [(45.0, 180.0, 1.0, 12)]
    return [(a, z, w / total, h) for a, z, w, h in positions]


def weighted_positions_for_calendar_day(
    lat_deg: float,
    lon_deg: float,
    d: date,
    step_hours: float = 1.0,
    min_alt_deg: float = 2.0,
) -> List[WeightedPosition]:
    """Hourly (or coarser) samples on one UTC calendar day; weights ~ sin(alt)."""
    out: List[WeightedPosition] = []
    t = 0.0
    while t < 24.0:
        h = int(t)
        mi = int((t - h) * 60)
        when = datetime(d.year, d.month, d.day, h, mi, 0, tzinfo=timezone.utc)
        alt, az = sun_altitude_azimuth_north(lat_deg, lon_deg, when)
        if alt >= min_alt_deg:
            out.append((alt, az, math.sin(math.radians(alt)), h))
        t += step_hours
    return _normalize_weights(out)


def merge_weighted_position_sets(sets: List[List[WeightedPosition]]) -> List[WeightedPosition]:
    """Concatenate position lists and renormalise weights (equal prior weight per set)."""
    flat: List[WeightedPosition] = []
    for s in sets:
        if not s:
            continue
        wsum = sum(x[2] for x in s)
        if wsum <= 0:
            continue
        scale = 1.0 / len(sets)
        for a, z, w, h in s:
            flat.append((a, z, w * scale, h))
    return _normalize_weights(flat)


def solar_positions_yearly(lat_deg: float, lon_deg: float, year: int) -> List[WeightedPosition]:
    """Solstices and equinoxes of the selected calendar year (UTC dates)."""
    key_days = [
        date(year, 3, 21),
        date(year, 6, 21),
        date(year, 9, 23),
        date(year, 12, 21),
    ]
    sets = [weighted_positions_for_calendar_day(lat_deg, lon_deg, kd, step_hours=1.5) for kd in key_days]
    return merge_weighted_position_sets(sets)


def solar_positions_quarterly(lat_deg: float, lon_deg: float, year: int, quarter: int) -> List[WeightedPosition]:
    """Three mid-month UTC days within the calendar quarter (15th of each month)."""
    if quarter < 1 or quarter > 4:
        raise ValueError("quarter must be 1..4")
    month_triples = {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9), 4: (10, 11, 12)}
    days = [date(year, m, 15) for m in month_triples[quarter]]
    sets = [weighted_positions_for_calendar_day(lat_deg, lon_deg, d, step_hours=1.5) for d in days]
    return merge_weighted_position_sets(sets)


def solar_positions_monthly(lat_deg: float, lon_deg: float, year: int, month: int) -> List[WeightedPosition]:
    """Three representative UTC days in the selected month (8th, 15th, 22nd)."""
    if month < 1 or month > 12:
        raise ValueError("month must be 1..12")
    days = [date(year, month, 8), date(year, month, 15), date(year, month, 22)]
    sets = [weighted_positions_for_calendar_day(lat_deg, lon_deg, d, step_hours=1.0) for d in days]
    return merge_weighted_position_sets(sets)


def solar_positions_single_day(lat_deg: float, lon_deg: float, d: date) -> List[WeightedPosition]:
    """One UTC calendar day, hourly sin-weighted samples."""
    return weighted_positions_for_calendar_day(lat_deg, lon_deg, d, step_hours=1.0)
