#!/usr/bin/env python3
"""
Validate roof-masked ERA5 baseline computation.

Checks:
  - roof_area_m2 > 0
  - regional_irradiance_kwh_m2_year in plausible range for Delhi (1700-2200)
  - pre_penalty_total_kwh_year > 0

Run from project root:
  python scripts/tests/test_roof_baseline.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ee  # pyright: ignore[reportMissingImports]

DELHI_SMALL = [
    [77.20, 28.58],
    [77.25, 28.58],
    [77.25, 28.62],
    [77.20, 28.62],
    [77.20, 28.58],
]


def main() -> int:
    project_id = os.environ.get("GEE_PROJECT_ID", "pv-mapping-india")
    try:
        ee.Initialize(project=project_id)
        print("[OK] Earth Engine initialized")
    except Exception as e:
        print(f"[FAIL] Earth Engine init failed: {e}")
        return 1

    aoi = ee.Geometry.Polygon(DELHI_SMALL)

    try:
        from scripts.datasets import get_open_buildings_temporal
        from scripts.rooftops import build_rooftop_candidate_mask
        from scripts.irradiance_baseline import get_roof_masked_era5_baseline_info
    except ImportError:
        from datasets import get_open_buildings_temporal
        from rooftops import build_rooftop_candidate_mask
        from irradiance_baseline import get_roof_masked_era5_baseline_info

    buildings = get_open_buildings_temporal(aoi, year=2022)
    roof_mask = build_rooftop_candidate_mask(buildings, presence_threshold=0.5, min_height_m=0.0)

    info = get_roof_masked_era5_baseline_info(
        aoi=aoi,
        roof_mask=roof_mask,
        start_year=2020,
        end_year=2024,
    )

    roof_area = info["roof_area_m2"]
    irr = info["regional_irradiance_kwh_m2_year"]
    total = info["pre_penalty_total_kwh_year"]
    src = info["irradiance_source"]

    print(f"roof_area_m2                  = {roof_area:.1f} m^2")
    print(f"regional_irradiance_kwh_m2_yr = {irr:.1f} kWh/m^2/year  [source: {src}]")
    print(f"pre_penalty_total_kwh_year    = {total:.0f} kWh/year")
    print(f"collection                    = {info['collection']}")
    print(f"Global Solar Atlas reference  = ~1930 kWh/m^2/year (Delhi)")

    ok = True
    if roof_area <= 0:
        print("[FAIL] roof_area_m2 is zero or negative")
        ok = False
    else:
        print(f"[PASS] roof_area_m2 > 0")

    if 1700 <= irr <= 2200:
        print(f"[PASS] regional_irradiance in expected range [1700, 2200]")
    else:
        print(f"[WARN] regional_irradiance {irr:.1f} outside expected range -- check units/collection")
        ok = False

    if total > 0:
        print(f"[PASS] pre_penalty_total_kwh_year > 0")
    else:
        print(f"[FAIL] pre_penalty_total_kwh_year is zero")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
