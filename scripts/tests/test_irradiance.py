#!/usr/bin/env python3
"""Quick check: ERA5 mean annual GHI (kWh/m^2/year) over small Delhi AOI.

Expected range for Delhi: 1700-2100 kWh/m^2/year.
Global Solar Atlas reference (Solargis) for Delhi: ~1930 kWh/m^2/year.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import ee

DELHI_SMALL = [[77.20, 28.58], [77.25, 28.58], [77.25, 28.62], [77.20, 28.62], [77.20, 28.58]]
EXPECTED_MIN = 1700.0
EXPECTED_MAX = 2200.0


def main() -> int:
    project_id = os.environ.get("GEE_PROJECT_ID", "pv-mapping-india")
    try:
        ee.Initialize(project=project_id)
        print("[OK] Earth Engine initialized")
    except Exception as e:
        print(f"[ERROR] Init failed: {e}")
        return 1

    aoi = ee.Geometry.Polygon(DELHI_SMALL)
    try:
        from scripts.irradiance_baseline import get_era5_baseline_info
    except ImportError:
        from irradiance_baseline import get_era5_baseline_info

    info = get_era5_baseline_info(aoi, start_year=2020, end_year=2024)
    kwh = info["mean_annual_ghi_kwh_m2_year"]
    src = info["value_source"]
    print(f"ERA5 mean annual GHI (2020-2024): {kwh:.1f} kWh/m^2/year  [source: {src}]")
    print(f"  collection={info['collection']}, band={info['band']}")
    print(f"  Global Solar Atlas reference for Delhi: ~1930 kWh/m^2/year")

    if EXPECTED_MIN <= kwh <= EXPECTED_MAX:
        print(f"[PASS] Value in expected range [{EXPECTED_MIN}, {EXPECTED_MAX}]")
        return 0
    else:
        print(f"[WARN] Value outside expected range [{EXPECTED_MIN}, {EXPECTED_MAX}] -- check units or collection")
        return 1


if __name__ == "__main__":
    sys.exit(main())
