#!/usr/bin/env python3
"""Quick check: MERRA-2 mean annual SWGDN (kWh/m^2/year) over small Delhi AOI."""
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
        from scripts.irradiance_baseline import get_merra_baseline_info
    except ImportError:
        from irradiance_baseline import get_merra_baseline_info

    info = get_merra_baseline_info(aoi, start_year=2015, end_year=2019)
    kwh = info["merra_mean_annual_sw_kwh_m2"]
    print(f"[PASS] MERRA-2 mean annual SW (2015-2019): {kwh} kWh/m^2/year")
    print(f"       collection={info['merra_collection']}, band={info['merra_band']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
