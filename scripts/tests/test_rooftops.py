#!/usr/bin/env python3
"""
Verify rooftop candidate mask + area (Open Buildings 2.5D) on a small AOI.
Run from project root: python scripts/tests/test_rooftops.py
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

# Same small Delhi box as test_datasets
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
        from scripts.datasets import get_open_buildings_temporal
        from scripts.rooftops import build_rooftop_candidate_mask, rooftop_area_m2_reduce
    except ImportError:
        from datasets import get_open_buildings_temporal
        from rooftops import build_rooftop_candidate_mask, rooftop_area_m2_reduce

    buildings = get_open_buildings_temporal(aoi, year=2022)
    mask = build_rooftop_candidate_mask(buildings, presence_threshold=0.5, min_height_m=0.0)
    raw = rooftop_area_m2_reduce(mask, aoi, scale_m=4.0, tile_scale=4).getInfo()
    m2 = raw.get("roof_candidate") or 0.0
    print(f"[PASS] rooftop_area_m2 (scale=4m, no terrain filter): {m2:.2f}")
    print(f"       reduceRegion keys: {list(raw.keys())}")

    try:
        from scripts.rooftops import get_rooftop_area_m2_info
    except ImportError:
        from rooftops import get_rooftop_area_m2_info

    info = get_rooftop_area_m2_info(
        aoi,
        year=2022,
        presence_threshold=0.5,
        min_height_m=0.0,
        exclusion_mask=None,
        scale_m=4.0,
    )
    print(f"[PASS] get_rooftop_area_m2_info: area={info['rooftop_candidate_area_m2']:.2f} m2")
    print(f"       reduce_scale_m={info['reduce_scale_m']}, aoi_km2={info['aoi_area_km2']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
