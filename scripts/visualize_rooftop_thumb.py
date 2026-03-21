#!/usr/bin/env python3
"""
Minimal rooftop-mask preview: fetch a PNG thumbnail from Earth Engine (no Jupyter).

Usage (from project root, venv active, GEE authenticated):
  python scripts/visualize_rooftop_thumb.py
  python scripts/visualize_rooftop_thumb.py --out outputs/roof_thumb.png

Opens a getThumbURL; saves the PNG to disk and prints the URL.
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import ee

# Default: same small Delhi AOI as tests
DEFAULT_AOI = [[77.20, 28.58], [77.25, 28.58], [77.25, 28.62], [77.20, 28.62], [77.20, 28.58]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Save rooftop mask thumbnail from GEE")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "outputs", "rooftop_thumb.png"),
        help="Output PNG path",
    )
    parser.add_argument("--year", type=int, default=2022, help="Open Buildings temporal year")
    parser.add_argument(
        "--presence",
        type=float,
        default=0.5,
        help="building_presence threshold",
    )
    parser.add_argument("--width", type=int, default=512, help="Thumbnail width (px)")
    args = parser.parse_args()

    project_id = os.environ.get("GEE_PROJECT_ID", "pv-mapping-india")
    try:
        ee.Initialize(project=project_id)
    except Exception as e:
        print(f"[ERROR] Earth Engine init: {e}")
        return 1

    try:
        from scripts.rooftops import build_rooftop_candidate_mask
        from scripts.datasets import get_open_buildings_temporal
    except ImportError:
        from rooftops import build_rooftop_candidate_mask
        from datasets import get_open_buildings_temporal

    aoi = ee.Geometry.Polygon(DEFAULT_AOI)
    buildings = get_open_buildings_temporal(aoi, year=args.year)
    mask = build_rooftop_candidate_mask(
        buildings, presence_threshold=args.presence, min_height_m=0.0
    )
    # Green = roof candidate, black = non-roof
    thumb_image = mask.visualize(
        min=0,
        max=1,
        palette=["000000", "00ff66"],
    )

    # GeoJSON-style polygon: one outer ring [ [ [lon,lat], ... ] ]
    url = thumb_image.getThumbURL(
        {
            "region": [DEFAULT_AOI],
            "dimensions": args.width,
            "format": "png",
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    urllib.request.urlretrieve(url, args.out)
    print(f"[OK] Saved thumbnail: {args.out}")
    print(f"     URL (same image): {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
