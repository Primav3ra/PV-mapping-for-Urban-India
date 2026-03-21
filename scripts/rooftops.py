"""
Rooftop candidate mask and area from Open Buildings 2.5D Temporal (GEE).

Design choices (defaults):
- building_presence > 0.5: model confidence is uncalibrated; threshold is a tunable prior.
- min_height_m: optional floor to drop low/noise structures (0 = disabled).
- reduceRegion scale=4 m: matches ~4 m effective resolution of Open Buildings temporal.
"""

from __future__ import annotations

import ee
from typing import Any, Dict, Optional

try:
    from datasets import get_open_buildings_temporal
except ImportError:
    from scripts.datasets import get_open_buildings_temporal


def build_rooftop_candidate_mask(
    buildings: ee.Image,
    presence_threshold: float = 0.5,
    min_height_m: float = 0.0,
) -> ee.Image:
    """
    Binary mask (0/1) of rooftop candidate pixels from Open Buildings bands.

    Parameters
    ----------
    buildings : ee.Image
        Must include bands building_presence, building_height.
    presence_threshold : float
        Pixels with building_presence > threshold are candidates (default 0.5).
    min_height_m : float
        Require building_height >= this value (m). Use 0 to disable.

    Returns
    -------
    ee.Image
        Single band 'roof_candidate', values 0 or 1, uint8.
    """
    presence = buildings.select("building_presence")
    height = buildings.select("building_height")
    cand = presence.gt(presence_threshold)
    if min_height_m and min_height_m > 0:
        cand = cand.And(height.gte(min_height_m))
    return cand.rename("roof_candidate").toUint8()


def apply_terrain_exclusion(
    roof_mask: ee.Image,
    exclusion_mask: ee.Image,
    buildings: ee.Image,
    scale_m: float = 4.0,
) -> ee.Image:
    """
    Multiply roof mask by terrain exclusion (1 = keep, 0 = exclude steep slopes).

    Reprojects exclusion to the building layer projection for alignment.
    """
    ref = buildings.select("building_presence")
    proj = ref.projection()
    exclusion_repr = exclusion_mask.reproject(crs=proj, scale=scale_m).toFloat()
    # exclusion_mask is 0/1; ensure binary
    exclusion_bin = exclusion_repr.gt(0.5)
    combined = roof_mask.multiply(exclusion_bin.toUint8())
    return combined.rename("roof_candidate").toUint8()


def rooftop_area_m2_reduce(
    roof_mask: ee.Image,
    aoi: ee.Geometry,
    scale_m: float = 4.0,
    tile_scale: int = 4,
    max_pixels: int = 10_000_000,
) -> ee.Dictionary:
    """
    Total area (m^2) of pixels where roof_mask == 1 inside aoi.

    Returns
    -------
    ee.Dictionary
        Keys include 'roof_candidate' with sum in square meters.
    """
    area_img = roof_mask.multiply(ee.Image.pixelArea())
    return area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=scale_m,
        maxPixels=max_pixels,
        tileScale=tile_scale,
    )


def choose_reduce_scale_m(aoi_area_km2: float) -> float:
    """
    Pick reduceRegion scale to limit memory on huge polygons (full-city AOIs).

    Open Buildings is ~4 m; coarser scale underestimates fragmented edges slightly
    but keeps city-wide runs feasible.
    """
    if aoi_area_km2 <= 25:
        return 4.0
    if aoi_area_km2 <= 500:
        return 30.0
    return 100.0


def get_rooftop_area_m2_info(
    aoi: ee.Geometry,
    year: Optional[int] = 2022,
    presence_threshold: float = 0.5,
    min_height_m: float = 0.0,
    exclusion_mask: Optional[ee.Image] = None,
    scale_m: Optional[float] = None,
    tile_scale: int = 4,
) -> Dict[str, Any]:
    """
    End-to-end: load Open Buildings, build mask, optional terrain filter, return plain dict (uses getInfo).

    Use from API / run_analysis; for tests that only need EE objects, call lower-level functions.

    Parameters
    ----------
    scale_m : float, optional
        reduceRegion scale in meters. If None, chosen from AOI area via choose_reduce_scale_m().
    """
    if scale_m is None:
        aoi_km2 = float(aoi.area().divide(1e6).getInfo())
        scale_m = choose_reduce_scale_m(aoi_km2)
    else:
        aoi_km2 = float(aoi.area().divide(1e6).getInfo())

    buildings = get_open_buildings_temporal(aoi, year=year)
    mask = build_rooftop_candidate_mask(
        buildings,
        presence_threshold=presence_threshold,
        min_height_m=min_height_m,
    )
    if exclusion_mask is not None:
        mask = apply_terrain_exclusion(mask, exclusion_mask, buildings, scale_m=scale_m)
    raw = rooftop_area_m2_reduce(
        mask, aoi, scale_m=scale_m, tile_scale=tile_scale
    ).getInfo()
    m2 = raw.get("roof_candidate")
    if m2 is None:
        m2 = 0.0
    return {
        "rooftop_candidate_area_m2": float(m2),
        "open_buildings_year": year,
        "presence_threshold": presence_threshold,
        "min_height_m": min_height_m,
        "terrain_exclusion_applied": exclusion_mask is not None,
        "reduce_scale_m": scale_m,
        "aoi_area_km2": aoi_km2,
        "reduce_region_raw": raw,
    }
