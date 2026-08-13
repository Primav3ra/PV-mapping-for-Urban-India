"""
Rooftop candidate mask + area, off the Open Buildings 2.5D layer.

A few defaults worth knowing: presence > 0.5 is just a prior (the model's confidence
isn't calibrated, so tune it if needed); min_height_m lets you drop low/noise structures
but is off by default; and we reduce at 4 m to match Open Buildings' effective resolution.
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
    0/1 mask of likely-rooftop pixels: presence over the threshold, and (optionally)
    tall enough. `buildings` needs the presence + height bands. Band out: 'roof_candidate'.
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
    AND the roof mask with the terrain keep-mask (drops steep slopes). Reprojects the
    exclusion onto the building layer's grid first so the pixels line up.
    """
    ref = buildings.select("building_presence")
    proj = ref.projection()
    exclusion_repr = exclusion_mask.reproject(crs=proj, scale=scale_m).toFloat()
    exclusion_bin = exclusion_repr.gt(0.5)   # re-binarize after the resample
    combined = roof_mask.multiply(exclusion_bin.toUint8())
    return combined.rename("roof_candidate").toUint8()


def rooftop_area_m2_reduce(
    roof_mask: ee.Image,
    aoi: ee.Geometry,
    scale_m: float = 4.0,
    tile_scale: int = 4,
    max_pixels: int = 10_000_000,
) -> ee.Dictionary:
    """Total roof area (m^2) inside aoi -- sum of pixel areas where the mask is 1.
    Comes back as an ee.Dictionary keyed 'roof_candidate'."""
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
    Coarsen the reduce scale on big AOIs so city-wide runs don't blow up on memory.
    Native is ~4 m; going coarser nibbles a bit off fragmented edges but keeps it running.
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
    The whole thing end to end -> plain dict (calls getInfo). Loads buildings, builds the
    mask, optionally filters terrain, sums the area. This is the API-facing helper; tests
    that just want EE objects should call the lower-level functions instead. Leave scale_m
    as None to let choose_reduce_scale_m() pick one from the AOI size.
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
