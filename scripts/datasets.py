"""
One place for all the Earth Engine dataset loaders + their catalog IDs, so the IDs
aren't scattered across the codebase.
"""
from __future__ import annotations

import ee
from typing import Optional
from datetime import datetime


CATALOG = {
    "srtm_dem": "USGS/SRTMGL1_003",
    "fabdem": "projects/sat-io/open-datasets/FABDEM",
    "open_buildings_temporal": "GOOGLE/Research/open-buildings-temporal/v1",
    "open_buildings_vector": "GOOGLE/Research/open-buildings/v3/polygons",
}


def get_dem(aoi: ee.Geometry, dem_type: str = "srtm") -> ee.Image:
    """Return elevation (metres) clipped to aoi. dem_type: 'srtm' or 'fabdem'."""
    if dem_type == "srtm":
        return ee.Image(CATALOG["srtm_dem"]).select("elevation").clip(aoi)
    if dem_type == "fabdem":
        return (
            ee.ImageCollection(CATALOG["fabdem"])
            .filterBounds(aoi)
            .mosaic()
            .select(0)
            .rename("elevation")
            .clip(aoi)
        )
    raise ValueError(f"dem_type must be 'srtm' or 'fabdem', got: {dem_type}")


def get_open_buildings_temporal(aoi: ee.Geometry, year: Optional[int] = None) -> ee.Image:
    """
    Open Buildings 2.5D Temporal mosaic over aoi (bands: presence, height,
    fractional_count). Pass a year (2016-2023) to pin the vintage; otherwise you get the
    latest.
    """
    col = ee.ImageCollection(CATALOG["open_buildings_temporal"]).filterBounds(aoi)
    if year is not None:
        start_ms = int(datetime(year, 1, 1).timestamp() * 1000)
        end_ms = int(datetime(year + 1, 1, 1).timestamp() * 1000)
        col = col.filter(
            ee.Filter.And(
                ee.Filter.gte("system:time_start", start_ms),
                ee.Filter.lt("system:time_start", end_ms),
            )
        )
    return (
        col.mosaic()
        .clip(aoi)
        .select(["building_presence", "building_height", "building_fractional_count"])
    )


def get_open_buildings_vector(
    aoi: ee.Geometry,
    confidence_threshold: float = 0.7,
) -> ee.FeatureCollection:
    """Open Buildings v3 polygons filtered to aoi and confidence >= threshold."""
    return (
        ee.FeatureCollection(CATALOG["open_buildings_vector"])
        .filterBounds(aoi)
        .filter(ee.Filter.gte("confidence", confidence_threshold))
    )
