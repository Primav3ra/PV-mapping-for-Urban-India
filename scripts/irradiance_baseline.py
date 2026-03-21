"""
Baseline surface incoming shortwave from MERRA-2 reanalysis (hourly SWGDN).

Collection: NASA/GSFC/MERRA/rad/2, band SWGDN (W/m^2), hourly means.
Annual energy density (kWh/m^2/year): sum over all hours in interval of SWGDN/1000
per year-mean over inclusive [start_year, end_year].

Spatial resolution is coarse (~50-70 km); values are regional climatology, not rooftop-scale.
"""

from __future__ import annotations

import ee
from typing import Any, Dict, Optional


MERRA_RAD_COLLECTION = "NASA/GSFC/MERRA/rad/2"
SWGDN_BAND = "SWGDN"


def merra2_mean_annual_sw_kwh_m2(
    aoi: ee.Geometry,
    start_year: int = 2015,
    end_year: int = 2019,
) -> ee.Image:
    """
    Mean annual surface incoming shortwave (all-sky) in kWh/m^2/year.

    For each hour: incremental energy ~ (SWGDN W/m^2) * 1 h = SWGDN/1000 kWh/m^2.
    Sums all hours from start_year-01-01 through end_year-12-31, divides by year count.
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    start = ee.Date.fromYMD(start_year, 1, 1)
    end = ee.Date.fromYMD(end_year + 1, 1, 1)
    n_years = end_year - start_year + 1

    col = (
        ee.ImageCollection(MERRA_RAD_COLLECTION)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select(SWGDN_BAND)
    )

    total_kwh_m2 = col.sum().divide(1000.0)
    mean_annual = total_kwh_m2.divide(n_years).rename("annual_SWGDN_kWh_m2")
    return mean_annual.clip(aoi)


def reduce_mean_annual_sw_at_aoi(
    aoi: ee.Geometry,
    start_year: int = 2015,
    end_year: int = 2019,
    scale_m: float = 50_000.0,
    tile_scale: int = 2,
) -> ee.Dictionary:
    """Spatial mean of mean-annual kWh/m^2 image over AOI (MERRA-native scale is coarse)."""
    img = merra2_mean_annual_sw_kwh_m2(aoi, start_year, end_year)
    return img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale_m,
        maxPixels=1e9,
        tileScale=tile_scale,
    )


def get_merra_baseline_info(
    aoi: ee.Geometry,
    start_year: int = 2015,
    end_year: int = 2019,
    scale_m: float = 50_000.0,
) -> Dict[str, Any]:
    """Plain dict for API / run_analysis (calls getInfo once)."""
    raw = reduce_mean_annual_sw_at_aoi(
        aoi, start_year, end_year, scale_m=scale_m
    ).getInfo()
    val = raw.get("annual_SWGDN_kWh_m2")
    return {
        "merra_mean_annual_sw_kwh_m2": float(val) if val is not None else None,
        "merra_start_year": start_year,
        "merra_end_year": end_year,
        "merra_collection": MERRA_RAD_COLLECTION,
        "merra_band": SWGDN_BAND,
        "reduce_scale_m": scale_m,
        "reduce_region_raw": raw,
    }
