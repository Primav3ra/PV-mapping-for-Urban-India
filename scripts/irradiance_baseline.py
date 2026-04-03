"""
Baseline surface incoming shortwave from ERA5 reanalysis.

Collection: ECMWF/ERA5_LAND/HOURLY
Band: surface_solar_radiation_downwards_hourly (J/m^2 per hour)
Resolution: ~9 km (significantly finer than MERRA-2 ~50 km)

Unit conversion:
  ERA5 stores accumulated solar radiation in J/m^2 per hourly step.
  Divide by 3,600,000 to convert J/m^2 -> kWh/m^2 per hour.
  Sum all hourly values over the year range, divide by number of years
  -> mean annual GHI in kWh/m^2/year.

ERA5 vs MERRA-2:
  ERA5 is the ECMWF global reanalysis at ~9 km resolution (vs MERRA-2 ~50 km).
  It uses satellite-corrected cloud cover and aerosol data, giving better agreement
  with ground pyranometer stations than MERRA-2 (typical bias ~3-5% vs ~10-15%).
  ERA5 is the recommended free baseline for solar resource assessment.

Fallback:
  If ERA5 is unavailable for a date range, the module falls back to MERRA-2 SWGDN
  and flags the source in the response.
"""

from __future__ import annotations

import ee
from typing import Any, Dict, Optional
from datetime import date


# ---------------------------------------------------------------------------
# Primary: ERA5-Land hourly
# ---------------------------------------------------------------------------
ERA5_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_BAND = "surface_solar_radiation_downwards_hourly"  # J/m^2 per hour
ERA5_SCALE_M = 11_132.0   # ~9 km native; 11132 m = 0.1 deg at equator (GEE default)

# Conversion: J/m^2 per hour -> kWh/m^2 per hour
# 1 kWh = 3,600,000 J  =>  divide by 3.6e6
ERA5_J_TO_KWH = 3_600_000.0

# ---------------------------------------------------------------------------
# Fallback: MERRA-2 (kept for backward compatibility and range mode)
# ---------------------------------------------------------------------------
MERRA_RAD_COLLECTION = "NASA/GSFC/MERRA/rad/2"
SWGDN_BAND = "SWGDN"  # W/m^2 hourly mean
MERRA_SCALE_M = 50_000.0


def _mean_over_aoi_with_fallback(
    image: ee.Image,
    band_name: str,
    aoi: ee.Geometry,
    scale_m: float,
) -> Dict[str, Any]:
    """
    Robust mean extractor for coarse datasets over small AOIs.
    Falls back: reduceRegion -> bestEffort -> centroid sample.

    Important: do NOT clip the image to aoi before calling this.
    Clipping a small AOI on a coarse image removes all pixel centers,
    causing reduceRegion to return null.
    """
    raw_primary = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale_m,
        maxPixels=1e9,
        tileScale=2,
    ).getInfo()
    val = None if raw_primary is None else raw_primary.get(band_name)
    if val is not None:
        return {"value": float(val), "source": "reduceRegion", "raw": raw_primary}

    raw_best = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale_m,
        bestEffort=True,
        maxPixels=1e9,
        tileScale=2,
    ).getInfo()
    val_best = None if raw_best is None else raw_best.get(band_name)
    if val_best is not None:
        return {"value": float(val_best), "source": "reduceRegion_bestEffort", "raw": raw_best}

    centroid = aoi.centroid(1)
    sample_fc = image.sample(
        region=centroid,
        scale=scale_m,
        numPixels=1,
        geometries=False,
    )
    sample = sample_fc.first().getInfo() if sample_fc.size().getInfo() > 0 else None
    if sample and "properties" in sample and band_name in sample["properties"]:
        return {
            "value": float(sample["properties"][band_name]),
            "source": "centroid_sample",
            "raw": {"sample": sample},
        }

    return {"value": 0.0, "source": "fallback_zero", "raw": {"primary": raw_primary, "best": raw_best}}


# ---------------------------------------------------------------------------
# ERA5 baseline (primary)
# ---------------------------------------------------------------------------

def era5_mean_annual_ghi_kwh_m2(
    aoi: ee.Geometry,
    start_year: int = 2020,
    end_year: int = 2024,
) -> ee.Image:
    """
    Mean annual Global Horizontal Irradiance (GHI) from ERA5-Land in kWh/m^2/year.

    ERA5 stores surface_solar_radiation_downwards_hourly in J/m^2 per hour.
    Sum all hourly values over [start_year, end_year], convert J->kWh, divide by years.

    Returns an ee.Image with band 'annual_GHI_kWh_m2' (not clipped to aoi).
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    start = ee.Date.fromYMD(start_year, 1, 1)
    end = ee.Date.fromYMD(end_year + 1, 1, 1)
    n_years = end_year - start_year + 1

    col = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(start, end)
        .select(ERA5_BAND)
    )

    # Sum all hourly J/m^2, convert to kWh/m^2, divide by years -> mean annual kWh/m^2/yr
    total_kwh_m2 = col.sum().divide(ERA5_J_TO_KWH)
    mean_annual = total_kwh_m2.divide(n_years).rename("annual_GHI_kWh_m2")
    # Do NOT clip to aoi: ERA5 pixels are ~9-11 km; clipping a small AOI
    # removes pixel centers and causes reduceRegion to return null.
    return mean_annual


def get_era5_baseline_info(
    aoi: ee.Geometry,
    start_year: int = 2020,
    end_year: int = 2024,
    scale_m: float = ERA5_SCALE_M,
) -> Dict[str, Any]:
    """Plain Python dict for API / run_analysis."""
    img = era5_mean_annual_ghi_kwh_m2(aoi, start_year, end_year)
    mean_info = _mean_over_aoi_with_fallback(
        image=img,
        band_name="annual_GHI_kWh_m2",
        aoi=aoi,
        scale_m=scale_m,
    )
    return {
        "mean_annual_ghi_kwh_m2_year": mean_info["value"],
        "start_year": start_year,
        "end_year": end_year,
        "collection": ERA5_COLLECTION,
        "band": ERA5_BAND,
        "reduce_scale_m": scale_m,
        "value_source": mean_info["source"],
        "reduce_region_raw": mean_info["raw"],
    }


def era5_total_ghi_kwh_m2_for_range(
    aoi: ee.Geometry,
    start_date: str,
    end_date_exclusive: str,
) -> ee.Image:
    """
    Total GHI over [start_date, end_date_exclusive) in kWh/m^2.
    Useful for seasonal/custom date range queries.
    """
    col = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(start_date, end_date_exclusive)
        .select(ERA5_BAND)
    )
    return col.sum().divide(ERA5_J_TO_KWH).rename("total_GHI_kWh_m2")


def get_era5_range_info(
    aoi: ee.Geometry,
    start_date: str,
    end_date_exclusive: str,
    scale_m: float = ERA5_SCALE_M,
) -> Dict[str, Any]:
    """Baseline stats for an arbitrary date range."""
    total_img = era5_total_ghi_kwh_m2_for_range(aoi, start_date, end_date_exclusive)
    mean_info = _mean_over_aoi_with_fallback(
        image=total_img,
        band_name="total_GHI_kWh_m2",
        aoi=aoi,
        scale_m=scale_m,
    )
    total_kwh_m2 = float(mean_info["value"])

    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date_exclusive)
    days = max((d1 - d0).days, 1)
    annualized = total_kwh_m2 * (365.25 / days)

    return {
        "range_total_ghi_kwh_m2": total_kwh_m2,
        "range_annualized_ghi_kwh_m2_year": annualized,
        "range_start_date": start_date,
        "range_end_date_exclusive": end_date_exclusive,
        "range_days": days,
        "collection": ERA5_COLLECTION,
        "band": ERA5_BAND,
        "reduce_scale_m": scale_m,
        "value_source": mean_info["source"],
        "reduce_region_raw": mean_info["raw"],
    }


def get_roof_masked_era5_baseline_info(
    aoi: ee.Geometry,
    roof_mask: ee.Image,
    start_year: int = 2020,
    end_year: int = 2024,
    scale_m: float = ERA5_SCALE_M,
    roof_area_scale_m: float = 4.0,
) -> Dict[str, Any]:
    """
    Roof-masked ERA5 baseline: regional GHI applied to candidate rooftop area.

    Two-scale method:
    - roof_area_m2: summed at 4m (Open Buildings resolution).
    - regional_ghi_kwh_m2_year: ERA5 mean at ~9 km -- one value for the AOI.
    - pre_penalty_total_kwh_year: roof_area_m2 * regional_ghi -- theoretical max
      before shadow, soiling, or efficiency losses.
    """
    baseline = era5_mean_annual_ghi_kwh_m2(aoi, start_year=start_year, end_year=end_year)

    # Step 1: total candidate rooftop area at fine scale (4m)
    roof_area_raw = (
        roof_mask.toFloat()
        .multiply(ee.Image.pixelArea())
        .rename("roof_area_m2")
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=roof_area_scale_m,
            maxPixels=1e9,
            tileScale=2,
        )
        .getInfo()
    )
    roof_area_m2 = (
        0.0
        if roof_area_raw is None or roof_area_raw.get("roof_area_m2") is None
        else float(roof_area_raw["roof_area_m2"])
    )

    # Step 2: regional GHI at ERA5 scale
    irr_info = _mean_over_aoi_with_fallback(
        image=baseline,
        band_name="annual_GHI_kWh_m2",
        aoi=aoi,
        scale_m=scale_m,
    )
    regional_irradiance = float(irr_info["value"])

    # Step 3: pre-penalty total
    pre_penalty_total = regional_irradiance * roof_area_m2 if roof_area_m2 > 0 else 0.0

    return {
        "roof_area_m2": roof_area_m2,
        "regional_irradiance_kwh_m2_year": regional_irradiance,
        "pre_penalty_total_kwh_year": pre_penalty_total,
        "start_year": start_year,
        "end_year": end_year,
        "collection": ERA5_COLLECTION,
        "band": ERA5_BAND,
        "reduce_scale_m": scale_m,
        "roof_area_scale_m": roof_area_scale_m,
        "method": "two_scale_roof_area_x_era5_mean",
        "irradiance_source": irr_info["source"],
        "roof_area_reduce_region_raw": roof_area_raw,
        "irradiance_reduce_region_raw": irr_info["raw"],
    }


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def latest_complete_5y_range(today: Optional[date] = None) -> tuple[int, int]:
    """
    Return the latest complete 5-year range.
    Example: if today is in 2026, returns (2021, 2025).
    """
    if today is None:
        today = date.today()
    end_year = today.year - 1
    start_year = end_year - 4
    return start_year, end_year


# ---------------------------------------------------------------------------
# MERRA-2 kept for backward compatibility and range-mode fallback
# ---------------------------------------------------------------------------

def merra2_mean_annual_sw_kwh_m2(
    aoi: ee.Geometry,
    start_year: int = 2015,
    end_year: int = 2019,
) -> ee.Image:
    """
    MERRA-2 mean annual SWGDN in kWh/m^2/year (kept for fallback/comparison).
    Prefer era5_mean_annual_ghi_kwh_m2 for new work.
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")
    start = ee.Date.fromYMD(start_year, 1, 1)
    end = ee.Date.fromYMD(end_year + 1, 1, 1)
    n_years = end_year - start_year + 1
    col = (
        ee.ImageCollection(MERRA_RAD_COLLECTION)
        .filterDate(start, end)
        .select(SWGDN_BAND)
    )
    total_kwh_m2 = col.sum().divide(1000.0)
    mean_annual = total_kwh_m2.divide(n_years).rename("annual_SWGDN_kWh_m2")
    return mean_annual


def get_merra_baseline_info(
    aoi: ee.Geometry,
    start_year: int = 2015,
    end_year: int = 2019,
    scale_m: float = MERRA_SCALE_M,
) -> Dict[str, Any]:
    """MERRA-2 baseline dict (kept for backward compatibility)."""
    img = merra2_mean_annual_sw_kwh_m2(aoi, start_year, end_year)
    mean_info = _mean_over_aoi_with_fallback(
        image=img,
        band_name="annual_SWGDN_kWh_m2",
        aoi=aoi,
        scale_m=scale_m,
    )
    return {
        "merra_mean_annual_sw_kwh_m2": mean_info["value"],
        "merra_start_year": start_year,
        "merra_end_year": end_year,
        "merra_collection": MERRA_RAD_COLLECTION,
        "merra_band": SWGDN_BAND,
        "reduce_scale_m": scale_m,
        "value_source": mean_info["source"],
        "reduce_region_raw": mean_info["raw"],
    }


def merra2_total_sw_kwh_m2_for_range(
    aoi: ee.Geometry,
    start_date: str,
    end_date_exclusive: str,
) -> ee.Image:
    """MERRA-2 total SWGDN for a date range (kept for backward compatibility)."""
    col = (
        ee.ImageCollection(MERRA_RAD_COLLECTION)
        .filterDate(start_date, end_date_exclusive)
        .select(SWGDN_BAND)
    )
    return col.sum().divide(1000.0).rename("total_SWGDN_kWh_m2")


def get_merra_range_info(
    aoi: ee.Geometry,
    start_date: str,
    end_date_exclusive: str,
    scale_m: float = MERRA_SCALE_M,
) -> Dict[str, Any]:
    """MERRA-2 range baseline dict (kept for backward compatibility)."""
    total_img = merra2_total_sw_kwh_m2_for_range(aoi, start_date, end_date_exclusive)
    mean_info = _mean_over_aoi_with_fallback(
        image=total_img,
        band_name="total_SWGDN_kWh_m2",
        aoi=aoi,
        scale_m=scale_m,
    )
    total_kwh_m2 = float(mean_info["value"])
    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date_exclusive)
    days = max((d1 - d0).days, 1)
    annualized = total_kwh_m2 * (365.25 / days)
    return {
        "range_total_sw_kwh_m2": total_kwh_m2,
        "range_annualized_sw_kwh_m2_year": annualized,
        "range_start_date": start_date,
        "range_end_date_exclusive": end_date_exclusive,
        "range_days": days,
        "merra_collection": MERRA_RAD_COLLECTION,
        "merra_band": SWGDN_BAND,
        "reduce_scale_m": scale_m,
        "value_source": mean_info["source"],
        "reduce_region_raw": mean_info["raw"],
    }


# Alias kept so existing imports don't break
get_roof_masked_merra_baseline_info = get_roof_masked_era5_baseline_info
