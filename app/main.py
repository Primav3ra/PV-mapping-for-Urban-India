from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.utility import SolarMappingUtils
from scripts.irradiance_baseline import (
    sample_era5_period_ghi_kwh_m2_at_point,
    sample_era5_beam_fraction_at_point,
    ERA5_SCALE_M,
)
from scripts.penalties import (
    net_irradiance_image,
    UHIPenalty, SoilingPenalty, ShadowPenalty,
)
from scripts.solar_geometry import (
    solar_positions_yearly,
    solar_positions_quarterly,
    solar_positions_single_day,
)
from scripts.datasets import get_open_buildings_temporal, get_open_buildings_vector
from scripts.rooftops import build_rooftop_candidate_mask, apply_terrain_exclusion


def square_aoi_from_point(lat: float, lon: float, half_size_deg: float = 0.01) -> List[List[float]]:
    return [
        [lon - half_size_deg, lat - half_size_deg],
        [lon + half_size_deg, lat - half_size_deg],
        [lon + half_size_deg, lat + half_size_deg],
        [lon - half_size_deg, lat + half_size_deg],
        [lon - half_size_deg, lat - half_size_deg],
    ]


def _last_complete_calendar_year() -> int:
    return date.today().year - 1


def _quarter_bounds(year: int, quarter: int) -> Tuple[str, str]:
    if quarter == 1:
        return f"{year}-01-01", f"{year}-04-01"
    if quarter == 2:
        return f"{year}-04-01", f"{year}-07-01"
    if quarter == 3:
        return f"{year}-07-01", f"{year}-10-01"
    if quarter == 4:
        return f"{year}-10-01", f"{year + 1}-01-01"
    raise ValueError("quarter must be 1..4")


def _parse_daily_window(start_date: str, end_date_exclusive: str) -> int:
    d0 = date.fromisoformat(start_date)
    d1 = date.fromisoformat(end_date_exclusive)
    if d1 <= d0:
        raise ValueError("end_date_exclusive must be after start_date")
    return (d1 - d0).days


def resolve_temporal_window(
    baseline_mode: str,
    year: Optional[int],
    quarter: Optional[int],
    start_date: Optional[str],
    end_date_exclusive: Optional[str],
) -> Dict[str, Any]:
    """
    Map UI mode to [start_date, end_date_exclusive) for ERA5 and solar alignment.
    daily: exactly one UTC calendar day (end = start + 1 day).
    """
    ly = _last_complete_calendar_year()
    mode = (baseline_mode or "yearly").lower()
    if mode not in ("yearly", "quarterly", "daily"):
        raise ValueError("baseline_mode must be yearly, quarterly, or daily")
    if mode == "yearly":
        y = year if year is not None else ly
        if y < 2000 or y > ly:
            raise ValueError(f"year must be between 2000 and {ly} (last complete calendar year)")
        s, e = f"{y}-01-01", f"{y + 1}-01-01"
        return {
            "mode": "yearly",
            "start_date": s,
            "end_date_exclusive": e,
            "calendar_year": y,
            "quarter": None,
        }
    if mode == "quarterly":
        y = year if year is not None else ly
        q = quarter if quarter is not None else 2
        if y < 2000 or y > ly:
            raise ValueError(f"year must be between 2000 and {ly}")
        if q < 1 or q > 4:
            raise ValueError("quarter must be 1..4")
        s, e = _quarter_bounds(y, q)
        return {
            "mode": "quarterly",
            "start_date": s,
            "end_date_exclusive": e,
            "calendar_year": y,
            "quarter": q,
        }
    if not start_date or not end_date_exclusive:
        raise ValueError("daily mode requires start_date and end_date_exclusive (ISO YYYY-MM-DD)")
    try:
        nd = _parse_daily_window(start_date, end_date_exclusive)
    except ValueError as ex:
        raise ValueError(str(ex))
    if nd != 1:
        raise ValueError(
            "daily mode requires exactly one calendar day: end_date_exclusive must be start_date + 1 day"
        )
    return {
        "mode": "daily",
        "start_date": start_date,
        "end_date_exclusive": end_date_exclusive,
        "calendar_year": None,
        "quarter": None,
    }


def _centroid_lon_lat(centroid: ee.Geometry) -> Tuple[float, float]:
    g = centroid.getInfo()
    coords = g.get("coordinates")
    if not coords or len(coords) < 2:
        raise RuntimeError("Could not read centroid coordinates")
    return float(coords[0]), float(coords[1])


def _solar_positions_for_window(
    lat_deg: float,
    lon_deg: float,
    win: Dict[str, Any],
) -> List[Tuple[float, float, float]]:
    mode = win["mode"]
    if mode == "yearly":
        y = int(win["calendar_year"])
        pos = solar_positions_yearly(lat_deg, lon_deg, y)
    elif mode == "quarterly":
        pos = solar_positions_quarterly(lat_deg, lon_deg, int(win["calendar_year"]), int(win["quarter"]))
    else:
        d0 = date.fromisoformat(win["start_date"])
        pos = solar_positions_single_day(lat_deg, lon_deg, d0)
    if len(pos) > 42:
        pos = pos[::2]
    return pos


class BaselineRequest(BaseModel):
    project_id: Optional[str] = Field(default_factory=lambda: os.environ.get("GEE_PROJECT_ID", "pv-mapping-india"))
    coordinates: Optional[List[List[float]]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    half_size_deg: float = 0.01
    roof_year: int = 2022
    presence_threshold: float = 0.5
    min_height_m: float = 0.0
    baseline_mode: str = "yearly"  # yearly | quarterly | daily
    year: Optional[int] = None
    quarter: Optional[int] = None
    start_date: Optional[str] = None
    end_date_exclusive: Optional[str] = None


app = FastAPI(title="PV Baseline API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/presets")
def presets() -> Dict[str, Any]:
    ly = _last_complete_calendar_year()
    return {
        "baseline": {
            "modes": ["yearly", "quarterly", "daily"],
            "year_bounds": {"min": 2000, "max": ly, "default": ly},
            "quarter_default": 2,
            "daily_note": "Use start_date and end_date_exclusive in ISO format; end must be start + 1 day (exclusive).",
        },
    }


@app.post("/api/baseline")
def compute_baseline(req: BaselineRequest) -> Dict[str, Any]:
    if req.coordinates is None:
        if req.lat is None or req.lon is None:
            raise HTTPException(status_code=400, detail="Provide either coordinates or lat/lon.")
        coords = square_aoi_from_point(req.lat, req.lon, req.half_size_deg)
    else:
        coords = req.coordinates

    try:
        utils = SolarMappingUtils(req.project_id)
        aoi = ee.Geometry.Polygon(coords)

        dem = utils.get_elevation_data(aoi)
        exclusion = utils.create_exclusion_mask(dem, aoi)

        rooftop = utils.get_rooftop_candidate_stats(
            aoi=aoi,
            exclusion_mask=exclusion,
            year=req.roof_year,
            presence_threshold=req.presence_threshold,
            min_height_m=req.min_height_m,
        )

        try:
            win = resolve_temporal_window(
                req.baseline_mode,
                req.year,
                req.quarter,
                req.start_date,
                req.end_date_exclusive,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        mode = win["mode"]
        s, e = win["start_date"], win["end_date_exclusive"]
        aoibaseline = None
        range_info = None

        if mode == "yearly":
            y = int(win["calendar_year"])
            roof_baseline = utils.get_roof_masked_era5_baseline_stats(
                aoi=aoi,
                exclusion_mask=exclusion,
                roof_year=req.roof_year,
                presence_threshold=req.presence_threshold,
                min_height_m=req.min_height_m,
                start_year=y,
                end_year=y,
            )
            roof_baseline["baseline_time_mode"] = "yearly"
            roof_baseline["calendar_year"] = y
            roof_baseline["start_date"] = s
            roof_baseline["end_date_exclusive"] = e
            aoibaseline = utils.get_era5_baseline_stats(aoi, start_year=y, end_year=y)

        elif mode == "quarterly":
            roof_baseline = utils.get_roof_masked_era5_baseline_for_date_range_stats(
                aoi=aoi,
                exclusion_mask=exclusion,
                roof_year=req.roof_year,
                presence_threshold=req.presence_threshold,
                min_height_m=req.min_height_m,
                start_date=s,
                end_date_exclusive=e,
            )
            roof_baseline["baseline_time_mode"] = "quarterly"
            roof_baseline["calendar_year"] = win["calendar_year"]
            roof_baseline["quarter"] = win["quarter"]
            roof_baseline["start_date"] = s
            roof_baseline["end_date_exclusive"] = e
            range_info = utils.get_era5_range_stats(aoi, start_date=s, end_date_exclusive=e)

        else:
            roof_baseline = utils.get_roof_masked_era5_baseline_for_date_range_stats(
                aoi=aoi,
                exclusion_mask=exclusion,
                roof_year=req.roof_year,
                presence_threshold=req.presence_threshold,
                min_height_m=req.min_height_m,
                start_date=s,
                end_date_exclusive=e,
            )
            roof_baseline["baseline_time_mode"] = "daily"
            roof_baseline["start_date"] = s
            roof_baseline["end_date_exclusive"] = e
            range_info = utils.get_era5_range_stats(aoi, start_date=s, end_date_exclusive=e)

        return {
            "status": "ok",
            "baseline_time_mode": mode,
            "temporal_window": {"start_date": s, "end_date_exclusive": e},
            "aoi_coordinates": coords,
            "rooftop": rooftop,
            "roof_baseline": roof_baseline,
            "aoi_baseline": aoibaseline,
            "range_baseline": range_info,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class YieldRequest(BaseModel):
    project_id: Optional[str] = Field(default_factory=lambda: os.environ.get("GEE_PROJECT_ID", "pv-mapping-india"))
    coordinates: Optional[List[List[float]]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    half_size_deg: float = 0.01
    roof_year: int = 2022
    presence_threshold: float = 0.5
    min_height_m: float = 0.0
    baseline_mode: str = "yearly"
    year: Optional[int] = None
    quarter: Optional[int] = None
    start_date: Optional[str] = None
    end_date_exclusive: Optional[str] = None
    panel_efficiency: float = 0.18
    performance_ratio: float = 0.80
    building_confidence: float = 0.7


@app.post("/api/yield")
def compute_yield(req: YieldRequest) -> Dict[str, Any]:
    """
    Single-building PV energy for the same temporal window as /api/baseline.

    ERA5 GHI is summed over [start_date, end_date_exclusive) at the AOI centroid.
    Shadow retention uses sun positions aligned with that window (year / quarter / day)
    at the centroid latitude and longitude.
    """
    if req.coordinates is None:
        if req.lat is None or req.lon is None:
            raise HTTPException(status_code=400, detail="Provide either coordinates or lat/lon.")
        coords = square_aoi_from_point(req.lat, req.lon, req.half_size_deg)
    else:
        coords = req.coordinates

    try:
        try:
            win = resolve_temporal_window(
                req.baseline_mode,
                req.year,
                req.quarter,
                req.start_date,
                req.end_date_exclusive,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        ee.Initialize(project=req.project_id)
        aoi = ee.Geometry.Polygon(coords)
        centroid = aoi.centroid(1)
        lon_deg, lat_deg = _centroid_lon_lat(centroid)
        s, e = win["start_date"], win["end_date_exclusive"]

        ghi_info = sample_era5_period_ghi_kwh_m2_at_point(centroid, s, e, scale_m=ERA5_SCALE_M)
        regional_ghi_kwh_m2_period = float(ghi_info["value"])
        if ghi_info["source"] in ("no_sample", "null_band"):
            raise HTTPException(status_code=500, detail="Could not sample ERA5 GHI for the selected period at centroid.")

        solar_positions = _solar_positions_for_window(lat_deg, lon_deg, win)

        buildings_raster = get_open_buildings_temporal(aoi, year=req.roof_year)
        building_height = (
            buildings_raster
            .select("building_height")
            .setDefaultProjection(crs="EPSG:4326", scale=4)
        )
        roof_mask = build_rooftop_candidate_mask(
            buildings_raster,
            presence_threshold=req.presence_threshold,
            min_height_m=req.min_height_m,
        )
        dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(aoi)
        exclusion = ee.Terrain.products(dem).select("slope").lt(30)
        roof_mask = apply_terrain_exclusion(roof_mask, exclusion, buildings_raster, scale_m=4.0)

        # Shadow frequency (per-pixel, insolation-weighted, data-driven from building heights)
        shadow_freq = ShadowPenalty.frequency(building_height, solar_positions=solar_positions)

        # Beam fraction: direct / GHI from ERA5 HOURLY -- used to correct shadow losses.
        # Only the beam component is blocked by shadows; diffuse is unaffected for rooftops.
        beam_info = sample_era5_beam_fraction_at_point(centroid, s, e)
        beam_fraction = float(beam_info["beam_fraction"])

        uhi_info = UHIPenalty.stats(aoi, s)
        soiling_info = SoilingPenalty.stats(aoi, s)

        net_irr = net_irradiance_image(
            regional_ghi_kwh_m2_period,
            shadow_freq,
            beam_fraction=beam_fraction,
            uhi_derate=uhi_info["uhi_derate_factor"],
            soiling_retention=soiling_info["soiling_retention_factor"],
        )

        target_building = (
            get_open_buildings_vector(aoi, confidence_threshold=req.building_confidence)
            .filterBounds(centroid)
            .first()
            .getInfo()
        )

        period_label = {"yearly": "calendar_year", "quarterly": "calendar_quarter", "daily": "single_day"}[win["mode"]]

        if target_building is None:
            return {
                "status": "no_building_at_point",
                "message": "No Open Buildings polygon found at the selected point. Try clicking on a rooftop.",
                "regional_ghi_kwh_m2_period": regional_ghi_kwh_m2_period,
                "irradiance_source": "ERA5",
                "baseline_time_mode": win["mode"],
                "start_date": s,
                "end_date_exclusive": e,
                "accounting_period": period_label,
                "geojson": None,
            }

        building_geom = ee.Feature(target_building).geometry()
        building_props = target_building.get("properties", {})

        energy_img = (
            net_irr
            .multiply(roof_mask.toFloat())
            .multiply(ee.Image.pixelArea())
            .rename("energy_kwh_pixel")
        )
        stats = energy_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=building_geom,
            scale=4.0,
            maxPixels=1e7,
        ).getInfo()

        roof_area_stats = (
            roof_mask.toFloat()
            .multiply(ee.Image.pixelArea())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=building_geom,
                scale=4.0,
                maxPixels=1e7,
            )
            .getInfo()
        )

        shadow_stats = (
            shadow_freq
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=building_geom,
                scale=4.0,
                maxPixels=1e7,
            )
            .getInfo()
        )

        total_energy_kwh = (stats.get("energy_kwh_pixel") or 0.0) * req.panel_efficiency * req.performance_ratio
        roof_area_m2 = roof_area_stats.get("roof_candidate") or 0.0
        mean_shadow_frequency = shadow_stats.get("shadow_frequency")
        mean_shadow_fraction = mean_shadow_frequency  # shadow_freq IS the fraction in shadow
        mean_shadow_retention = (
            round(1.0 - mean_shadow_frequency * beam_fraction, 4)
            if mean_shadow_frequency is not None else None
        )
        combined_derate = uhi_info["uhi_derate_factor"] * soiling_info["soiling_retention_factor"]
        net_irr_mean = (
            regional_ghi_kwh_m2_period * combined_derate * mean_shadow_retention
            if mean_shadow_retention is not None else None
        )

        out = {
            "status": "ok",
            "baseline_time_mode": win["mode"],
            "start_date": s,
            "end_date_exclusive": e,
            "accounting_period": period_label,
            "regional_ghi_kwh_m2_period": regional_ghi_kwh_m2_period,
            "ghi_sample_source": ghi_info["source"],
            "irradiance_source": "ERA5",
            "panel_efficiency": req.panel_efficiency,
            "performance_ratio": req.performance_ratio,
            "calendar_year": win["calendar_year"],
            "quarter": win["quarter"],
            "building_confidence": building_props.get("confidence"),
            "building_area_in_meters": building_props.get("area_in_meters"),
            "roof_area_m2": roof_area_m2,
            "mean_shadow_fraction": mean_shadow_fraction,
            "mean_shadow_retention": mean_shadow_retention,
            "beam_fraction": beam_fraction,
            "diffuse_fraction": beam_info["diffuse_fraction"],
            "beam_fraction_source": beam_info["source"],
            "uhi_derate_factor": uhi_info["uhi_derate_factor"],
            "delta_t_uhi_celsius": uhi_info["delta_t_uhi_celsius"],
            "soiling_retention_factor": soiling_info["soiling_retention_factor"],
            "mean_aod_550nm": soiling_info["mean_aod_550nm"],
            "combined_derate_factor": round(combined_derate, 5),
            "net_irradiance_kwh_m2_period": net_irr_mean,
            "period_yield_kwh": total_energy_kwh,
            "uhi_penalty": uhi_info,
            "soiling_penalty": soiling_info,
            "geojson": {
                "type": "FeatureCollection",
                "features": [{
                    **target_building,
                    "properties": {
                        **building_props,
                        "roof_area_m2": roof_area_m2,
                        "mean_shadow_fraction": mean_shadow_fraction,
                        "uhi_derate_factor": uhi_info["uhi_derate_factor"],
                        "soiling_retention_factor": soiling_info["soiling_retention_factor"],
                        "net_irradiance_kwh_m2_period": net_irr_mean,
                        "period_yield_kwh": total_energy_kwh,
                    }
                }]
            },
        }
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

