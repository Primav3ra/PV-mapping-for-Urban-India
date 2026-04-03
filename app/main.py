from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Optional

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.utility import SolarMappingUtils
from scripts.irradiance_baseline import (
    latest_complete_5y_range,
    era5_mean_annual_ghi_kwh_m2,
    ERA5_SCALE_M,
)
from scripts.penalties import shadow_retention_fraction, net_irradiance_image, per_building_yield
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


class BaselineRequest(BaseModel):
    project_id: Optional[str] = Field(default_factory=lambda: os.environ.get("GEE_PROJECT_ID", "pv-mapping-india"))
    coordinates: Optional[List[List[float]]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    half_size_deg: float = 0.01
    roof_year: int = 2022
    presence_threshold: float = 0.5
    min_height_m: float = 0.0
    baseline_mode: str = "latest5y"  # latest5y | years | range
    start_year: Optional[int] = None
    end_year: Optional[int] = None
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
    y0, y1 = latest_complete_5y_range()
    current = date.today().year
    return {
        "latest5y": {"start_year": y0, "end_year": y1},
        "seasonal_examples": {
            "summer": {"start_date": f"{current-1}-04-01", "end_date_exclusive": f"{current-1}-07-01"},
            "monsoon": {"start_date": f"{current-1}-07-01", "end_date_exclusive": f"{current-1}-10-01"},
            "single_day": {"start_date": f"{current-1}-06-21", "end_date_exclusive": f"{current-1}-06-22"},
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

        roof_baseline = utils.get_roof_masked_merra_baseline_stats(
            aoi=aoi,
            exclusion_mask=exclusion,
            roof_year=req.roof_year,
            presence_threshold=req.presence_threshold,
            min_height_m=req.min_height_m,
            start_year=req.start_year,
            end_year=req.end_year,
        )

        if req.baseline_mode == "years":
            if req.start_year is None or req.end_year is None:
                raise HTTPException(status_code=400, detail="years mode requires start_year and end_year")
            aoibaseline = utils.get_merra_baseline_stats(aoi, start_year=req.start_year, end_year=req.end_year)
            range_info = None
        elif req.baseline_mode == "range":
            if not req.start_date or not req.end_date_exclusive:
                raise HTTPException(status_code=400, detail="range mode requires start_date and end_date_exclusive")
            range_info = utils.get_merra_range_stats(
                aoi, start_date=req.start_date, end_date_exclusive=req.end_date_exclusive
            )
            aoibaseline = None
        else:
            aoibaseline = utils.get_merra_latest_5y_baseline_stats(aoi)
            range_info = None

        return {
            "status": "ok",
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
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    panel_efficiency: float = 0.18
    performance_ratio: float = 0.80
    building_confidence: float = 0.7


@app.post("/api/yield")
def compute_yield(req: YieldRequest) -> Dict[str, Any]:
    """
    Single-building PV yield at the AOI centroid.

    Finds the one Open Buildings polygon that contains the clicked point,
    then runs all penalty layers (shadow, and later UHI/soiling) on just
    that one geometry. O(1) GEE calls regardless of AOI size or penalty count.

    Pipeline:
      1. ERA5 regional GHI (kWh/m2/year) -- one centroid sample (~9 km)
      2. Shadow retention image from 2.5D building heights (4m)
      3. net_irradiance = baseline x shadow_retention (per pixel)
      4. Find the single building polygon at the clicked point
      5. sum(net_irradiance x roof_mask x pixelArea) x efficiency x PR
         = annual_yield_kwh for that building
    """
    if req.coordinates is None:
        if req.lat is None or req.lon is None:
            raise HTTPException(status_code=400, detail="Provide either coordinates or lat/lon.")
        coords = square_aoi_from_point(req.lat, req.lon, req.half_size_deg)
    else:
        coords = req.coordinates

    try:
        ee.Initialize(project=req.project_id)
        aoi = ee.Geometry.Polygon(coords)
        centroid = aoi.centroid(1)  # the clicked point

        # -- year range --
        start_year = req.start_year
        end_year = req.end_year
        if start_year is None or end_year is None:
            start_year, end_year = latest_complete_5y_range()

        # -- 1. regional GHI at centroid (ERA5, ~9 km) --
        baseline_img = era5_mean_annual_ghi_kwh_m2(aoi, start_year=start_year, end_year=end_year)
        irr_sample = (
            baseline_img
            .sample(region=centroid, scale=ERA5_SCALE_M, numPixels=1, geometries=False)
            .first()
            .getInfo()
        )
        if irr_sample is None or "properties" not in irr_sample:
            raise HTTPException(status_code=500, detail="Could not sample ERA5 irradiance at centroid.")
        regional_irradiance = float(irr_sample["properties"].get("annual_GHI_kWh_m2", 0.0))

        # -- 2. building raster (raster for shadow model + roof mask) --
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

        # -- 3. net irradiance image (shadow penalty applied) --
        retention = shadow_retention_fraction(building_height)
        net_irr = net_irradiance_image(regional_irradiance, retention)

        # -- 4. find the single building polygon at the centroid --
        target_building = (
            get_open_buildings_vector(aoi, confidence_threshold=req.building_confidence)
            .filterBounds(centroid)   # only polygons that contain the clicked point
            .first()
            .getInfo()
        )

        if target_building is None:
            return {
                "status": "no_building_at_point",
                "message": "No Open Buildings polygon found at the selected point. Try clicking on a rooftop.",
                "regional_irradiance_kwh_m2_year": regional_irradiance,
                "irradiance_source": "ERA5",
                "start_year": start_year,
                "end_year": end_year,
                "geojson": None,
            }

        building_geom = ee.Feature(target_building).geometry()
        building_props = target_building.get("properties", {})

        # -- 5. compute yield over that one building polygon --
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
            retention
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
        mean_shadow_retention = shadow_stats.get("shadow_retention")
        mean_shadow_fraction = (1.0 - mean_shadow_retention) if mean_shadow_retention is not None else None
        net_irradiance_mean = (regional_irradiance * mean_shadow_retention) if mean_shadow_retention is not None else None

        return {
            "status": "ok",
            "regional_irradiance_kwh_m2_year": regional_irradiance,
            "irradiance_source": "ERA5",
            "start_year": start_year,
            "end_year": end_year,
            "panel_efficiency": req.panel_efficiency,
            "performance_ratio": req.performance_ratio,
            # -- building identity --
            "building_confidence": building_props.get("confidence"),
            "building_area_in_meters": building_props.get("area_in_meters"),
            # -- computed outputs --
            "roof_area_m2": roof_area_m2,
            "mean_shadow_fraction": mean_shadow_fraction,
            "mean_shadow_retention": mean_shadow_retention,
            "net_irradiance_kwh_m2_year": net_irradiance_mean,
            "annual_yield_kwh": total_energy_kwh,
            # -- GeoJSON of the single building for map rendering --
            "geojson": {
                "type": "FeatureCollection",
                "features": [{
                    **target_building,
                    "properties": {
                        **building_props,
                        "roof_area_m2": roof_area_m2,
                        "mean_shadow_fraction": mean_shadow_fraction,
                        "net_irradiance_kwh_m2_year": net_irradiance_mean,
                        "annual_yield_kwh": total_energy_kwh,
                    }
                }]
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

