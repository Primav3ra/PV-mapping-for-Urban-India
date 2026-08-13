from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Literal

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scripts.utility import SolarMappingUtils
from scripts.irradiance_baseline import (
    sample_era5_period_ghi_kwh_m2_at_point,
    sample_era5_beam_fraction_at_point,
    sample_era5_period_ghi_multi,
    sample_era5_beam_multi,
    ERA5_SCALE_M,
    _ERA5_HOURLY_SCALE_M,
)
from scripts.penalties import (
    net_irradiance_image,
    UHIPenalty, SoilingPenalty, ShadowPenalty, SkyViewFactor,
)
from scripts.solar_geometry import (
    solar_positions_yearly,
    solar_positions_quarterly,
    solar_positions_monthly,
    solar_positions_single_day,
)
from scripts.datasets import get_dem, get_open_buildings_temporal, get_open_buildings_vector
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
    month: Optional[int],
    start_date: Optional[str],
    end_date_exclusive: Optional[str],
) -> Dict[str, Any]:
    """
    Map UI mode to [start_date, end_date_exclusive) for ERA5 and solar alignment.
    monthly: one UTC calendar month.
    daily: exactly one UTC calendar day (end = start + 1 day).
    """
    ly = _last_complete_calendar_year()
    mode = (baseline_mode or "yearly").lower()
    if mode not in ("yearly", "quarterly", "monthly", "daily"):
        raise ValueError("baseline_mode must be yearly, quarterly, monthly, or daily")
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
            "month": None,
        }
    if mode == "monthly":
        y = year if year is not None else ly
        m = month if month is not None else 1
        if y < 2000 or y > ly:
            raise ValueError(f"year must be between 2000 and {ly}")
        if m < 1 or m > 12:
            raise ValueError("month must be 1..12")
        s = f"{y}-{m:02d}-01"
        e = f"{y + 1}-01-01" if m == 12 else f"{y}-{m + 1:02d}-01"
        return {
            "mode": "monthly",
            "start_date": s,
            "end_date_exclusive": e,
            "calendar_year": y,
            "quarter": None,
            "month": m,
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
        "month": None,
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
    elif mode == "monthly":
        pos = solar_positions_monthly(lat_deg, lon_deg, int(win["calendar_year"]), int(win["month"]))
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
    month: Optional[int] = None
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
            "modes": ["yearly", "quarterly", "monthly", "daily"],
            "year_bounds": {"min": 2000, "max": ly, "default": ly},
            "quarter_default": 2,
            "month_default": 1,
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
                req.month,
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

        elif mode == "monthly":
            roof_baseline = utils.get_roof_masked_era5_baseline_for_date_range_stats(
                aoi=aoi,
                exclusion_mask=exclusion,
                roof_year=req.roof_year,
                presence_threshold=req.presence_threshold,
                min_height_m=req.min_height_m,
                start_date=s,
                end_date_exclusive=e,
            )
            roof_baseline["baseline_time_mode"] = "monthly"
            roof_baseline["calendar_year"] = win["calendar_year"]
            roof_baseline["month"] = win["month"]
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
    month: Optional[int] = None
    start_date: Optional[str] = None
    end_date_exclusive: Optional[str] = None
    panel_efficiency: float = 0.18
    performance_ratio: float = 0.80
    packing_factor: float = 0.7  # usable-roof coverage fraction: panels never tile 100% of a roof
                                 # (setbacks, obstructions, water tanks, access). Typical 0.6-0.75.
    building_confidence: float = 0.7


class TilesRequest(BaseModel):
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
    month: Optional[int] = None
    start_date: Optional[str] = None
    end_date_exclusive: Optional[str] = None
    layer: Literal["roof_mask", "shadow_frequency", "sky_view_factor", "net_irradiance", "combined_derate", "temperature_delta"] = "roof_mask"


class BuildingsRequest(BaseModel):
    project_id: Optional[str] = Field(default_factory=lambda: os.environ.get("GEE_PROJECT_ID", "pv-mapping-india"))
    coordinates: Optional[List[List[float]]] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    half_size_deg: float = 0.01
    building_confidence: float = 0.7
    limit: int = 400


def _aoi_from_req(req: Any) -> Tuple[List[List[float]], ee.Geometry]:
    if getattr(req, "coordinates", None) is None:
        if getattr(req, "lat", None) is None or getattr(req, "lon", None) is None:
            raise HTTPException(status_code=400, detail="Provide either coordinates or lat/lon.")
        coords = square_aoi_from_point(float(req.lat), float(req.lon), float(req.half_size_deg))
    else:
        coords = req.coordinates
    return coords, ee.Geometry.Polygon(coords)


_EE_INIT_PROJECT: Optional[str] = None


def _ensure_ee(project_id: str) -> None:
    """
    Initialize Earth Engine once per process (re-init only if the project changes).
    Avoids a redundant ee.Initialize round-trip on every request.
    """
    global _EE_INIT_PROJECT
    if _EE_INIT_PROJECT != project_id:
        ee.Initialize(project=project_id)
        _EE_INIT_PROJECT = project_id


def _build_roof_layers(
    aoi: ee.Geometry,
    roof_year: Optional[int],
    presence_threshold: float,
    min_height_m: float,
) -> Tuple[ee.Image, ee.Image, ee.Image]:
    """
    Shared rooftop-layer construction for /api/yield, /api/tiles and /api/series.
    Returns (buildings_raster, building_height, roof_mask) with terrain exclusion applied.
    """
    buildings_raster = get_open_buildings_temporal(aoi, year=roof_year)
    building_height = (
        buildings_raster
        .select("building_height")
        .setDefaultProjection(crs="EPSG:4326", scale=4)
    )
    roof_mask = build_rooftop_candidate_mask(
        buildings_raster,
        presence_threshold=presence_threshold,
        min_height_m=min_height_m,
    )
    exclusion = ee.Terrain.products(get_dem(aoi, "srtm")).select("slope").lt(30)
    roof_mask = apply_terrain_exclusion(roof_mask, exclusion, buildings_raster, scale_m=4.0)
    return buildings_raster, building_height, roof_mask


def _select_target_building(
    aoi: ee.Geometry,
    coords: List[List[float]],
    centroid: ee.Geometry,
    confidence: float,
) -> Tuple[ee.Geometry, Dict[str, Any], Dict[str, Any], str, Optional[str]]:
    """
    Pick the Open Buildings polygon at the AOI centroid, with buffer + AOI fallbacks.
    Returns (building_geom, building_props, building_geojson_feature, source, warning).
    """
    tb = (
        get_open_buildings_vector(aoi, confidence_threshold=confidence)
        .filterBounds(centroid)
        .first()
        .getInfo()
    )
    source = "vector_centroid_point"
    warning: Optional[str] = None

    # A point on a polygon edge (or a low-confidence building) can yield empty;
    # retry with a small buffer, then fall back to the AOI roof mask.
    if tb is None:
        try:
            tb = (
                get_open_buildings_vector(aoi, confidence_threshold=confidence)
                .filterBounds(centroid.buffer(30))
                .first()
                .getInfo()
            )
            if tb is not None:
                source = "vector_centroid_buffer30m"
        except Exception:
            tb = None

    if tb is None:
        source = "aoi_fallback"
        warning = "No Open Buildings polygon found at the selected point; using the AOI roof mask for calculations."
        building_geom = aoi
        building_props = {"confidence": confidence, "area_in_meters": None}
        building_geojson_feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {},
        }
    else:
        building_geom = ee.Feature(tb).geometry()
        building_props = tb.get("properties", {})
        building_geojson_feature = tb

    return building_geom, building_props, building_geojson_feature, source, warning


def _ee_tile_template(image: ee.Image, vis: Dict[str, Any]) -> str:
    """
    Return Map ID tile template URL for an EE image.
    This yields a URL like: https://earthengine.googleapis.com/v1alpha/projects/.../maps/{mapid}/tiles/{z}/{x}/{y}
    """
    m = image.getMapId(vis)
    return m["tile_fetcher"].url_format


@app.post("/api/tiles")
def tiles(req: TilesRequest) -> Dict[str, Any]:
    """
    Generate Earth Engine tile URL templates (XYZ) for raster overlays within the AOI.
    Layers:
      - roof_mask: rooftop candidate mask (0/1)
      - shadow_frequency: shadow frequency (0..1)
      - sky_view_factor: fraction of diffuse sky visible from the rooftop (0..1)
      - net_irradiance: net irradiance (kWh/m^2 over window)
      - combined_derate: uhi_derate * soiling_retention (scalar image)
      - temperature_delta: UHI delta temperature (MODIS LST daytime anomaly; degC)
    """
    try:
        try:
            win = resolve_temporal_window(
                req.baseline_mode,
                req.year,
                req.quarter,
                req.month,
                req.start_date,
                req.end_date_exclusive,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        _ensure_ee(req.project_id)
        coords, aoi = _aoi_from_req(req)
        centroid = aoi.centroid(1)
        lon_deg, lat_deg = _centroid_lon_lat(centroid)
        s, e = win["start_date"], win["end_date_exclusive"]

        _, building_height, roof_mask = _build_roof_layers(
            aoi, req.roof_year, req.presence_threshold, req.min_height_m
        )

        solar_positions = _solar_positions_for_window(lat_deg, lon_deg, win)
        shadow_freq = ShadowPenalty.frequency(building_height, solar_positions=solar_positions)
        svf_img = SkyViewFactor.image(building_height)

        # Scalars needed for net irradiance (same as /api/yield)
        ghi_info = sample_era5_period_ghi_kwh_m2_at_point(centroid, s, e, scale_m=ERA5_SCALE_M)
        regional_ghi_kwh_m2_period = float(ghi_info["value"])
        beam_info = sample_era5_beam_fraction_at_point(centroid, s, e)
        beam_fraction = float(beam_info["beam_fraction"])
        uhi_info = UHIPenalty.stats(aoi, s)
        soiling_info = SoilingPenalty.stats(aoi, s)
        combined_derate = float(uhi_info["uhi_derate_factor"]) * float(soiling_info["soiling_retention_factor"])

        net_irr = net_irradiance_image(
            regional_ghi_kwh_m2_period,
            shadow_freq,
            beam_fraction=beam_fraction,
            uhi_derate=float(uhi_info["uhi_derate_factor"]),
            soiling_retention=float(soiling_info["soiling_retention_factor"]),
            sky_view_factor=svf_img,
        )

        if req.layer == "roof_mask":
            img = roof_mask.selfMask()
            vis = {"min": 0, "max": 1, "palette": ["00e5ff"]}
        elif req.layer == "shadow_frequency":
            img = shadow_freq.clamp(0, 1)
            vis = {"min": 0, "max": 1, "palette": ["0b1020", "f97316"]}
        elif req.layer == "sky_view_factor":
            img = svf_img.clip(aoi).clamp(0, 1)
            # Low SVF (sky blocked) -> warm; high SVF (open sky) -> cool/green.
            vis = {"min": 0.5, "max": 1.0, "palette": ["ef4444", "f59e0b", "22c55e"]}
        elif req.layer == "temperature_delta":
            # UHI = urban mean LST - ~30km background focal mean (see UHIPenalty.stats).
            uhi_year = int(s[:4])
            lst = (
                ee.ImageCollection(UHIPenalty.MODIS_COLLECTION)
                .filterBounds(aoi)
                .filterDate(f"{uhi_year}-01-01", f"{uhi_year + 1}-01-01")
                .select(UHIPenalty.LST_DAY_BAND)
                .median()
                .multiply(UHIPenalty.LST_SCALE)
                .subtract(UHIPenalty.K_TO_C_OFFSET)
                .rename("LST_celsius")
            )
            background = lst.focal_mean(
                radius=UHIPenalty.BACKGROUND_KERNEL_PX,
                kernelType="circle",
                units="pixels",
            )
            img = lst.subtract(background).rename("delta_t_uhi_celsius").clip(aoi).clamp(-3.0, 8.0)
            # Typical Indian UHI anomalies: ~2-6 degC (but allow a bit wider).
            # Avoid the bright yellow/orange used by irradiance visualizations; keep it cleaner.
            vis = {"min": -3.0, "max": 8.0, "palette": ["2563eb", "22c55e", "a855f7", "ef4444"]}
        elif req.layer == "combined_derate":
            img = ee.Image.constant(combined_derate).rename("combined_derate").clip(aoi)
            vis = {"min": 0.9, "max": 1.0, "palette": ["ef4444", "f59e0b", "22c55e"]}
        else:
            img = net_irr.clip(aoi)
            # Dynamic max for visibility: assume max ~ 1.1x baseline as rough upper bound.
            vis = {"min": 0, "max": max(50.0, regional_ghi_kwh_m2_period * 1.05), "palette": ["0b1020", "2563eb", "22c55e", "f59e0b"]}

        url = _ee_tile_template(img, vis)
        # Approx bounds from request polygon (lon,lat)
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        bounds = [[min(lons), min(lats)], [max(lons), max(lats)]]

        return {
            "status": "ok",
            "layer": req.layer,
            "baseline_time_mode": win["mode"],
            "start_date": s,
            "end_date_exclusive": e,
            "urlTemplate": url,
            "tileSize": 256,
            "minZoom": 0,
            "maxZoom": 19,
            "bounds": bounds,
            "attribution": "Google Earth Engine",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/buildings")
def buildings(req: BuildingsRequest) -> Dict[str, Any]:
    """
    Return Open Buildings v3 polygons within the AOI as GeoJSON.
    Intended for map rendering / selection (open-data-only).
    """
    try:
        _ensure_ee(req.project_id)
        coords, aoi = _aoi_from_req(req)
        fc = get_open_buildings_vector(aoi, confidence_threshold=req.building_confidence).limit(req.limit)
        gj = fc.getInfo()
        # Keep payload reasonable: strip any huge property blobs, keep key fields only.
        features = []
        for f in (gj or {}).get("features", []) or []:
            props = (f.get("properties") or {})
            features.append({
                "type": "Feature",
                "id": f.get("id"),
                "geometry": f.get("geometry"),
                "properties": {
                    "confidence": props.get("confidence"),
                    "area_in_meters": props.get("area_in_meters"),
                    "full_id": props.get("full_id") or props.get("id"),
                },
            })
        return {
            "status": "ok",
            "aoi_coordinates": coords,
            "count": len(features),
            "limit": req.limit,
            "building_confidence": req.building_confidence,
            "geojson": {"type": "FeatureCollection", "features": features},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/yield")
def compute_yield(req: YieldRequest) -> Dict[str, Any]:
    """
    Single-building PV energy for the same temporal window as /api/baseline.

    ERA5 GHI is summed over [start_date, end_date_exclusive) at the AOI centroid.
    Shadow retention uses sun positions aligned with that window (year / quarter / day)
    at the centroid latitude and longitude.
    """
    try:
        try:
            win = resolve_temporal_window(
                req.baseline_mode,
                req.year,
                req.quarter,
                req.month,
                req.start_date,
                req.end_date_exclusive,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        _ensure_ee(req.project_id)
        coords, aoi = _aoi_from_req(req)
        centroid = aoi.centroid(1)
        lon_deg, lat_deg = _centroid_lon_lat(centroid)
        s, e = win["start_date"], win["end_date_exclusive"]

        ghi_info = sample_era5_period_ghi_kwh_m2_at_point(centroid, s, e, scale_m=ERA5_SCALE_M)
        regional_ghi_kwh_m2_period = float(ghi_info["value"])
        if ghi_info["source"] in ("no_sample", "null_band"):
            raise HTTPException(status_code=500, detail="Could not sample ERA5 GHI for the selected period at centroid.")

        solar_positions = _solar_positions_for_window(lat_deg, lon_deg, win)

        _, building_height, roof_mask = _build_roof_layers(
            aoi, req.roof_year, req.presence_threshold, req.min_height_m
        )

        # Shadow frequency (per-pixel, insolation-weighted, data-driven from building heights)
        shadow_freq = ShadowPenalty.frequency(building_height, solar_positions=solar_positions)

        # Beam fraction: direct / GHI from ERA5 HOURLY -- used to correct shadow losses.
        # Only the beam component is blocked by shadows; diffuse is governed by SVF below.
        beam_info = sample_era5_beam_fraction_at_point(centroid, s, e)
        beam_fraction = float(beam_info["beam_fraction"])

        # Sky View Factor: per-pixel fraction of the diffuse sky still visible from the
        # rooftop after neighbouring buildings occlude part of the hemisphere. Diffuse
        # counterpart of the shadow (beam) penalty; both come from the same height raster.
        # (Mean SVF is reduced once, over the building geometry, further below.)
        svf_img = SkyViewFactor.image(building_height)

        uhi_info = UHIPenalty.stats(aoi, s)
        soiling_info = SoilingPenalty.stats(aoi, s)

        net_irr = net_irradiance_image(
            regional_ghi_kwh_m2_period,
            shadow_freq,
            beam_fraction=beam_fraction,
            uhi_derate=uhi_info["uhi_derate_factor"],
            soiling_retention=soiling_info["soiling_retention_factor"],
            sky_view_factor=svf_img,
        )

        period_label = {"yearly": "calendar_year", "quarterly": "calendar_quarter", "monthly": "calendar_month", "daily": "single_day"}[win["mode"]]
        (
            building_geom,
            building_props,
            building_geojson_feature,
            building_selection_source,
            selection_warning,
        ) = _select_target_building(aoi, coords, centroid, req.building_confidence)

        # ------------------------------------------------------------------
        # Stage irradiance images (per-pixel kWh/m^2 for the period), all on the
        # SAME building geometry so stage losses are directly comparable:
        #   baseline : GHI, no penalties
        #   shadow   : beam blocking only (SVF = 1, diffuse fully received)
        #   svf      : shadow + diffuse occlusion (Sky View Factor)
        #   uhi      : shadow + svf + uhi
        #   net      : shadow + svf + uhi + soiling  (== net_irr, already built)
        # ------------------------------------------------------------------
        baseline_irr = ee.Image.constant(regional_ghi_kwh_m2_period).rename("baseline")
        shadow_only_irr = net_irradiance_image(
            regional_ghi_kwh_m2_period, shadow_freq, beam_fraction=beam_fraction,
            uhi_derate=1.0, soiling_retention=1.0, sky_view_factor=None,
        )
        svf_only_irr = net_irradiance_image(
            regional_ghi_kwh_m2_period, shadow_freq, beam_fraction=beam_fraction,
            uhi_derate=1.0, soiling_retention=1.0, sky_view_factor=svf_img,
        )
        uhi_only_irr = net_irradiance_image(
            regional_ghi_kwh_m2_period, shadow_freq, beam_fraction=beam_fraction,
            uhi_derate=float(uhi_info["uhi_derate_factor"]), soiling_retention=1.0,
            sky_view_factor=svf_img,
        )

        # Batched reduction #1: all five stage energies (kWh = irr * roof_mask * area)
        # plus roof area, summed over the building in ONE getInfo call.
        area_img = roof_mask.toFloat().multiply(ee.Image.pixelArea())
        sum_stack = (
            baseline_irr.multiply(area_img).rename("e_baseline")
            .addBands(shadow_only_irr.multiply(area_img).rename("e_shadow"))
            .addBands(svf_only_irr.multiply(area_img).rename("e_svf"))
            .addBands(uhi_only_irr.multiply(area_img).rename("e_uhi"))
            .addBands(net_irr.multiply(area_img).rename("e_soiling"))
            .addBands(area_img.rename("roof_area"))
        )
        sum_raw = sum_stack.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=building_geom, scale=4.0, maxPixels=1e7,
        ).getInfo() or {}

        # Batched reduction #2: per-pixel means of shadow frequency and SVF in ONE call.
        mean_stack = shadow_freq.rename("shadow_frequency").addBands(svf_img.rename("sky_view_factor"))
        mean_raw = mean_stack.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=building_geom, scale=4.0, maxPixels=1e7,
        ).getInfo() or {}

        baseline_roof_kwh = float(sum_raw.get("e_baseline") or 0.0)
        after_shadow_roof_kwh = float(sum_raw.get("e_shadow") or 0.0)
        after_svf_roof_kwh = float(sum_raw.get("e_svf") or 0.0)
        after_uhi_roof_kwh = float(sum_raw.get("e_uhi") or 0.0)
        after_soiling_roof_kwh = float(sum_raw.get("e_soiling") or 0.0)
        roof_area_m2 = float(sum_raw.get("roof_area") or 0.0)
        mean_shadow_frequency = mean_raw.get("shadow_frequency")
        mean_sky_view_factor = mean_raw.get("sky_view_factor")
        mean_shadow_fraction = mean_shadow_frequency  # shadow_freq IS the fraction in shadow

        # packing_factor: usable-roof coverage fraction. Applied uniformly to every
        # stage so penalty percentages are unchanged; only absolute kWh scale down to
        # reflect that panels cover ~60-75% of a roof, not 100%.
        yield_scale = req.panel_efficiency * req.performance_ratio * req.packing_factor
        baseline_yield_kwh = baseline_roof_kwh * yield_scale
        after_shadow_yield_kwh = after_shadow_roof_kwh * yield_scale
        after_svf_yield_kwh = after_svf_roof_kwh * yield_scale
        after_uhi_yield_kwh = after_uhi_roof_kwh * yield_scale
        after_soiling_yield_kwh = after_soiling_roof_kwh * yield_scale

        total_energy_kwh = after_soiling_yield_kwh

        penalty_loss_kwh = max(0.0, baseline_yield_kwh - total_energy_kwh)
        penalty_loss_pct = (penalty_loss_kwh / baseline_yield_kwh * 100.0) if baseline_yield_kwh > 0 else 0.0

        shadow_loss_kwh = max(0.0, baseline_yield_kwh - after_shadow_yield_kwh)
        svf_loss_kwh = max(0.0, after_shadow_yield_kwh - after_svf_yield_kwh)
        uhi_loss_kwh = max(0.0, after_svf_yield_kwh - after_uhi_yield_kwh)
        soiling_loss_kwh = max(0.0, after_uhi_yield_kwh - after_soiling_yield_kwh)
        loss_total_for_split = shadow_loss_kwh + svf_loss_kwh + uhi_loss_kwh + soiling_loss_kwh
        if loss_total_for_split <= 0:
            shadow_contrib_pct = 0.0
            svf_contrib_pct = 0.0
            uhi_contrib_pct = 0.0
            soiling_contrib_pct = 0.0
        else:
            shadow_contrib_pct = shadow_loss_kwh / loss_total_for_split * 100.0
            svf_contrib_pct = svf_loss_kwh / loss_total_for_split * 100.0
            uhi_contrib_pct = uhi_loss_kwh / loss_total_for_split * 100.0
            soiling_contrib_pct = soiling_loss_kwh / loss_total_for_split * 100.0

        # ------------------------------------------------------------------
        # Rooftop shade matrix (6 evenly distributed 4-hour UTC buckets).
        # Each non-empty bucket's shadow-frequency image is stacked as a band and
        # reduced together in ONE getInfo call (instead of one call per bucket).
        # ------------------------------------------------------------------
        bucket_specs = [
            ("00-04", 0, 4),
            ("04-08", 4, 8),
            ("08-12", 8, 12),
            ("12-16", 12, 16),
            ("16-20", 16, 20),
            ("20-24", 20, 24),
        ]
        # solar_positions is a list of (alt_deg, az_deg, weight, hour_utc)
        bucket_band = {}   # label -> band name (only for non-empty buckets)
        shade_stack = None
        for label, h0, h1 in bucket_specs:
            bucket_positions = [
                (p[0], p[1], p[2], p[3])
                for p in (solar_positions or [])
                if len(p) >= 4 and p[3] >= h0 and p[3] < h1
            ]
            wsum = sum(float(p[2]) for p in bucket_positions) if bucket_positions else 0.0
            if not bucket_positions or wsum <= 0:
                continue
            norm_positions = [(p[0], p[1], p[2] / wsum, p[3]) for p in bucket_positions]
            band = "shade_" + label.replace("-", "_")
            freq_band = ShadowPenalty.frequency(
                building_height, solar_positions=norm_positions
            ).rename(band)
            bucket_band[label] = band
            shade_stack = freq_band if shade_stack is None else shade_stack.addBands(freq_band)

        shade_raw = (
            shade_stack.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=building_geom,
                scale=4.0,
                maxPixels=1e7,
            ).getInfo() or {}
        ) if shade_stack is not None else {}

        shade_intervals = []
        for label, h0, h1 in bucket_specs:
            band = bucket_band.get(label)
            raw_bucket = shade_raw.get(band) if band else None
            shade_fraction = float(raw_bucket) if raw_bucket is not None else 0.0
            shade_area_m2 = float(roof_area_m2) * float(shade_fraction)
            shade_intervals.append(
                {
                    "label": label,
                    "shade_fraction": round(shade_fraction, 5),
                    "shade_percent": round(shade_fraction * 100.0, 2),
                    "shade_area_m2": round(shade_area_m2, 2),
                }
            )

        mean_shadow_retention = (
            round(1.0 - mean_shadow_frequency * beam_fraction, 4)
            if mean_shadow_frequency is not None else None
        )

        # mean_sky_view_factor already reduced above (batched reduction #2).
        diffuse_fraction = 1.0 - beam_fraction
        # Full retention scalar: diffuse * SVF + beam * (1 - shadow). Falls back to the
        # beam-only shadow retention if SVF could not be sampled.
        if mean_shadow_frequency is not None and mean_sky_view_factor is not None:
            mean_net_retention = round(
                diffuse_fraction * float(mean_sky_view_factor)
                + beam_fraction * (1.0 - mean_shadow_frequency),
                4,
            )
        else:
            mean_net_retention = mean_shadow_retention

        combined_derate = uhi_info["uhi_derate_factor"] * soiling_info["soiling_retention_factor"]
        net_irr_mean = (
            regional_ghi_kwh_m2_period * combined_derate * mean_net_retention
            if mean_net_retention is not None else None
        )

        shadow_penalty_percent = (
            round((1.0 - mean_shadow_retention) * 100.0, 2) if mean_shadow_retention is not None else None
        )
        svf_penalty_percent = (
            round(diffuse_fraction * (1.0 - float(mean_sky_view_factor)) * 100.0, 2)
            if mean_sky_view_factor is not None else None
        )
        uhi_penalty_percent = round((1.0 - uhi_info["uhi_derate_factor"]) * 100.0, 2)
        soiling_penalty_percent = round((1.0 - soiling_info["soiling_retention_factor"]) * 100.0, 2)
        combined_penalty_percent = round((1.0 - combined_derate) * 100.0, 2)

        out = {
            "status": "ok",
            "baseline_time_mode": win["mode"],
            "start_date": s,
            "end_date_exclusive": e,
            "accounting_period": period_label,
            "building_selection_source": building_selection_source,
            "selection_warning": selection_warning,
            "regional_ghi_kwh_m2_period": regional_ghi_kwh_m2_period,
            "ghi_sample_source": ghi_info["source"],
            "irradiance_source": "ERA5",
            "panel_efficiency": req.panel_efficiency,
            "performance_ratio": req.performance_ratio,
            "packing_factor": req.packing_factor,
            # Some windows don't define quarter/month keys; keep response stable.
            "calendar_year": win.get("calendar_year"),
            "quarter": win.get("quarter"),
            "month": win.get("month"),
            "building_confidence": building_props.get("confidence"),
            "building_area_in_meters": building_props.get("area_in_meters"),
            "roof_area_m2": roof_area_m2,
            "mean_shadow_fraction": mean_shadow_fraction,
            "mean_shadow_retention": mean_shadow_retention,
            # Authoritative stage yields (PV output, kWh)
            "baseline_yield_kwh": round(float(baseline_yield_kwh), 6),
            "after_shadow_yield_kwh": round(float(after_shadow_yield_kwh), 6),
            "after_svf_yield_kwh": round(float(after_svf_yield_kwh), 6),
            "after_uhi_yield_kwh": round(float(after_uhi_yield_kwh), 6),
            "after_soiling_yield_kwh": round(float(after_soiling_yield_kwh), 6),
            # Loss + contribution (of total loss) in kWh / %
            "penalty_loss_kwh": round(float(penalty_loss_kwh), 6),
            "penalty_loss_pct": round(float(penalty_loss_pct), 4),
            "penalty_contribution": {
                "shadow_loss_kwh": round(float(shadow_loss_kwh), 6),
                "svf_loss_kwh": round(float(svf_loss_kwh), 6),
                "uhi_loss_kwh": round(float(uhi_loss_kwh), 6),
                "soiling_loss_kwh": round(float(soiling_loss_kwh), 6),
                "shadow_contribution_pct": round(float(shadow_contrib_pct), 3),
                "svf_contribution_pct": round(float(svf_contrib_pct), 3),
                "uhi_contribution_pct": round(float(uhi_contrib_pct), 3),
                "soiling_contribution_pct": round(float(soiling_contrib_pct), 3),
            },
            "shade_intervals": shade_intervals,
            "beam_fraction": beam_fraction,
            "diffuse_fraction": beam_info["diffuse_fraction"],
            "beam_fraction_source": beam_info["source"],
            "mean_sky_view_factor": (round(float(mean_sky_view_factor), 5)
                                     if mean_sky_view_factor is not None else None),
            "svf_penalty_percent": svf_penalty_percent,
            "sky_view_factor_meta": {
                "n_azimuth": SkyViewFactor.N_AZIMUTH,
                "sample_radii_px": list(SkyViewFactor.DIST_PX),
            },
            "uhi_derate_factor": uhi_info["uhi_derate_factor"],
            "delta_t_uhi_celsius": uhi_info["delta_t_uhi_celsius"],
            "soiling_retention_factor": soiling_info["soiling_retention_factor"],
            "mean_aod_550nm": soiling_info["mean_aod_550nm"],
            "combined_derate_factor": round(combined_derate, 5),
            "net_irradiance_kwh_m2_period": net_irr_mean,
            "period_yield_kwh": total_energy_kwh,
            "shadow_penalty_percent": shadow_penalty_percent,
            "uhi_penalty_percent": uhi_penalty_percent,
            "soiling_penalty_percent": soiling_penalty_percent,
            "combined_penalty_percent": combined_penalty_percent,
            "uhi_penalty": uhi_info,
            "soiling_penalty": soiling_info,
            "geojson": {
                "type": "FeatureCollection",
                "features": [{
                    **building_geojson_feature,
                    "properties": {
                        **building_props,
                        "roof_area_m2": roof_area_m2,
                        "mean_shadow_fraction": mean_shadow_fraction,
                        "uhi_derate_factor": uhi_info["uhi_derate_factor"],
                        "soiling_retention_factor": soiling_info["soiling_retention_factor"],
                        "net_irradiance_kwh_m2_period": net_irr_mean,
                        "period_yield_kwh": total_energy_kwh,
                        "shade_intervals": shade_intervals,
                    }
                }]
            },
        }
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _cap_positions(pos: List[Tuple]) -> List[Tuple]:
    """Match /api/yield's position cap so series shadow sampling is identical."""
    return pos[::2] if len(pos) > 42 else pos


def _series_layout(
    mode: str,
    win: Dict[str, Any],
    lat_deg: float,
    lon_deg: float,
) -> Tuple[List[str], List[Tuple[str, str, List[Tuple]]], List[int]]:
    """
    Build the sub-periods for the generation curve.

    Returns (labels, items, bin_of) where:
      labels : output x-axis labels
      items  : list of (start_date, end_date_exclusive, solar_positions) to evaluate
      bin_of : item index -> output bucket index (identity except weekly binning)
    """
    year = win.get("calendar_year")

    if mode == "yearly":
        items = []
        for m in range(1, 13):
            s = f"{year}-{m:02d}-01"
            e = f"{year + 1}-01-01" if m == 12 else f"{year}-{m + 1:02d}-01"
            items.append((s, e, _cap_positions(solar_positions_monthly(lat_deg, lon_deg, year, m))))
        return list(_MONTH_ABBR), items, list(range(12))

    if mode == "quarterly":
        q = int(win["quarter"])
        months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}[q]
        items, labels = [], []
        for m in months:
            s = f"{year}-{m:02d}-01"
            e = f"{year + 1}-01-01" if m == 12 else f"{year}-{m + 1:02d}-01"
            items.append((s, e, _cap_positions(solar_positions_monthly(lat_deg, lon_deg, year, m))))
            labels.append(_MONTH_ABBR[m - 1])
        return labels, items, list(range(len(items)))

    if mode == "monthly":
        m = int(win["month"])
        first = date(year, m, 1)
        nxt = date(year + 1, 1, 1) if m == 12 else date(year, m + 1, 1)
        ndays = (nxt - first).days
        items, bin_of = [], []
        for d in range(1, ndays + 1):
            day = date(year, m, d)
            s = day.isoformat()
            e = (day + timedelta(days=1)).isoformat()
            items.append((s, e, _cap_positions(solar_positions_single_day(lat_deg, lon_deg, day))))
            bin_of.append(min(4, (d - 1) // 7))
        return ["W1", "W2", "W3", "W4", "W5"], items, bin_of

    # daily: a single point
    s, e = win["start_date"], win["end_date_exclusive"]
    d0 = date.fromisoformat(s)
    return [s], [(s, e, _cap_positions(solar_positions_single_day(lat_deg, lon_deg, d0)))], [0]


@app.post("/api/series")
def compute_series(req: YieldRequest) -> Dict[str, Any]:
    """
    Generation curve for the selected window in a single HTTP call (vs one full
    /api/yield per point). GHI + beam are sampled batched; the per-period shadow
    retention is reduced one period at a time (a light, bounded EE request each)
    so no single reduceRegion stacks every period's focal-max shadow at once.

      yearly    -> 12 monthly points
      quarterly -> the quarter's 3 monthly points
      monthly   -> daily yields binned into weeks W1..W5
      daily     -> single point

    Uses the exact factorization of net_irradiance_image:
      net_period = GHI * uhi * soiling * eff*PR*packing
                   * [ (1-beam) * SUM(SVF*area) + beam * SUM((1-shadow_freq)*area) ]
    so each point equals what /api/yield would return for that sub-window.
    """
    try:
        try:
            win = resolve_temporal_window(
                req.baseline_mode, req.year, req.quarter, req.month,
                req.start_date, req.end_date_exclusive,
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))

        _ensure_ee(req.project_id)
        coords, aoi = _aoi_from_req(req)
        centroid = aoi.centroid(1)
        lon_deg, lat_deg = _centroid_lon_lat(centroid)
        mode = win["mode"]

        labels, items, bin_of = _series_layout(mode, win, lat_deg, lon_deg)
        if not items:
            return {"status": "ok", "baseline_time_mode": mode, "labels": labels,
                    "values": [0.0] * len(labels)}

        _, building_height, roof_mask = _build_roof_layers(
            aoi, req.roof_year, req.presence_threshold, req.min_height_m
        )
        building_geom, _, _, _, _ = _select_target_building(
            aoi, coords, centroid, req.building_confidence
        )
        svf_img = SkyViewFactor.image(building_height)
        area_img = roof_mask.toFloat().multiply(ee.Image.pixelArea())

        # Per-sub-period scalars: GHI (1 getInfo) and beam fraction (1 getInfo).
        windows = [(s, e) for (s, e, _pos) in items]
        ghi_list = sample_era5_period_ghi_multi(centroid, windows, scale_m=ERA5_SCALE_M)
        beam_list = sample_era5_beam_multi(centroid, windows, scale_m=_ERA5_HOURLY_SCALE_M)

        # SUM(SVF*area) is static across sub-periods -> one reduction.
        svf_area = float(
            (svf_img.multiply(area_img).rename("svf_area")
             .reduceRegion(ee.Reducer.sum(), building_geom, 4.0, maxPixels=1e7)
             .getInfo() or {}).get("svf_area") or 0.0
        )

        # SUM((1-shadow_freq)*area) varies per sub-period (solar geometry differs).
        # Reduce each period on its own: one period carries ~one window of solar
        # positions -- the same focal-max load a single /api/yield handles. Stacking
        # all periods into one reduceRegion would exceed EE's per-request memory.
        retained_beam_area = []
        for (_s, _e, pos) in items:
            shadow_freq = ShadowPenalty.frequency(building_height, solar_positions=pos)
            ba = ee.Image(1.0).subtract(shadow_freq).multiply(area_img).rename("ba")
            raw = ba.reduceRegion(
                reducer=ee.Reducer.sum(), geometry=building_geom, scale=4.0, maxPixels=1e7,
            ).getInfo() or {}
            retained_beam_area.append(float(raw.get("ba") or 0.0))

        # UHI + soiling are annual (static across the sub-periods of one year).
        uhi = UHIPenalty.stats(aoi, items[0][0])
        soiling = SoilingPenalty.stats(aoi, items[0][0])
        derate = float(uhi["uhi_derate_factor"]) * float(soiling["soiling_retention_factor"])
        scale = req.panel_efficiency * req.performance_ratio * req.packing_factor

        values = [0.0] * len(labels)
        for i in range(len(items)):
            beam_i = beam_list[i]
            diffuse_i = 1.0 - beam_i
            net_i = ghi_list[i] * derate * scale * (diffuse_i * svf_area + beam_i * retained_beam_area[i])
            values[bin_of[i]] += net_i

        return {
            "status": "ok",
            "baseline_time_mode": mode,
            "labels": labels,
            "values": [round(v, 3) for v in values],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

