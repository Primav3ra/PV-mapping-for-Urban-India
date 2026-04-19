"""
SolarMappingUtils: GEE init + thin wrappers over datasets / rooftops / irradiance_baseline.
All heavy computation stays in the domain modules; this class exists for the FastAPI layer.
"""
from __future__ import annotations

import ee
from typing import Any, Dict, List, Optional

from scripts.datasets import get_dem as _get_dem, get_available_datasets as _get_datasets_info
from scripts.rooftops import get_rooftop_area_m2_info as _get_rooftop_area_m2_info
from scripts.irradiance_baseline import (
    get_era5_baseline_info as _get_era5_baseline_info,
    get_era5_range_info as _get_era5_range_info,
    get_roof_masked_era5_baseline_info as _get_roof_masked_era5_baseline_info,
    get_roof_masked_era5_baseline_for_date_range as _get_roof_masked_era5_baseline_for_date_range,
    latest_complete_5y_range as _latest_complete_5y_range,
)
from scripts.rooftops import build_rooftop_candidate_mask, apply_terrain_exclusion
from scripts.datasets import get_open_buildings_temporal


class SolarMappingUtils:
    def __init__(self, project_id: str):
        self.project_id = project_id
        ee.Initialize(project=project_id)

    # ------------------------------------------------------------------
    # AOI helpers
    # ------------------------------------------------------------------

    def create_aoi_from_coordinates(self, coordinates: List[List[float]]) -> ee.Geometry:
        return ee.Geometry.Polygon(coordinates)

    def load_aoi_from_geojson(self, geojson_path: str) -> ee.Geometry:
        import json
        with open(geojson_path) as f:
            data = json.load(f)
        return ee.Geometry.Polygon(data["features"][0]["geometry"]["coordinates"][0])

    # ------------------------------------------------------------------
    # Dataset catalogue
    # ------------------------------------------------------------------

    def get_available_datasets(self) -> Dict[str, Any]:
        return _get_datasets_info()

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------

    def get_elevation_data(self, aoi: ee.Geometry, dem_type: str = "srtm") -> ee.Image:
        return _get_dem(aoi, dem_type=dem_type)

    def create_exclusion_mask(self, dem: ee.Image, aoi: ee.Geometry) -> ee.Image:
        """Binary mask: 1 = suitable (slope < 30 deg), 0 = excluded."""
        return ee.Terrain.products(dem).select("slope").lt(30)

    # ------------------------------------------------------------------
    # Rooftop candidates
    # ------------------------------------------------------------------

    def get_rooftop_candidate_stats(
        self,
        aoi: ee.Geometry,
        exclusion_mask: Optional[ee.Image] = None,
        year: Optional[int] = 2022,
        presence_threshold: float = 0.5,
        min_height_m: float = 0.0,
        reduce_scale_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        return _get_rooftop_area_m2_info(
            aoi,
            year=year,
            presence_threshold=presence_threshold,
            min_height_m=min_height_m,
            exclusion_mask=exclusion_mask,
            scale_m=reduce_scale_m,
        )

    # ------------------------------------------------------------------
    # Irradiance baselines
    # ------------------------------------------------------------------

    def get_era5_baseline_stats(
        self,
        aoi: ee.Geometry,
        start_year: int = 2020,
        end_year: int = 2024,
        scale_m: float = 11_132.0,
    ) -> Dict[str, Any]:
        """ERA5 annual GHI stats."""
        return _get_era5_baseline_info(aoi, start_year=start_year, end_year=end_year, scale_m=scale_m)

    def get_era5_range_stats(
        self,
        aoi: ee.Geometry,
        start_date: str,
        end_date_exclusive: str,
        scale_m: float = 11_132.0,
    ) -> Dict[str, Any]:
        """ERA5 period-range GHI stats."""
        return _get_era5_range_info(aoi, start_date=start_date, end_date_exclusive=end_date_exclusive, scale_m=scale_m)

    # ------------------------------------------------------------------
    # Roof-masked baselines (shared building mask builder)
    # ------------------------------------------------------------------

    def _build_roof_mask(
        self,
        aoi: ee.Geometry,
        exclusion_mask: Optional[ee.Image],
        roof_year: Optional[int],
        presence_threshold: float,
        min_height_m: float,
    ) -> ee.Image:
        buildings = get_open_buildings_temporal(aoi, year=roof_year)
        mask = build_rooftop_candidate_mask(buildings, presence_threshold=presence_threshold, min_height_m=min_height_m)
        if exclusion_mask is not None:
            mask = apply_terrain_exclusion(mask, exclusion_mask=exclusion_mask, buildings=buildings, scale_m=4.0)
        return mask

    def get_roof_masked_era5_baseline_stats(
        self,
        aoi: ee.Geometry,
        exclusion_mask: Optional[ee.Image] = None,
        roof_year: Optional[int] = 2022,
        presence_threshold: float = 0.5,
        min_height_m: float = 0.0,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        scale_m: float = 11_132.0,
    ) -> Dict[str, Any]:
        """Pre-penalty rooftop baseline: roof area x ERA5 annual GHI (yearly mode)."""
        if start_year is None or end_year is None:
            start_year, end_year = _latest_complete_5y_range()
        return _get_roof_masked_era5_baseline_info(
            aoi=aoi,
            roof_mask=self._build_roof_mask(aoi, exclusion_mask, roof_year, presence_threshold, min_height_m),
            start_year=start_year,
            end_year=end_year,
            scale_m=scale_m,
        )

    def get_roof_masked_era5_baseline_for_date_range_stats(
        self,
        aoi: ee.Geometry,
        exclusion_mask: Optional[ee.Image] = None,
        roof_year: Optional[int] = 2022,
        presence_threshold: float = 0.5,
        min_height_m: float = 0.0,
        start_date: str = "",
        end_date_exclusive: str = "",
        scale_m: float = 11_132.0,
    ) -> Dict[str, Any]:
        """Pre-penalty rooftop baseline: roof area x ERA5 period GHI (quarterly / daily mode)."""
        return _get_roof_masked_era5_baseline_for_date_range(
            aoi=aoi,
            roof_mask=self._build_roof_mask(aoi, exclusion_mask, roof_year, presence_threshold, min_height_m),
            start_date=start_date,
            end_date_exclusive=end_date_exclusive,
            scale_m=scale_m,
        )
