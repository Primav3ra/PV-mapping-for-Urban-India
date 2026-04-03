import ee
from typing import List, Dict, Any, Optional

# Optional: use centralized dataset loaders (Open Buildings, FABDEM, Sentinel-2, MODIS LST)
_HAS_DATASETS = False
try:
    from datasets import get_dem as _get_dem_datasets
    from datasets import get_available_datasets as _get_datasets_info
    _HAS_DATASETS = True
except ImportError:
    try:
        from scripts.datasets import get_dem as _get_dem_datasets
        from scripts.datasets import get_available_datasets as _get_datasets_info
        _HAS_DATASETS = True
    except ImportError:
        pass

try:
    from rooftops import get_rooftop_area_m2_info as _get_rooftop_area_m2_info
    _HAS_ROOFTOPS = True
except ImportError:
    try:
        from scripts.rooftops import get_rooftop_area_m2_info as _get_rooftop_area_m2_info
        _HAS_ROOFTOPS = True
    except ImportError:
        _HAS_ROOFTOPS = False

try:
    from irradiance_baseline import get_era5_baseline_info as _get_era5_baseline_info
    from irradiance_baseline import get_era5_range_info as _get_era5_range_info
    from irradiance_baseline import latest_complete_5y_range as _latest_complete_5y_range
    from irradiance_baseline import ERA5_SCALE_M as _ERA5_SCALE_M
    _HAS_IRRADIANCE = True
except ImportError:
    try:
        from scripts.irradiance_baseline import get_era5_baseline_info as _get_era5_baseline_info
        from scripts.irradiance_baseline import get_era5_range_info as _get_era5_range_info
        from scripts.irradiance_baseline import latest_complete_5y_range as _latest_complete_5y_range
        from scripts.irradiance_baseline import ERA5_SCALE_M as _ERA5_SCALE_M
        _HAS_IRRADIANCE = True
    except ImportError:
        _ERA5_SCALE_M = 11_132.0
        _HAS_IRRADIANCE = False

try:
    from irradiance_baseline import get_roof_masked_era5_baseline_info as _get_roof_masked_era5_baseline_info
    _HAS_ROOF_MASKED_IRRADIANCE = True
except ImportError:
    try:
        from scripts.irradiance_baseline import get_roof_masked_era5_baseline_info as _get_roof_masked_era5_baseline_info
        _HAS_ROOF_MASKED_IRRADIANCE = True
    except ImportError:
        _HAS_ROOF_MASKED_IRRADIANCE = False


class SolarMappingUtils:
    def __init__(self, project_id: str):
        self.project_id = project_id
        ee.Initialize(project=project_id)
        print("Earth Engine initialized successfully!")
    
    def create_aoi_from_coordinates(self, coordinates: List[List[float]]) -> ee.Geometry:
        """Create Area of Interest from coordinate list"""
        return ee.Geometry.Polygon(coordinates)
    
    def load_aoi_from_geojson(self, geojson_path: str) -> ee.Geometry:
        """Load AOI from GeoJSON file"""
        import json
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        # Get first feature's coordinates
        coordinates = geojson_data['features'][0]['geometry']['coordinates'][0]
        return ee.Geometry.Polygon(coordinates)
    
    def load_aoi_from_master_geojson(self, city_name: str, geojson_path: str = 'data/aoi.geojson') -> ee.Geometry:
        """Load AOI from master GeoJSON file by city name"""
        import json
        
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Find the city feature
        city_feature = None
        for feature in geojson_data['features']:
            if feature['properties']['city_key'] == city_name.lower():
                city_feature = feature
                break
        
        if city_feature is None:
            available_cities = [f['properties']['city_key'] for f in geojson_data['features']]
            raise ValueError(f"City '{city_name}' not found in GeoJSON. Available: {available_cities}")
        
        # Convert to Earth Engine geometry
        coordinates = city_feature['geometry']['coordinates'][0]
        return ee.Geometry.Polygon(coordinates)
    
    def load_aoi_from_city_file(self, city_name: str) -> ee.Geometry:
        """Load AOI from individual city file in organized structure"""
        file_path = f'data/aoi/cities/{city_name.lower()}.geojson'
        return self.load_aoi_from_geojson(file_path)
    
    def load_aoi_from_tier_file(self, tier: str) -> ee.Geometry:
        """Load AOI from tier file in organized structure"""
        file_path = f'data/aoi/tiers/{tier.lower().replace(" ", "_")}.geojson'
        return self.load_aoi_from_geojson(file_path)
    
    def load_aoi_from_region_file(self, region: str) -> ee.Geometry:
        """Load AOI from region file in organized structure"""
        file_path = f'data/aoi/regions/{region.lower()}.geojson'
        return self.load_aoi_from_geojson(file_path)
    
    def get_available_datasets(self) -> Dict[str, Any]:
        """Get information about key datasets (IDs and purpose). Uses datasets.py if available."""
        if _HAS_DATASETS:
            return _get_datasets_info()
        return {
            'srtm_dem': 'USGS/SRTMGL1_003',
            'aster_dem': 'NASA/ASTER_GED/AG100_003',
            'sentinel2': 'COPERNICUS/S2_SR',
            'landsat8': 'LANDSAT/LC08/C02/T1_L2',
            'global_solar_atlas': 'projects/global-solar-atlas/solar-irradiance',
            'nasa_power': 'NASA/NCEP_RE/2m_temperature'
        }
    
    def get_elevation_data(self, aoi: ee.Geometry, dem_type: str = "srtm") -> ee.Image:
        """
        Get Digital Elevation Model for the AOI.

        Parameters
        ----------
        aoi : ee.Geometry
            Area of interest.
        dem_type : str
            "srtm" (default) or "fabdem". FABDEM requires sat-io community catalog access.
        """
        if _HAS_DATASETS:
            return _get_dem_datasets(aoi, dem_type=dem_type)
        dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(aoi)
        return dem
    
    def calculate_slope_aspect(self, dem: ee.Image) -> Dict[str, ee.Image]:
        """Calculate slope and aspect from DEM"""
        terrain = ee.Terrain.products(dem)
        slope = terrain.select('slope')
        aspect = terrain.select('aspect')
        
        return {
            'slope': slope,
            'aspect': aspect,
            'elevation': dem
        }
    
    def create_exclusion_mask(self, dem: ee.Image, aoi: ee.Geometry) -> ee.Image:
        """Create mask for unsuitable areas"""
        # Calculate slope
        slope = ee.Terrain.products(dem).select('slope')
        
        # Exclude very steep slopes (>30 degrees)
        steep_slopes = slope.gt(30)
        
        # Create exclusion mask (0 = excluded, 1 = suitable)
        exclusion_mask = steep_slopes.Not()
        
        return exclusion_mask

    def get_rooftop_candidate_stats(
        self,
        aoi: ee.Geometry,
        exclusion_mask: Optional[ee.Image] = None,
        year: Optional[int] = 2022,
        presence_threshold: float = 0.5,
        min_height_m: float = 0.0,
        reduce_scale_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Rooftop candidate area (m^2) from Open Buildings 2.5D + optional terrain exclusion.

        Requires scripts/rooftops.py and datasets.py. See rooftops.choose_reduce_scale_m for
        default scaling on large AOIs.
        """
        if not _HAS_ROOFTOPS:
            raise RuntimeError("rooftops module not found; ensure scripts/rooftops.py is on PYTHONPATH")
        return _get_rooftop_area_m2_info(
            aoi,
            year=year,
            presence_threshold=presence_threshold,
            min_height_m=min_height_m,
            exclusion_mask=exclusion_mask,
            scale_m=reduce_scale_m,
        )

    def get_merra_baseline_stats(
        self,
        aoi: ee.Geometry,
        start_year: int = 2020,
        end_year: int = 2024,
        scale_m: float = 11_132.0,
    ) -> Dict[str, Any]:
        """ERA5 mean annual GHI (kWh/m^2/year). Method name kept for backward compatibility."""
        if not _HAS_IRRADIANCE:
            raise RuntimeError("irradiance_baseline module not found")
        return _get_era5_baseline_info(
            aoi, start_year=start_year, end_year=end_year, scale_m=scale_m
        )

    def get_merra_latest_5y_baseline_stats(
        self,
        aoi: ee.Geometry,
        scale_m: float = 11_132.0,
    ) -> Dict[str, Any]:
        """Latest complete 5-year mean annual ERA5 GHI baseline."""
        if not _HAS_IRRADIANCE:
            raise RuntimeError("irradiance_baseline module not found")
        start_year, end_year = _latest_complete_5y_range()
        return _get_era5_baseline_info(aoi, start_year=start_year, end_year=end_year, scale_m=scale_m)

    def get_merra_range_stats(
        self,
        aoi: ee.Geometry,
        start_date: str,
        end_date_exclusive: str,
        scale_m: float = 11_132.0,
    ) -> Dict[str, Any]:
        """Arbitrary date-range ERA5 GHI totals/annualized stats."""
        if not _HAS_IRRADIANCE:
            raise RuntimeError("irradiance_baseline module not found")
        return _get_era5_range_info(
            aoi=aoi,
            start_date=start_date,
            end_date_exclusive=end_date_exclusive,
            scale_m=scale_m,
        )

    def get_roof_masked_merra_baseline_stats(
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
        """
        Pre-penalty rooftop baseline: candidate roof area (m2) x regional ERA5 GHI.

        Returns:
          roof_area_m2                    -- total candidate rooftop area at 4m resolution
          regional_irradiance_kwh_m2_year -- ERA5 mean annual GHI for the region (~9 km)
          pre_penalty_total_kwh_year      -- theoretical max energy before shadow/soiling/efficiency

        Shadow and UHI penalties are applied in the next pipeline stage (scripts/penalties.py).
        """
        if not _HAS_ROOFTOPS or not _HAS_IRRADIANCE or not _HAS_ROOF_MASKED_IRRADIANCE:
            raise RuntimeError("Roof-masked irradiance modules not found; check module imports.")

        if start_year is None or end_year is None:
            start_year, end_year = _latest_complete_5y_range()

        try:
            from rooftops import build_rooftop_candidate_mask, apply_terrain_exclusion
        except ImportError:
            from scripts.rooftops import build_rooftop_candidate_mask, apply_terrain_exclusion

        try:
            from datasets import get_open_buildings_temporal
        except ImportError:
            from scripts.datasets import get_open_buildings_temporal

        buildings = get_open_buildings_temporal(aoi, year=roof_year)
        roof_mask = build_rooftop_candidate_mask(
            buildings,
            presence_threshold=presence_threshold,
            min_height_m=min_height_m,
        )
        if exclusion_mask is not None:
            roof_mask = apply_terrain_exclusion(
                roof_mask,
                exclusion_mask=exclusion_mask,
                buildings=buildings,
                scale_m=4.0,
            )

        return _get_roof_masked_era5_baseline_info(
            aoi=aoi,
            roof_mask=roof_mask,
            start_year=start_year,
            end_year=end_year,
            scale_m=scale_m,
        )

    def calculate_solar_potential(self, dem: ee.Image, aoi: ee.Geometry) -> Dict[str, ee.Image]:
        """Calculate basic solar potential metrics"""
        # Get terrain data
        terrain = self.calculate_slope_aspect(dem)
        
        # Get exclusion mask
        exclusion = self.create_exclusion_mask(dem, aoi)
        
        # Basic solar potential calculation (simplified)
        # In reality, you'd integrate with solar irradiance data
        solar_potential = terrain['slope'].multiply(-1).add(90).multiply(exclusion)
        
        return {
            'solar_potential': solar_potential,
            'slope': terrain['slope'],
            'aspect': terrain['aspect'],
            'exclusion_mask': exclusion
        }
    
    def export_results_to_geojson(self, results: Dict[str, Any], output_path: str):
        """Export analysis results to GeoJSON format"""
        import json
        
        # Convert Earth Engine results to GeoJSON format
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # This is a simplified export - you'd need to convert EE images to GeoJSON
        # For now, just save the metadata
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results exported to {output_path}")