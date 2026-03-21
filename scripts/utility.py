import ee
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

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
    from irradiance_baseline import get_merra_baseline_info as _get_merra_baseline_info
    _HAS_IRRADIANCE = True
except ImportError:
    try:
        from scripts.irradiance_baseline import get_merra_baseline_info as _get_merra_baseline_info
        _HAS_IRRADIANCE = True
    except ImportError:
        _HAS_IRRADIANCE = False


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
        start_year: int = 2015,
        end_year: int = 2019,
        scale_m: float = 50_000.0,
    ) -> Dict[str, Any]:
        """
        Mean annual surface incoming shortwave (all-sky) from MERRA-2 SWGDN, kWh/m^2/year.

        Coarse grid (~50-70 km); use as regional climatology before rooftop penalties.
        """
        if not _HAS_IRRADIANCE:
            raise RuntimeError("irradiance_baseline module not found")
        return _get_merra_baseline_info(
            aoi, start_year=start_year, end_year=end_year, scale_m=scale_m
        )

    def get_solar_irradiance_data(self, aoi: ee.Geometry) -> ee.Image:
        """Get solar irradiance data for the AOI"""
        # Using NASA POWER data as a proxy for solar irradiance
        # This is a simplified approach - you may want to use Global Solar Atlas
        solar_data = ee.ImageCollection('NASA/NCEP_RE/2m_temperature').first()
        return solar_data.clip(aoi)
    
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

# Test the connection
PROJECT_ID = 'pv-mapping-india' 
ee.Initialize(project=PROJECT_ID)
print(ee.String('Connection successful!').getInfo())