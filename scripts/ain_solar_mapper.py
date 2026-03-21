import ee
from utility import SolarMappingUtils
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

class CityTier(Enum):
    TIER_1 = "Tier 1"
    TIER_2 = "Tier 2" 
    TIER_3 = "Tier 3"

@dataclass
class CityAOI:
    name: str
    tier: CityTier
    coordinates: List[List[float]]
    description: str
    population: Optional[int] = None
    state: Optional[str] = None
    area_km2: Optional[float] = None

class SolarPotentialMapper:
    def __init__(self, project_id: str):
        self.utils = SolarMappingUtils(project_id)
        self.cities = self._get_cities_data()
        self.aoi = None
        self.current_city = None
        self.datasets = self.utils.get_available_datasets()
    
    def _get_cities_data(self) -> Dict[str, CityAOI]:
        """Get all city definitions"""
        return {
            # TIER 1 CITIES (Metro cities)
            "delhi": CityAOI(
                name="Delhi",
                tier=CityTier.TIER_1,
                coordinates=[
                    [77.0, 28.4], [77.4, 28.4], [77.4, 28.8], [77.0, 28.8], [77.0, 28.4]
                ],
                description="National Capital Region - Central Delhi",
                population=32000000,
                state="Delhi",
                area_km2=1484
            ),
            "mumbai": CityAOI(
                name="Mumbai",
                tier=CityTier.TIER_1,
                coordinates=[
                    [72.7, 18.8], [73.0, 18.8], [73.0, 19.3], [72.7, 19.3], [72.7, 18.8]
                ],
                description="Financial capital of India",
                population=20000000,
                state="Maharashtra",
                area_km2=603
            ),
            "bangalore": CityAOI(
                name="Bangalore",
                tier=CityTier.TIER_1,
                coordinates=[
                    [77.4, 12.8], [77.8, 12.8], [77.8, 13.2], [77.4, 13.2], [77.4, 12.8]
                ],
                description="Silicon Valley of India",
                population=12000000,
                state="Karnataka",
                area_km2=741
            ),
            "chennai": CityAOI(
                name="Chennai",
                tier=CityTier.TIER_1,
                coordinates=[
                    [80.0, 12.8], [80.4, 12.8], [80.4, 13.2], [80.0, 13.2], [80.0, 12.8]
                ],
                description="Gateway to South India",
                population=11000000,
                state="Tamil Nadu",
                area_km2=426
            ),
            "kolkata": CityAOI(
                name="Kolkata",
                tier=CityTier.TIER_1,
                coordinates=[
                    [88.2, 22.4], [88.6, 22.4], [88.6, 22.8], [88.2, 22.8], [88.2, 22.4]
                ],
                description="Cultural capital of India",
                population=15000000,
                state="West Bengal",
                area_km2=205
            ),
            "hyderabad": CityAOI(
                name="Hyderabad",
                tier=CityTier.TIER_1,
                coordinates=[
                    [78.2, 17.2], [78.6, 17.2], [78.6, 17.6], [78.2, 17.6], [78.2, 17.2]
                ],
                description="City of Pearls",
                population=10000000,
                state="Telangana",
                area_km2=650
            ),
            
            # TIER 2 CITIES (State capitals and major cities)
            "jaipur": CityAOI(
                name="Jaipur",
                tier=CityTier.TIER_2,
                coordinates=[
                    [75.6, 26.6], [76.0, 26.6], [76.0, 27.0], [75.6, 27.0], [75.6, 26.6]
                ],
                description="Pink City - Capital of Rajasthan",
                population=4000000,
                state="Rajasthan",
                area_km2=467
            ),
            "ahmedabad": CityAOI(
                name="Ahmedabad",
                tier=CityTier.TIER_2,
                coordinates=[
                    [72.4, 23.0], [72.8, 23.0], [72.8, 23.4], [72.4, 23.4], [72.4, 23.0]
                ],
                description="Manchester of India",
                population=8000000,
                state="Gujarat",
                area_km2=464
            ),
            "pune": CityAOI(
                name="Pune",
                tier=CityTier.TIER_2,
                coordinates=[
                    [73.6, 18.4], [74.0, 18.4], [74.0, 18.8], [73.6, 18.8], [73.6, 18.4]
                ],
                description="Oxford of the East",
                population=7000000,
                state="Maharashtra",
                area_km2=331
            ),
            "kochi": CityAOI(
                name="Kochi",
                tier=CityTier.TIER_2,
                coordinates=[
                    [76.0, 9.8], [76.4, 9.8], [76.4, 10.2], [76.0, 10.2], [76.0, 9.8]
                ],
                description="Queen of Arabian Sea",
                population=2000000,
                state="Kerala",
                area_km2=95
            ),
            "chandigarh": CityAOI(
                name="Chandigarh",
                tier=CityTier.TIER_2,
                coordinates=[
                    [76.6, 30.6], [77.0, 30.6], [77.0, 31.0], [76.6, 31.0], [76.6, 30.6]
                ],
                description="City Beautiful - Planned city",
                population=1200000,
                state="Chandigarh",
                area_km2=114
            ),
            
            # TIER 3 CITIES (Emerging cities)
            "indore": CityAOI(
                name="Indore",
                tier=CityTier.TIER_3,
                coordinates=[
                    [75.6, 22.6], [76.0, 22.6], [76.0, 23.0], [75.6, 23.0], [75.6, 22.6]
                ],
                description="Commercial capital of Madhya Pradesh",
                population=3000000,
                state="Madhya Pradesh",
                area_km2=530
            ),
            "bhopal": CityAOI(
                name="Bhopal",
                tier=CityTier.TIER_3,
                coordinates=[
                    [77.2, 23.0], [77.6, 23.0], [77.6, 23.4], [77.2, 23.4], [77.2, 23.0]
                ],
                description="City of Lakes",
                population=2000000,
                state="Madhya Pradesh",
                area_km2=285
            ),
            "coimbatore": CityAOI(
                name="Coimbatore",
                tier=CityTier.TIER_3,
                coordinates=[
                    [76.8, 10.8], [77.2, 10.8], [77.2, 11.2], [76.8, 11.2], [76.8, 10.8]
                ],
                description="Manchester of South India",
                population=2000000,
                state="Tamil Nadu",
                area_km2=246
            ),
            "vadodara": CityAOI(
                name="Vadodara",
                tier=CityTier.TIER_3,
                coordinates=[
                    [73.0, 22.2], [73.4, 22.2], [73.4, 22.6], [73.0, 22.6], [73.0, 22.2]
                ],
                description="Cultural capital of Gujarat",
                population=2000000,
                state="Gujarat",
                area_km2=235
            )
        }
        
    def set_aoi_by_city(self, city_name: str):
        """Set AOI by city name"""
        if city_name.lower() not in self.cities:
            available = list(self.cities.keys())
            raise ValueError(f"City '{city_name}' not found. Available cities: {available}")
        
        self.current_city = self.cities[city_name.lower()]
        self.aoi = ee.Geometry.Polygon(self.current_city.coordinates)
        print(f"[OK] AOI set to: {self.current_city.name} ({self.current_city.tier.value})")
        print(f"     State: {self.current_city.state}")
        print(f"     Population: {self.current_city.population:,}")
        print(f"     Area: {self.current_city.area_km2} km^2")
    
    def set_aoi_by_coordinates(self, coordinates: List[List[float]]):
        """Set AOI by custom coordinates"""
        self.aoi = ee.Geometry.Polygon(coordinates)
        self.current_city = None
        print("[OK] AOI set to custom coordinates")
    
    def set_aoi_from_geojson(self, city_name: str, geojson_path: str = 'data/aoi.geojson'):
        """Set AOI from GeoJSON file"""
        self.aoi = self.utils.load_aoi_from_master_geojson(city_name, geojson_path)
        # Try to get city info from AOI manager
        try:
            self.current_city = self.aoi_manager.get_city(city_name)
            print(f"[OK] AOI loaded from GeoJSON: {self.current_city.name}")
        except Exception:
            self.current_city = None
            print(f"[OK] AOI loaded from GeoJSON: {city_name}")
    
    def set_aoi_from_city_file(self, city_name: str):
        """Set AOI from individual city file in organized structure"""
        self.aoi = self.utils.load_aoi_from_city_file(city_name)
        try:
            self.current_city = self.aoi_manager.get_city(city_name)
            print(f"[OK] AOI loaded from city file: {self.current_city.name}")
        except Exception:
            self.current_city = None
            print(f"[OK] AOI loaded from city file: {city_name}")
    
    def set_aoi_from_tier_file(self, tier: str):
        """Set AOI from tier file in organized structure"""
        self.aoi = self.utils.load_aoi_from_tier_file(tier)
        self.current_city = None
        print(f"[OK] AOI loaded from tier file: {tier}")
    
    def set_aoi_from_region_file(self, region: str):
        """Set AOI from region file in organized structure"""
        self.aoi = self.utils.load_aoi_from_region_file(region)
        self.current_city = None
        print(f"[OK] AOI loaded from region file: {region}")
    
    def get_elevation_data(self) -> ee.Image:
        """Get Digital Elevation Model for current AOI"""
        if self.aoi is None:
            raise ValueError("AOI not set. Use set_aoi_by_city() or set_aoi_by_coordinates()")
        return self.utils.get_elevation_data(self.aoi)
    
    def calculate_slope_aspect(self, dem: ee.Image) -> Dict[str, ee.Image]:
        """Calculate slope and aspect from DEM"""
        return self.utils.calculate_slope_aspect(dem)
    
    def create_exclusion_mask(self) -> ee.Image:
        """Create mask for unsuitable areas"""
        if self.aoi is None:
            raise ValueError("AOI not set. Use set_aoi_by_city() or set_aoi_by_coordinates()")
        dem = self.get_elevation_data()
        return self.utils.create_exclusion_mask(dem, self.aoi)
    
    def calculate_solar_potential(self) -> Dict[str, ee.Image]:
        """Calculate solar potential for current AOI"""
        if self.aoi is None:
            raise ValueError("AOI not set. Use set_aoi_by_city() or set_aoi_by_coordinates()")
        
        dem = self.get_elevation_data()
        return self.utils.calculate_solar_potential(dem, self.aoi)
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the solar potential analysis for current AOI"""
        if self.aoi is None:
            raise ValueError("AOI not set. Use set_aoi_by_city() or set_aoi_by_coordinates()")
        
        print("[INFO] Running solar potential analysis...")
        
        # Get elevation data
        dem = self.get_elevation_data()
        print("[OK] Elevation data loaded")
        
        # Calculate terrain
        terrain = self.calculate_slope_aspect(dem)
        print("[OK] Slope and aspect calculated")
        
        # Create exclusion mask
        exclusion = self.create_exclusion_mask()
        print("[OK] Exclusion mask created")

        # Rooftop candidate mask area (Open Buildings 2.5D + terrain exclusion)
        rooftop_stats = self.utils.get_rooftop_candidate_stats(
            self.aoi,
            exclusion_mask=exclusion,
            year=2022,
            presence_threshold=0.5,
            min_height_m=0.0,
        )
        print("[OK] Rooftop candidate area computed")

        # MERRA-2 mean annual surface incoming shortwave (baseline resource, coarse grid)
        irradiance_stats = self.utils.get_merra_baseline_stats(
            self.aoi, start_year=2015, end_year=2019
        )
        print("[OK] MERRA-2 baseline irradiance (mean annual SW) computed")
        
        # Calculate solar potential
        solar_data = self.calculate_solar_potential()
        print("[OK] Solar potential calculated")
        
        # Get basic statistics
        aoi_area = self.aoi.area().divide(1e6).getInfo()  # km^2
        
        return {
            "city": self.current_city.name if self.current_city else "Custom AOI",
            "aoi_area_km2": aoi_area,
            "analysis_completed": True,
            "timestamp": ee.Date.now().format('yyyy-MM-dd HH:mm:ss').getInfo(),
            "elevation_range": {
                "min": dem.reduceRegion(ee.Reducer.min(), self.aoi, 1000).getInfo(),
                "max": dem.reduceRegion(ee.Reducer.max(), self.aoi, 1000).getInfo()
            },
            "rooftop": rooftop_stats,
            "irradiance_baseline": irradiance_stats,
        }
    
    def analyze_multiple_cities(self, city_names: List[str]) -> Dict[str, Any]:
        """Analyze multiple cities in batch"""
        results = {}
        
        for city_name in city_names:
            print(f"\n[INFO] Analyzing {city_name}...")
            try:
                self.set_aoi_by_city(city_name)
                results[city_name] = self.run_analysis()
            except Exception as e:
                print(f"[FAIL] Error analyzing {city_name}: {str(e)}")
                results[city_name] = {"error": str(e)}
        
        return results
    
    def analyze_by_tier(self, tier: CityTier) -> Dict[str, Any]:
        """Analyze all cities in a specific tier"""
        cities_in_tier = [city for city in self.cities.values() if city.tier == tier]
        city_names = [city.name.lower() for city in cities_in_tier]
        
        print(f"[INFO] Analyzing all {tier.value} cities: {len(cities_in_tier)} cities")
        return self.analyze_multiple_cities(city_names)
    
    def analyze_by_region(self, region: str) -> Dict[str, Any]:
        """Analyze all cities in a specific region"""
        regions = {
            "north": ["delhi", "chandigarh", "jaipur"],
            "south": ["bangalore", "chennai", "kochi", "coimbatore"],
            "west": ["mumbai", "pune", "ahmedabad", "vadodara"],
            "east": ["kolkata"],
            "central": ["hyderabad", "indore", "bhopal"]
        }
        
        if region.lower() not in regions:
            available = list(regions.keys())
            raise ValueError(f"Region '{region}' not found. Available regions: {available}")
        
        city_names = regions[region.lower()]
        cities_in_region = [self.cities[name] for name in city_names if name in self.cities]
        
        print(f"[INFO] Analyzing {region} region: {len(cities_in_region)} cities")
        return self.analyze_multiple_cities(city_names)
    
    def export_results(self, results: Dict[str, Any], output_path: str):
        """Export analysis results to file"""
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[OK] Results exported to {output_path}")
    
    def get_city_info(self, city_name: str) -> Dict[str, Any]:
        """Get detailed information about a city"""
        if city_name.lower() not in self.cities:
            available = list(self.cities.keys())
            raise ValueError(f"City '{city_name}' not found. Available cities: {available}")
        
        city = self.cities[city_name.lower()]
        return {
            "name": city.name,
            "tier": city.tier.value,
            "description": city.description,
            "population": city.population,
            "state": city.state,
            "area_km2": city.area_km2,
            "coordinates": city.coordinates
        }
    
    def list_available_cities(self):
        """List all available cities"""
        print("AVAILABLE CITIES FOR SOLAR MAPPING")
        print("=" * 50)
        
        for tier in CityTier:
            cities_in_tier = [city for city in self.cities.values() if city.tier == tier]
            print(f"\n{tier.value} CITIES:")
            for city in cities_in_tier:
                print(f"  - {city.name} ({city.state}) - Pop: {city.population:,}")
        
        print(f"\n[INFO] Total: {len(self.cities)} cities available")
    
    def export_aoi_to_geojson(self, city_name: str, output_path: str):
        """Export city AOI to GeoJSON file"""
        if city_name.lower() not in self.cities:
            available = list(self.cities.keys())
            raise ValueError(f"City '{city_name}' not found. Available cities: {available}")
        
        city = self.cities[city_name.lower()]
        
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": city.name,
                        "tier": city.tier.value,
                        "description": city.description,
                        "population": city.population,
                        "state": city.state,
                        "area_km2": city.area_km2
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [city.coordinates]
                    }
                }
            ]
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"[OK] AOI exported to {output_path}")