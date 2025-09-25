# AOI (Area of Interest) Files

This directory contains organized Area of Interest files for the Solar Mapping project.

## 📁 Directory Structure

```
data/aoi/
├── cities/          # Individual city AOI files
├── tiers/           # Tier-based AOI files  
├── regions/         # Regional AOI files
├── custom/          # Custom AOI templates
└── README.md        # This file
```

## 🏙️ Cities Directory

Contains individual GeoJSON files for each city:
- `delhi.geojson` - Delhi (Tier 1)
- `mumbai.geojson` - Mumbai (Tier 1)
- `bangalore.geojson` - Bangalore (Tier 1)
- `chennai.geojson` - Chennai (Tier 1)
- `kolkata.geojson` - Kolkata (Tier 1)
- `hyderabad.geojson` - Hyderabad (Tier 1)
- `jaipur.geojson` - Jaipur (Tier 2)
- `ahmedabad.geojson` - Ahmedabad (Tier 2)
- `pune.geojson` - Pune (Tier 2)
- `kochi.geojson` - Kochi (Tier 2)
- `chandigarh.geojson` - Chandigarh (Tier 2)
- `indore.geojson` - Indore (Tier 3)
- `bhopal.geojson` - Bhopal (Tier 3)
- `coimbatore.geojson` - Coimbatore (Tier 3)
- `vadodara.geojson` - Vadodara (Tier 3)

## 📊 Tiers Directory

Contains files grouped by city tier:
- `tier_1.geojson` - All Tier 1 cities (6 cities)
- `tier_2.geojson` - All Tier 2 cities (5 cities)
- `tier_3.geojson` - All Tier 3 cities (4 cities)

## 🌍 Regions Directory

Contains files grouped by geographical region:
- `north.geojson` - North India (Delhi, Chandigarh, Jaipur)
- `south.geojson` - South India (Bangalore, Chennai, Kochi, Coimbatore)
- `west.geojson` - West India (Mumbai, Pune, Ahmedabad, Vadodara)
- `east.geojson` - East India (Kolkata)
- `central.geojson` - Central India (Hyderabad, Indore, Bhopal)

## 🛠️ Custom Directory

Contains templates for creating custom AOI files:
- `template.geojson` - Template for custom AOI creation
- `README.md` - Instructions for creating custom AOIs

## 💻 Usage Examples

### Loading Individual Cities
```python
from scripts.ain_solar_mapper import SolarPotentialMapper

mapper = SolarPotentialMapper('your-project-id')

# Load from individual city file
mapper.set_aoi_from_city_file('delhi')

# Load from master file
mapper.set_aoi_from_geojson('mumbai')

# Load from tier file
mapper.set_aoi_from_tier_file('tier_1')

# Load from region file
mapper.set_aoi_from_region_file('south')
```

### Using Utility Functions
```python
from scripts.utility import SolarMappingUtils

utils = SolarMappingUtils('your-project-id')

# Load individual city
aoi = utils.load_aoi_from_city_file('bangalore')

# Load tier-based AOI
aoi = utils.load_aoi_from_tier_file('tier_2')

# Load regional AOI
aoi = utils.load_aoi_from_region_file('west')
```

## 🔄 Regenerating AOI Files

To regenerate all AOI files with the organized structure:

```bash
python scripts/simple_aoi_generator.py
```

This will:
1. Create the master `data/aoi.geojson` file
2. Generate individual city files in `data/aoi/cities/`
3. Generate tier-based files in `data/aoi/tiers/`
4. Generate regional files in `data/aoi/regions/`

## 📋 File Format

All AOI files follow the GeoJSON FeatureCollection format:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "City Name",
        "tier": "Tier 1",
        "description": "City description",
        "population": 1000000,
        "state": "State Name",
        "area_km2": 100.0,
        "city_key": "city_name"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [longitude1, latitude1],
          [longitude2, latitude2],
          [longitude3, latitude3],
          [longitude4, latitude4],
          [longitude1, latitude1]
        ]]
      }
    }
  ]
}
```

## 🎯 Benefits of Organized Structure

1. **Clean Organization**: Files are logically grouped by type
2. **Easy Navigation**: Quick access to specific cities, tiers, or regions
3. **Scalable**: Easy to add new cities or create new groupings
4. **Flexible**: Multiple ways to load AOI data
5. **Maintainable**: Clear separation of concerns

## 📝 Notes

- All coordinates are in [longitude, latitude] format
- City keys are lowercase (e.g., 'delhi', 'mumbai')
- Tier names use underscores in filenames (e.g., 'tier_1.geojson')
- Region names are lowercase (e.g., 'north.geojson')
