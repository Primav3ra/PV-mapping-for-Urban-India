# Custom AOI Files

This folder contains custom Area of Interest (AOI) files created by users.

## 🚀 Quick Start

1. Copy `template.geojson` to create your custom AOI
2. Modify the coordinates in the geometry section
3. Update the properties (name, description, etc.)
4. Use the file in your solar mapping analysis

## 📍 Coordinate Format

Coordinates should be in [longitude, latitude] format:
- **Longitude**: -180 to 180 (negative for west, positive for east)
- **Latitude**: -90 to 90 (negative for south, positive for north)

## 📝 Example Custom AOI

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "My Custom City",
        "description": "Custom area for solar analysis",
        "created_by": "your_name",
        "date_created": "2024-01-01"
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [77.0, 28.4],
          [77.4, 28.4],
          [77.4, 28.8],
          [77.0, 28.8],
          [77.0, 28.4]
        ]]
      }
    }
  ]
}
```

## 🛠️ Usage in Code

```python
from scripts.ain_solar_mapper import SolarPotentialMapper

mapper = SolarPotentialMapper('your-project-id')

# Load custom AOI
mapper.set_aoi_by_coordinates([
    [77.0, 28.4],
    [77.4, 28.4], 
    [77.4, 28.8],
    [77.0, 28.8],
    [77.0, 28.4]
])

# Or load from custom file
mapper.set_aoi_from_geojson('custom_city', 'data/aoi/custom/my_custom_city.geojson')
```

## 💡 Tips

- Use online mapping tools to get accurate coordinates
- Ensure the polygon is closed (first and last coordinates are the same)
- Keep AOI size reasonable for analysis performance
- Add meaningful descriptions for future reference
