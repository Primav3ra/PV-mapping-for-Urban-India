"""
Centralized dataset loaders for the PV mapping pipeline.
Uses Google Earth Engine (GEE) for: elevation, buildings, optical imagery, and LST.
All catalog IDs and band names are documented for reproducibility.
"""

import ee
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# GEE catalog IDs (single source of truth)
# ---------------------------------------------------------------------------

CATALOG = {
    # Elevation: bare-earth DEM for slope/terrain and shadow ground plane
    "srtm_dem": "USGS/SRTMGL1_003",
    "fabdem": "projects/sat-io/open-datasets/FABDEM",  # Community catalog; 30m, buildings/forest removed
    # Buildings: 2.5D heights and presence at ~4m effective resolution
    "open_buildings_temporal": "GOOGLE/Research/open-buildings-temporal/v1",
    # Optical: for DBSI (soiling), NDVI, cloud-free composites
    "sentinel2_sr": "COPERNICUS/S2_SR_HARMONIZED",  # Surface reflectance, harmonized with S2_SR
    "sentinel2_sr_legacy": "COPERNICUS/S2_SR",
    # Land surface temperature: for UHI
    "modis_lst": "MODIS/061/MOD11A2",
    # MERRA-2 hourly radiation (surface incoming shortwave SWGDN)
    "merra_rad": "NASA/GSFC/MERRA/rad/2",
}


def get_dem(aoi: ee.Geometry, dem_type: str = "srtm") -> ee.Image:
    """
    Load a single DEM image clipped to the AOI.

    Parameters
    ----------
    aoi : ee.Geometry
        Area of interest (polygon).
    dem_type : str
        One of:
        - "srtm": SRTM 30m global DEM (default).
        - "fabdem": FABDEM 30m, buildings/forest removed (requires sat-io access).

    Returns
    -------
    ee.Image
        Single-band elevation in meters, clipped to aoi.
    """
    if dem_type == "srtm":
        dem = ee.Image(CATALOG["srtm_dem"]).select("elevation").clip(aoi)
    elif dem_type == "fabdem":
        # FABDEM is ImageCollection (1 band per tile); mosaic and use first band as elevation
        fabdem = ee.ImageCollection(CATALOG["fabdem"]).filterBounds(aoi)
        dem = fabdem.mosaic().select(0).rename("elevation").clip(aoi)
    else:
        raise ValueError(f"dem_type must be 'srtm' or 'fabdem', got: {dem_type}")
    return dem


def get_open_buildings_temporal(
    aoi: ee.Geometry,
    year: Optional[int] = None,
) -> ee.Image:
    """
    Load Open Buildings 2.5D Temporal: building presence, height, fractional count.
    Effective resolution ~4 m; annual composites 2016-2023.

    Parameters
    ----------
    aoi : ee.Geometry
        Area of interest.
    year : int, optional
        Year to use (2016-2023). If None, uses latest available (2023).

    Returns
    -------
    ee.Image
        Bands: building_presence, building_height, building_fractional_count.
    """
    col = ee.ImageCollection(CATALOG["open_buildings_temporal"]).filterBounds(aoi)
    if year is not None:
        # Filter to images that represent this year (inference_time_epoch_s in that year)
        start_ts = int(datetime(year, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(year + 1, 1, 1).timestamp() * 1000)
        col = col.filter(
            ee.Filter.And(
                ee.Filter.gte("system:time_start", start_ts),
                ee.Filter.lt("system:time_start", end_ts),
            )
        )
    # Mosaic all tiles that intersect AOI for the selected time
    img = col.mosaic().clip(aoi)
    return img.select(["building_presence", "building_height", "building_fractional_count"])


def get_sentinel2_composite(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: Optional[int] = 20,
) -> ee.Image:
    """
    Cloud-masked median composite of Sentinel-2 L2A surface reflectance.

    Parameters
    ----------
    aoi : ee.Geometry
        Area of interest.
    start_date, end_date : str
        Format "YYYY-MM-DD".
    cloud_pct : int, optional
        Max cloud probability (SCL / cloud mask); default 20%.

    Returns
    -------
    ee.Image
        Multiband surface reflectance (e.g. B2, B3, B4, B8, B11, B12 for DBSI/NDVI).
    """
    col = (
        ee.ImageCollection(CATALOG["sentinel2_sr"])
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    # Cloud mask using QA60 (bits 10 and 11 = opaque clouds and cirrus)
    def mask_s2(img: ee.Image) -> ee.Image:
        qa = img.select("QA60")
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(cloud_mask)
    col = col.map(mask_s2)
    median = col.median().clip(aoi)
    return median


def get_modis_lst_composite(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    use_night: bool = True,
) -> ee.Image:
    """
    Median Land Surface Temperature from MODIS MOD11A2 (8-day LST).
    Used for UHI: typically use night-time LST.

    Parameters
    ----------
    aoi : ee.Geometry
        Area of interest.
    start_date, end_date : str
        "YYYY-MM-DD".
    use_night : bool
        If True, use LST_Night_1km (K); if False, LST_Day_1km.

    Returns
    -------
    ee.Image
        Single band: LST in Kelvin (scale 0.02). Convert to °C: (LST * 0.02) - 273.15.
    """
    band = "LST_Night_1km" if use_night else "LST_Day_1km"
    col = (
        ee.ImageCollection(CATALOG["modis_lst"])
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select(band)
    )
    return col.median().clip(aoi).rename("LST")


def get_available_datasets() -> dict:
    """Return a short description of each dataset (ID and purpose)."""
    return {
        "srtm_dem": {"id": CATALOG["srtm_dem"], "purpose": "30m DEM, global"},
        "fabdem": {"id": CATALOG["fabdem"], "purpose": "30m bare-earth DEM, buildings/forest removed"},
        "open_buildings_temporal": {
            "id": CATALOG["open_buildings_temporal"],
            "purpose": "Building presence, height, fractional count; ~4m; 2016-2023",
        },
        "sentinel2_sr": {"id": CATALOG["sentinel2_sr"], "purpose": "Surface reflectance for DBSI, NDVI"},
        "modis_lst": {"id": CATALOG["modis_lst"], "purpose": "8-day LST for UHI (day/night)"},
        "merra_rad": {
            "id": CATALOG["merra_rad"],
            "purpose": "MERRA-2 hourly radiation; SWGDN baseline (kWh/m2/yr climatology)",
        },
    }
