# Rooftop Solar Mapping for Urban India

Geospatial pipeline + web app for estimating **rooftop solar PV yield** for urban India using **Google Earth Engine (GEE)** datasets, served via a **FastAPI** backend and an interactive **Leaflet** map UI.

## What it does

- Builds a rooftop candidate mask from Open Buildings 2.5D + slope exclusion
- Computes an irradiance baseline from ERA5-Land GHI for a user-selected time window
- Applies physics-inspired penalty layers (shadow, UHI temperature derate, soiling)
- Returns per-building yield and a breakdown of contributing factors via REST APIs

## Architecture (high level)

```
Google Earth Engine
  ├─ Open Buildings 2.5D Temporal (4 m)   → rooftop mask + heights
  ├─ Open Buildings v3 Polygons (vector)  → building footprints
  ├─ ERA5-Land Hourly (~9 km)             → GHI baseline
  ├─ ERA5 Hourly (~28 km)                → direct fraction / beam ratio
  ├─ MODIS MOD11A2 (1 km)                → daytime LST for UHI
  └─ MODIS MCD19A2 (1 km)                → MAIAC AOD for soiling

FastAPI backend  →  /api/* endpoints
Leaflet frontend →  map UI served from app/static/
```

## Setup

**Prerequisites**

- Python 3.9+
- A Google Earth Engine account
- A Google Cloud project with the Earth Engine API enabled

```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS / WSL:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

earthengine authenticate

# Set your Earth Engine project ID
# Windows (PowerShell):
$env:GEE_PROJECT_ID="pv-mapping-india"
# Linux / macOS / WSL:
export GEE_PROJECT_ID="pv-mapping-india"
```

> On Debian/Ubuntu/WSL, use `python3` instead of `python` if the `python` command is not found.

## Run locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## API (summary)

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/presets` | GET | Supported modes and year bounds |
| `/api/baseline` | POST | AOI rooftop area + ERA5 irradiance summary |
| `/api/yield` | POST | Per-building PV yield with penalty breakdowns |

### Temporal modes

Use `baseline_mode`:

| Mode | Required fields | Window |
|---|---|---|
| `yearly` | `year` | Full calendar year |
| `quarterly` | `year`, `quarter` (1–4) | Three calendar months |
| `daily` | `start_date`, `end_date_exclusive` | Single calendar day |

## Using the map UI

1. Click a point on the map to define an AOI square.
2. Choose a temporal mode (Yearly / Quarterly / Daily).
3. Run **baseline** to get rooftop area + irradiance summary.
4. Run **per-building yield** to compute PV yield for the selected building with a full penalty breakdown.

## Repo structure

```
app/
  main.py                 FastAPI backend
  static/                 Leaflet UI assets
scripts/
  datasets.py             GEE catalog IDs + loaders
  rooftops.py             rooftop mask + area stats
  irradiance_baseline.py  ERA5 baseline + beam fraction
  solar_geometry.py       solar altitude/azimuth sampling
  penalties.py            Shadow / UHI / Soiling penalty logic
  utility.py              orchestration utilities
scripts/tests/            basic tests
```

## License

Academic / research use. GEE datasets are subject to their respective licenses (Open Buildings: CC BY 4.0, ERA5: Copernicus, MODIS: NASA open data, SRTM: USGS public domain).
