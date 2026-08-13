"""
Penalty layers applied to the ERA5 baseline irradiance at 4m resolution.

Penalty layers:
  ShadowPenalty  -- 2.5D building-height shadow model, blocks direct beam (Open Buildings raster, 4m)
  SkyViewFactor  -- 2.5D building-height sky occlusion, blocks diffuse    (Open Buildings raster, 4m)
  UHIPenalty     -- Temperature derate from UHI effect  (MODIS LST, 1km)
  SoilingPenalty -- Dust/soiling derate from MODIS MAIAC AOD (1km, 550nm)

Combined net energy formula (see net_irradiance_image):
  E_net = GHI_period
          * [ diffuse_fraction * SVF + beam_fraction * (1 - shadow_frequency) ]  [Shadow + SkyViewFactor]
          * uhi_derate_factor           [UHIPenalty,     scalar ~0.97-1.00]
          * soiling_retention_factor    [SoilingPenalty, scalar ~0.94-1.00]
          * panel_efficiency * PR * packing_factor * roof_area_m2

Research basis:
  Shadow  : 2.5D geometric shadow casting. Known approximations documented in
            ShadowPenalty class docstring.
            Shadow frequency image is data-driven from Open Buildings 2.5D at 4m.
            Beam fraction is sampled from ERA5 HOURLY direct radiation band and
            applied so only the beam component is attenuated by shadows:
              net = GHI * (1 - shadow_frequency * beam_fraction)
            Diffuse (~30-45 % of GHI in urban India) reaches rooftops from the
            open sky hemisphere and is NOT blocked by surrounding buildings
            (rooftop Sky View Factor ~0.85-0.95; SVF correction deferred).

  UHI     : De Soto et al. (2006) / IEC 60891 temperature-coefficient derating.
              P_loss = gamma * delta_T   [gamma ~ -0.004 /degC for c-Si]
            UHI intensity delta_T estimated from MODIS daytime LST minus a 20 km
            focal-mean background (removes the regional temperature gradient).
            Observed UHI in Indian cities: 2-6 degC
            (Mohan et al. 2011; Bhati & Mohan 2018).
            Expected derate range: ~0.99 (2 degC) to ~0.976 (6 degC).

  Soiling : MODIS MAIAC AOD at 550 nm (MCD19A2 v061; Lyapustin et al. 2011).
            Direct measure of atmospheric aerosol column loading -- the actual
            driver of dry-deposition soiling on PV cover glass.
              soiling_loss = mean_AOD_550nm * SOILING_COEFFICIENT
            SOILING_COEFFICIENT = 0.08 /AOD_unit/year (Kimber et al. 2006;
            Sayyah et al. 2014; Mani & Pillai 2010).
            No output cap: loss follows directly from the measured AOD.
            Urban India AOD (0.5-1.2) is 3-5x rural; a genuine urban penalty.
"""
from __future__ import annotations

import math
import ee
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers shared across penalty classes
# ---------------------------------------------------------------------------

def _reduce_mean(image: ee.Image, band: str, aoi: ee.Geometry, scale: float) -> Optional[float]:
    """Mean of a single band over AOI. Returns None if no valid pixels."""
    raw = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()
    val = (raw or {}).get(band)
    return float(val) if val is not None else None


def _make_solar_positions() -> List[Tuple[float, float, float]]:
    """
    18 representative (alt_deg, az_deg, weight) positions: solstices x2, equinox x1,
    6 times of day each. weight = sin(alt_rad); equinox doubled for spring+autumn.
    Weights normalised to sum = 1. Solar geometry for Delhi 28.6 N.
    """
    seasons = [
        ("summer", [
            (8.0,  69.0), (28.0,  84.0), (58.0, 100.0),
            (84.0, 180.0),
            (58.0, 260.0), (28.0, 276.0), (8.0, 291.0),
        ]),
        ("winter", [
            (3.0, 120.0), (14.0, 136.0), (28.0, 150.0),
            (38.0, 180.0),
            (28.0, 210.0), (14.0, 224.0), (3.0, 240.0),
        ]),
        ("equinox", [
            (5.0,  90.0), (21.0,  98.0), (44.0, 112.0),
            (62.0, 180.0),
            (44.0, 248.0), (21.0, 262.0), (5.0, 270.0),
        ]),
    ]
    positions: List[Tuple[float, float, float]] = []
    for label, entries in seasons:
        repeat = 2 if label == "equinox" else 1
        for alt, az in entries:
            if alt < 2.0:
                continue
            positions.append((alt, az, math.sin(math.radians(alt)) * repeat))
    total = sum(w for _, _, w in positions)
    return [(a, z, w / total) for a, z, w in positions]


# ---------------------------------------------------------------------------
# Class 1 – ShadowPenalty
# ---------------------------------------------------------------------------

class ShadowPenalty:
    """
    2.5D shadow model off the Open Buildings height raster, weighted by how much sun
    each position actually delivers.

    Per sun position (alt, az) a building of height H throws a shadow H / tan(alt) metres
    long. Doing that exactly per pixel is too slow in EE, so it's approximated: project
    the shadow-length image along the shadow direction, focal_max over the same radius to
    catch any caster in range, and flag a pixel as shadowed when a taller neighbour reaches
    it (the height check is what stops a building shadowing itself).

    Positions are weighted by sin(altitude), so low winter/morning sun -- long shadows but
    little energy -- doesn't dominate the yearly figure.

    Rough edges: the focal_max kernel is a circle rather than directional, so shadow area
    runs maybe 5-10% high; and shadows only take out the direct beam -- diffuse is dealt
    with elsewhere (beam_fraction in net_irradiance_image + the SkyViewFactor layer), not
    here.
    """

    MAX_SHADOW_PIXELS: int = 100  # 100 px * 4 m/px = 400 m maximum shadow reach

    # Default positions (Delhi 28.6 N); overridden by dynamic solar_geometry module
    _DELHI_POSITIONS: List[Tuple[float, float, float]] = _make_solar_positions()

    @staticmethod
    def _mask_for_position(
        building_height: ee.Image,
        alt_deg: float,
        az_deg: float,
        pixel_size_m: float = 4.0,
    ) -> ee.Image:
        """Binary shadow mask (1=shadow, 0=sunlit) for one solar geometry."""
        alt_rad = math.radians(max(alt_deg, 2.0))
        tan_alt = math.tan(alt_rad)

        shadow_len_px = building_height.divide(tan_alt * pixel_size_m)

        # Unit vector pointing in the shadow direction (opposite to sun azimuth)
        shadow_az_rad = math.radians(az_deg + 180.0)
        dx = math.sin(shadow_az_rad) * ShadowPenalty.MAX_SHADOW_PIXELS
        dy = math.cos(shadow_az_rad) * ShadowPenalty.MAX_SHADOW_PIXELS

        translated = shadow_len_px.translate(dx, dy)
        kernel = ee.Kernel.circle(
            radius=ShadowPenalty.MAX_SHADOW_PIXELS, units="pixels", normalize=False
        )
        dilated = translated.focal_max(kernel=kernel)
        caster_h = building_height.translate(dx, dy)

        return (
            dilated.gte(1.0)
            .And(caster_h.gt(building_height))
            .rename("in_shadow")
            .toUint8()
        )

    @staticmethod
    def frequency(
        building_height: ee.Image,
        solar_positions: Optional[List[Tuple]] = None,
        pixel_size_m: float = 4.0,
    ) -> ee.Image:
        """
        Insolation-weighted shadow frequency image [0, 1]. Band: shadow_frequency.
        0 = never in shadow; 1 = always in shadow across all weighted positions.
        """
        if solar_positions is None:
            pwt = ShadowPenalty._DELHI_POSITIONS
        elif len(solar_positions[0]) >= 3:
            # solar_positions entries are expected as:
            #   (alt_deg, az_deg_from_north, weight, ...metadata)
            pwt = [(a, z, w) for (a, z, w, *_rest) in solar_positions]
        else:
            # Fallback: entries do not include weights (e.g. (alt, az)).
            n = len(solar_positions)
            pwt = [(a, z, 1.0 / n) for (a, z, *_rest) in solar_positions]

        imgs = [
            ShadowPenalty._mask_for_position(building_height, a, z, pixel_size_m)
            .toFloat()
            .multiply(w)
            for a, z, w in pwt
        ]
        result = imgs[0]
        for img in imgs[1:]:
            result = result.add(img)
        return result.rename("shadow_frequency")


# ---------------------------------------------------------------------------
# Class 1b – SkyViewFactor  (diffuse counterpart of ShadowPenalty)
# ---------------------------------------------------------------------------

class SkyViewFactor:
    """
    The diffuse-light sibling of ShadowPenalty. Shadows block the direct beam; tall
    neighbours also hide part of the sky dome, so a roof doesn't collect the full
    diffuse either. SVF = the fraction of sky the roof can still see.

        SVF = 1 - avg over 8 compass directions of sin^2(horizon angle)

    where the horizon angle in a direction is the steepest obstruction we hit stepping
    outward. The sin^2 weighting is the usual flat-surface diffuse form (Oke). Net
    retention then reads: diffuse_fraction * SVF + beam_fraction * (1 - shadow_freq).

    Since we subtract self-height, only genuinely TALLER neighbours count, so roof SVF
    usually lands ~0.85-1.0. Street canyons would be far lower, but roof pixels aren't
    down in the canyon.
    """

    N_AZIMUTH: int = 8
    # how far out to look, in pixels (x4m). Roughly log-spaced -- the near buildings set
    # the horizon; anything far away barely subtends an angle.
    DIST_PX: Tuple[int, ...] = (1, 2, 4, 8, 16)

    @staticmethod
    def image(
        building_height: ee.Image,
        pixel_size_m: float = 4.0,
        n_azimuth: Optional[int] = None,
        dist_px: Optional[Tuple[int, ...]] = None,
    ) -> ee.Image:
        """Per-pixel Sky View Factor [0, 1]. Band: sky_view_factor."""
        n_az = int(n_azimuth or SkyViewFactor.N_AZIMUTH)
        dists = dist_px or SkyViewFactor.DIST_PX

        sin2_terms: List[ee.Image] = []
        for k in range(n_az):
            az = 2.0 * math.pi * k / n_az
            ux, uy = math.sin(az), math.cos(az)

            # walk outward in this direction, keep the steepest obstruction
            angle_imgs: List[ee.Image] = []
            for d in dists:
                dm = float(d) * pixel_size_m
                neighbour = building_height.translate(ux * d, uy * d)
                rise = neighbour.subtract(building_height).max(0.0)   # ignore anything shorter than us
                angle_imgs.append(rise.divide(dm).atan())
            horizon = angle_imgs[0]
            for a in angle_imgs[1:]:
                horizon = horizon.max(a)

            sin2_terms.append(horizon.sin().pow(2))

        acc = sin2_terms[0]
        for t in sin2_terms[1:]:
            acc = acc.add(t)
        mean_sin2 = acc.divide(n_az)

        return ee.Image(1.0).subtract(mean_sin2).clamp(0.0, 1.0).rename("sky_view_factor")


# ---------------------------------------------------------------------------
# Class 2 – UHIPenalty
# ---------------------------------------------------------------------------

class UHIPenalty:
    """
    Efficiency hit from the city running hotter than its surroundings (urban heat island).

    Panels lose output above 25 degC (STC): dP/P = gamma * (T_cell - 25), with
    gamma ~ -0.004 /degC for crystalline silicon (IEC 60891; De Soto 2006). Cell temp is
    the usual NOCT form, T_cell = T_ambient + (NOCT-20)/800 * G. We only charge for the
    *urban excess* here -- delta_T = urban - rural background -- so uhi_derate =
    1 + gamma*delta_T. For an Indian city that's small: 2-6 degC of UHI (Mohan et al. 2011)
    works out to roughly 0.8-2.4% off.

    Getting delta_T from MODIS: take the annual median daytime LST (MOD11A2, 1 km; raw*0.02
    = Kelvin, then -273.15), subtract a 30 km focal-mean "background". The 30 km is
    deliberate -- for a sprawling city like Delhi a tighter window sits entirely inside the
    heat island and the anomaly collapses to ~0. A full year of data keeps the composite
    stable across seasons.
    """

    MODIS_COLLECTION = "MODIS/061/MOD11A2"
    LST_DAY_BAND = "LST_Day_1km"
    LST_SCALE = 0.02          # raw integer * 0.02 = Kelvin (MODIS scale factor)
    K_TO_C_OFFSET = 273.15
    BACKGROUND_KERNEL_PX = 30  # 30 km at 1 km/pixel (large-city UHI footprint; see class docstring)
    DEFAULT_TEMP_COEFF = -0.004  # /degC, crystalline silicon (IEC 60891)

    @classmethod
    def _lst_celsius(cls, aoi: ee.Geometry, year: int) -> ee.Image:
        """Annual median daytime LST in degC for the given calendar year."""
        return (
            ee.ImageCollection(cls.MODIS_COLLECTION)
            .filterBounds(aoi)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select(cls.LST_DAY_BAND)
            .median()
            .multiply(cls.LST_SCALE)
            .subtract(cls.K_TO_C_OFFSET)
            .rename("LST_celsius")
        )

    @classmethod
    def stats(
        cls,
        aoi: ee.Geometry,
        start_date: str,
        temp_coeff: float = DEFAULT_TEMP_COEFF,
        scale_m: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        UHI intensity + derate for the AOI. start_date just supplies the year for the
        annual LST composite; scale_m should sit near MODIS' native 1 km. Returns a dict
        with delta_t_uhi_celsius, the mean/background LSTs, the uhi_derate_factor
        (1 + gamma*delta_T), the gamma used, and a source tag.
        """
        year = int(start_date[:4])
        lst = cls._lst_celsius(aoi, year)

        # 30 km focal mean as rural/background reference
        background = lst.focal_mean(
            radius=cls.BACKGROUND_KERNEL_PX,
            kernelType="circle",
            units="pixels",
        )
        uhi_anomaly = lst.subtract(background).rename("uhi_anomaly")

        # Single reduceRegion call for both bands
        combined = lst.addBands(uhi_anomaly)
        raw = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=scale_m,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo() or {}

        urban_lst = raw.get("LST_celsius")
        delta_t = raw.get("uhi_anomaly")
        source = "reduceRegion"

        if delta_t is None:
            delta_t = 0.0
            source = "fallback_zero"
        if urban_lst is None:
            urban_lst = 35.0  # representative Indian urban daytime T (degC)

        delta_t = float(delta_t)
        urban_lst = float(urban_lst)
        background_lst = urban_lst - delta_t
        derate = 1.0 + temp_coeff * delta_t

        return {
            "delta_t_uhi_celsius": round(delta_t, 3),
            "mean_lst_day_celsius": round(urban_lst, 2),
            "background_lst_celsius": round(background_lst, 2),
            "uhi_derate_factor": round(derate, 5),
            "temp_coeff_per_c": temp_coeff,
            "source": source,
            "modis_collection": cls.MODIS_COLLECTION,
            "background_kernel_km": cls.BACKGROUND_KERNEL_PX,
            "accounting_year": year,
            "scale_m": scale_m,
        }


# ---------------------------------------------------------------------------
# Class 3 – SoilingPenalty
# ---------------------------------------------------------------------------

class SoilingPenalty:
    """
    Dust/soiling derate driven by MODIS MAIAC aerosol optical depth (AOD, 550 nm).

    The idea: airborne aerosols (dust, soot, sulphate/nitrate) settle on the cover glass
    and cut its transmittance, and how fast they deposit tracks the aerosol column above.
    So we take annual loss = mean AOD * 0.08 per AOD unit (Kimber et al. 2006; Sayyah
    et al. 2014; Mani & Pillai 2010 for South Asian dust). No clamp -- the loss just
    follows the AOD, e.g. ~4% at a typical urban 0.5, ~8% at a Delhi-winter 1.0.

    Why AOD and not a bare-soil index like DBSI: DBSI measures the dust *source*, two steps
    removed from what lands on a panel, and turning it into a loss % needs an arbitrary
    fudge range. AOD is the actual atmospheric loading that drives dry deposition, and the
    0.08 coefficient maps to a real physical process. It's also genuinely urban -- city AOD
    runs 3-5x its rural surroundings, which is the whole point of the project.

    Source: MODIS MAIAC MCD19A2 v061 (Lyapustin et al. 2011), 1 km daily -- MAIAC is the
    one that actually works over bright urban surfaces where dark-target fails. We take the
    annual mean of valid retrievals (GEE's QA masking drops the cloudy days).
    """

    MAIAC_COLLECTION = "MODIS/061/MCD19A2_GRANULES"
    AOD_BAND = "Optical_Depth_055"   # 550 nm standard reference; scale factor 0.001
    AOD_SCALE = 0.001
    SOILING_COEFFICIENT = 0.08       # fractional loss per unit mean AOD per year
                                     # (Kimber et al. 2006; Sayyah et al. 2014)

    @classmethod
    def aod_image(cls, aoi: ee.Geometry, year: int) -> ee.Image:
        """Annual mean AOD at 550 nm for the given calendar year. Band: AOD_550nm."""
        return (
            ee.ImageCollection(cls.MAIAC_COLLECTION)
            .filterBounds(aoi)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select(cls.AOD_BAND)
            .mean()
            .multiply(cls.AOD_SCALE)
            .rename("AOD_550nm")
        )

    @classmethod
    def stats(
        cls,
        aoi: ee.Geometry,
        start_date: str,
        soiling_coefficient: float = SOILING_COEFFICIENT,
        scale_m: float = 1000.0,
    ) -> Dict[str, Any]:
        """
        Soiling retention from MAIAC AOD. Year comes from start_date; scale_m near MAIAC's
        1 km. Returns the mean AOD, the loss fraction (mean_AOD * coefficient, uncapped),
        the retention factor (1 - loss), the coefficient, and a source tag.
        """
        year = int(start_date[:4])
        aod_img = cls.aod_image(aoi, year)

        mean_aod = _reduce_mean(aod_img, "AOD_550nm", aoi, scale_m)
        source = "reduceRegion"

        if mean_aod is None:
            # No valid MAIAC retrievals for this AOI/year (very unlikely for India)
            mean_aod = 0.50   # conservative urban India annual mean
            source = "fallback_urban_midpoint"

        mean_aod = float(mean_aod)
        loss = mean_aod * soiling_coefficient
        retention = 1.0 - loss

        return {
            "mean_aod_550nm": round(mean_aod, 4),
            "soiling_loss_fraction": round(loss, 4),
            "soiling_retention_factor": round(retention, 5),
            "soiling_coefficient": soiling_coefficient,
            "source": source,
            "maiac_collection": cls.MAIAC_COLLECTION,
            "accounting_year": year,
            "scale_m": scale_m,
        }


# ---------------------------------------------------------------------------
# Combined net-irradiance builder (used by /api/yield and /api/tiles)
# ---------------------------------------------------------------------------

def net_irradiance_image(
    baseline_kwh_m2_period: float,
    shadow_frequency: ee.Image,
    beam_fraction: float = 1.0,
    uhi_derate: float = 1.0,
    soiling_retention: float = 1.0,
    sky_view_factor: Optional[Any] = None,
) -> ee.Image:
    """
    Stitches the penalty layers into one per-pixel net-irradiance image:

      net = GHI * [ diffuse_frac * SVF + beam_frac * (1 - shadow_freq) ] * uhi * soiling

    The bracket is the whole point: split GHI into diffuse + beam, then knock the beam
    down by shadows and the diffuse down by the sky-view factor. Pass sky_view_factor
    as an image or a scalar; leave it None for an open sky (SVF=1), which reduces to the
    old beam-only form GHI * (1 - shadow_freq * beam_frac) -- handy for the shadow-only
    stage. uhi/soiling are plain scalars from their stats() calls. Band out:
    'net_irradiance_kwh_m2_period'.
    """
    diffuse_fraction = 1.0 - beam_fraction

    if sky_view_factor is None:
        svf = ee.Image(1.0)
    elif isinstance(sky_view_factor, (int, float)):
        svf = ee.Image(float(sky_view_factor))
    else:
        svf = sky_view_factor

    beam_retention = ee.Image(1.0).subtract(shadow_frequency).multiply(beam_fraction)
    diffuse_retention = svf.multiply(diffuse_fraction)
    corrected_retention = diffuse_retention.add(beam_retention)

    effective_baseline = baseline_kwh_m2_period * uhi_derate * soiling_retention
    return corrected_retention.multiply(effective_baseline).rename("net_irradiance_kwh_m2_period")
