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
    Insolation-weighted 2.5D shadow model from Open Buildings height raster.

    For each solar position (alt, az) the geometric shadow length of a building of
    height H is:
      L = H / tan(alt)  [metres]  ->  L_px = L / pixel_size_m  [pixels]

    Shadow propagation (GEE-efficient approximation):
      1. shadow_length_px = building_height / (tan_alt * pixel_size_m)
      2. Translate shadow_length_px by MAX_SHADOW_PIXELS in the shadow direction.
      3. focal_max with same radius captures any caster in range.
      4. Pixel P is in shadow if dilated_len >= 1 AND caster height > P height
         (the second condition prevents self-shadowing).

    Solar positions are insolation-weighted (weight = sin(altitude)) so low-sun
    morning/winter positions, which cast long but energetically small shadows,
    contribute proportionally less to the annual penalty.

    Known approximations (targets for ML calibration):
      - focal_max kernel is circular, not directional -> ~5-10 % overestimate of
        shadow area.
      - Diffuse irradiance (~30-45 % of GHI in urban India) is NOT blocked by
        building shadows for rooftop pixels (Sky View Factor ~0.85-0.95).
        This is now corrected in net_irradiance_image() via beam_fraction from
        ERA5 HOURLY direct radiation, which removes the dominant ~30-40 % share
        of shadow loss that was being incorrectly applied to diffuse irradiance.
      - Rooftop SVF is assumed ~1.0 (diffuse fully received). In canyons between
        buildings SVF could be 0.2-0.4, but roof_candidate pixels are by
        definition at the top of buildings with open sky above.
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
    Planar (cosine-weighted) Sky View Factor from the 2.5D building-height raster.

    Where ShadowPenalty removes the DIRECT beam blocked by buildings, this layer
    removes the DIFFUSE light blocked by buildings: surrounding structures occlude
    part of the sky hemisphere, so a rooftop receives less than the full diffuse
    irradiance that an open horizontal surface would.

    For a horizontal rooftop pixel, the fraction of isotropic diffuse sky radiation
    that is still received:

        SVF = 1 - mean_over_azimuth( sin^2( horizon_elevation_angle ) )

        horizon_elevation_angle(azimuth) = max over distance d of
            atan( (H_neighbour(d) - H_self) / d )      [>= 0; only taller neighbours block]

    The sin^2 weighting is the standard planar-surface isotropic-diffuse form (Oke,
    *Boundary Layer Climates*): an obstruction rising to elevation beta in an azimuth
    sector blocks a fraction sin^2(beta) of the diffuse flux a horizontal surface
    would otherwise receive from that sector. Integrating over azimuth and normalising
    gives 1 - mean(sin^2(beta)).

    Rooftop pixels sit at (or near) the top of their own building, so self-height is
    subtracted and SVF is typically high (~0.85-1.0); only genuinely TALLER neighbours
    reduce it. Deep street canyons would be much lower, but roof-candidate pixels are
    by definition above the canyon.

    Combined with the beam/shadow term the net retention becomes:
        retention = diffuse_fraction * SVF + beam_fraction * (1 - shadow_frequency)
    """

    N_AZIMUTH: int = 8
    # Sample radii in pixels; * pixel_size_m = metres. Log-spaced: near obstructions
    # dominate the horizon angle, distant ones subtend a small angle and matter less.
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

            # Horizon elevation angle in this azimuth = max over sampled distances.
            angle_imgs: List[ee.Image] = []
            for d in dists:
                dm = float(d) * pixel_size_m
                neighbour = building_height.translate(ux * d, uy * d)
                rise = neighbour.subtract(building_height).max(0.0)   # taller neighbours only
                angle_imgs.append(rise.divide(dm).atan())             # radians, >= 0
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
    Temperature-coefficient derate caused by the Urban Heat Island effect.

    Physics:
      PV output decreases linearly above 25 degC (STC):
        dP/P = gamma * (T_cell - 25)
      where gamma ~ -0.004 /degC for crystalline Si (IEC 60891; De Soto 2006).

      Cell temperature:
        T_cell = T_ambient + (NOCT - 20) / 800 * G
      where NOCT ~ 45 degC, G is irradiance in W/m^2.

      The UHI contribution to T_ambient:
        delta_T_UHI = T_ambient_urban - T_ambient_rural_ref

      Simplified UHI-only derate (urban excess above background only):
        uhi_derate = 1 + gamma * delta_T_UHI

      Typical Indian city UHI: 2-6 degC (Mohan et al. 2011).
      At gamma = -0.004:
        delta_T = 2 degC -> derate = 0.992  (~0.8 % loss)
        delta_T = 4 degC -> derate = 0.984  (~1.6 % loss)
        delta_T = 6 degC -> derate = 0.976  (~2.4 % loss)

    UHI estimation (MODIS LST):
      1. Annual median of MODIS MOD11A2 daytime LST (1 km) over the AOI.
         Raw integer values * 0.02 = Kelvin; subtract 273.15 for degC.
      2. Compute 30 km focal_mean of the LST image as the regional background.
         (30 px at 1 km/px ~ 30 km; sized to reach beyond the built-up core of
          large polynuclear cities like Delhi/Mumbai, where a smaller kernel
          stays entirely inside the urban heat footprint and drives the
          urban-minus-background anomaly artificially toward zero.)
      3. delta_T_UHI = AOI mean LST - background mean at AOI location.

    Note: UHI is a quasi-static location property; we use the full calendar year
    of the accounting period for a robust seasonal composite.
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
        Compute UHI intensity and derate factor for the AOI.

        Parameters
        ----------
        aoi        : GEE geometry of the area under study.
        start_date : ISO date string; year is extracted for annual LST composite.
        temp_coeff : PV temperature coefficient /degC (default -0.004 for c-Si).
        scale_m    : reduceRegion scale; should match MODIS native ~1000 m.

        Returns dict keys:
          delta_t_uhi_celsius    -- UHI intensity above 20 km background (degC)
          mean_lst_day_celsius   -- mean daytime LST over AOI (degC)
          background_lst_celsius -- 20 km smoothed regional background (degC)
          uhi_derate_factor      -- scalar multiplier: 1 + gamma * delta_T
          temp_coeff_per_c       -- gamma value used
          source                 -- "reduceRegion" | "fallback_zero"
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
    Dust/soiling derate from MODIS MAIAC Aerosol Optical Depth (AOD) at 550 nm.

    Physics:
      Atmospheric aerosols (mineral dust, soot, secondary sulphate/nitrate)
      settle on the PV cover glass, reducing its transmittance.
      Dry deposition rate is proportional to the ambient aerosol column loading:

        soiling_loss_annual = mean_AOD_550nm * SOILING_COEFFICIENT

      SOILING_COEFFICIENT ~ 0.08 /AOD_unit/year  (Kimber et al. 2006;
      Sayyah et al. 2014; validated for South Asian aerosol types in
      Mani & Pillai 2010).

      No output clamp is applied. The AOD-driven loss is reported as computed:
        AOD = 0.15 (clean rural)   -> loss = 1.2 %
        AOD = 0.50 (typical urban) -> loss = 4.0 %
        AOD = 1.00 (Delhi winter)  -> loss = 8.0 %
        AOD = 1.50 (severe event)  -> loss = 12.0 %

    Why AOD instead of DBSI:
      DBSI (Sentinel-2 spectral index) measures bare soil exposure -- a dust
      SOURCE proxy two steps removed from actual panel soiling. The conversion
      from DBSI to loss % requires an arbitrary normalisation range that forces
      the output to match assumed literature values.

      MODIS MAIAC AOD directly measures the atmospheric aerosol column loading
      that causes soiling via dry deposition. The physics chain is:
        AOD -> aerosol surface concentration -> deposition flux -> loss
      The single coefficient (SOILING_COEFFICIENT) is physically motivated and
      its units correspond directly to a measurable deposition process.

      Urban India has AOD 3-5x higher than surrounding rural areas, making this
      a genuinely urban-specific penalty consistent with the project problem statement.

    Data source:
      MODIS MAIAC MCD19A2 v061 (Lyapustin et al. 2011), 1 km daily.
      MAIAC is specifically designed for urban and bright-surface retrievals where
      the standard MODIS dark-target algorithm fails.
      Annual mean of valid daily retrievals (cloudy days excluded automatically
      by MODIS QA masking in GEE).
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
        Compute soiling retention factor from MODIS MAIAC AOD.

        Parameters
        ----------
        aoi                 : GEE geometry of the area under study.
        start_date          : ISO date; year is extracted for the annual composite.
        soiling_coefficient : Fractional loss per unit mean AOD per year (default 0.08).
        scale_m             : reduceRegion scale; MAIAC native is ~1000 m.

        Returns dict keys:
          mean_aod_550nm          -- annual mean AOD at 550 nm over AOI
          soiling_loss_fraction   -- mean_AOD * soiling_coefficient  (uncapped)
          soiling_retention_factor-- 1 - loss
          soiling_coefficient     -- coefficient used
          source                  -- "reduceRegion" | "fallback_urban_midpoint"
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
    Per-pixel net irradiance after all penalty layers with beam/diffuse correction.

    Formula:
      net = GHI * [ diffuse_fraction * SVF + beam_fraction * (1 - shadow_frequency) ]
                * uhi_derate * soiling_retention

    Derivation:
      GHI = DHI + DNI_h  (diffuse + direct horizontal)
      Shadows block only the beam (DNI_h); the diffuse (DHI) reaches the rooftop from
      the visible sky, whose fraction is the Sky View Factor (SVF).
        net = DHI * SVF + DNI_h * (1 - shadow_frequency)
            = GHI * diffuse_fraction * SVF + GHI * beam_fraction * (1 - shadow_frequency)

    With SVF = 1 (open sky) this collapses to the previous beam-only correction
    GHI * (1 - shadow_frequency * beam_fraction), so callers that omit sky_view_factor
    are unaffected.

    Parameters
    ----------
    baseline_kwh_m2_period : float
        ERA5-Land GHI integrated over the accounting period (kWh/m^2).
    shadow_frequency : ee.Image
        Per-pixel insolation-weighted shadow frequency [0, 1]; band 'shadow_frequency'.
        From ShadowPenalty.frequency().
    beam_fraction : float
        Fraction of GHI that is direct beam, computed from ERA5 HOURLY
        total_sky_direct_solar_radiation_at_surface / surface_solar_radiation_downwards.
        Default 1.0 preserves old behaviour if caller does not pass it.
    uhi_derate : float
        Scalar from UHIPenalty.stats()['uhi_derate_factor'].
    soiling_retention : float
        Scalar from SoilingPenalty.stats()['soiling_retention_factor'].
    sky_view_factor : ee.Image | float | None
        Per-pixel Sky View Factor [0, 1] from SkyViewFactor.image(), a scalar, or
        None. None (or 1.0) means an unobstructed sky -> full diffuse received.

    Returns ee.Image band 'net_irradiance_kwh_m2_period'.
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
