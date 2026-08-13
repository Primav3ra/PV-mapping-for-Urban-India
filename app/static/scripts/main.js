import { fetchPresets, fetchTiles, fetchYield, fetchBuildings, fetchSeries } from './api.js';
import { createMapController } from './map.js';

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const BASE_CONFIG = {
  half_size_deg: 0.01,
  roof_year: 2022,
  presence_threshold: 0.5,
  min_height_m: 0,
  panel_efficiency: 0.18,
  performance_ratio: 0.8,
  packing_factor: 0.7,
  building_confidence: 0.7,
};

// India grid CO2 per kWh — CEA baseline DB, FY23-24 (0.710 is the provisional FY24-25 number).
const GRID_EMISSION_FACTOR = 0.727;

const dom = {
  toast: document.getElementById('toast'),
  clock: document.getElementById('clock'),
  startMonth: document.getElementById('startMonth'),
  startYear: document.getElementById('startYear'),
  endMonth: document.getElementById('endMonth'),
  endYear: document.getElementById('endYear'),
  aoiLat: document.getElementById('aoiLat'),
  aoiLon: document.getElementById('aoiLon'),
  aoiArea: document.getElementById('aoiArea'),
  status: document.getElementById('status'),
  runComputeBtn: document.getElementById('runComputeBtn'),
  totalEnergy: document.getElementById('totalEnergy'),
  dailyAvg: document.getElementById('dailyAvg'),
  peakPower: document.getElementById('peakPower'),
  co2Avoided: document.getElementById('co2Avoided'),
  seriesLabel: document.getElementById('seriesLabel'),
  generationChart: document.getElementById('generationChart'),
  pvBaseline: document.getElementById('pvBaseline'),
  pvAfterShadow: document.getElementById('pvAfterShadow'),
  pvAfterSvf: document.getElementById('pvAfterSvf'),
  pvAfterUhi: document.getElementById('pvAfterUhi'),
  pvAfterSoiling: document.getElementById('pvAfterSoiling'),
  pvNet: document.getElementById('pvNet'),
  penaltyChart: document.getElementById('penaltyChart'),
  penaltyLossTotal: document.getElementById('penaltyLossTotal'),
  roofAreaM2: document.getElementById('roofAreaM2'),
  roofShadeAreaM2: document.getElementById('roofShadeAreaM2'),
  roofShadePct: document.getElementById('roofShadePct'),
  shadeMatrix: document.getElementById('shadeMatrix'),
  skipIntroFallback: document.getElementById('skipIntroFallback'),
  skipIntro: document.getElementById('skipIntro'),
  btnStreet: document.getElementById('btnStreet'),
  btnSatellite: document.getElementById('btnSatellite'),
  btnHybrid: document.getElementById('btnHybrid'),
  btnTemperature: document.getElementById('btnTemperature'),
  introOverlay: document.getElementById('introOverlay'),
};

const state = {
  lat: null,
  lon: null,
  half_size_deg: BASE_CONFIG.half_size_deg,
  chart: null,
  penaltyChart: null,
  lastYield: null,
  tileCache: {},
  currentOverlayId: null,
  savedCalcs: [],
  activeSavedId: null,
};

let mapCtrl = null;

function showToast(text, timeout = 3500) {
  dom.toast.textContent = text;
  dom.toast.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => dom.toast.classList.remove('show'), timeout);
}

function setStatus(text) {
  dom.status.textContent = text;
}

function fmtNum(value, digits = 0) {
  if (value == null || Number.isNaN(value)) return '-';
  return Number(value).toFixed(digits).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function pointInRing(lon, lat, ring) {
  // Ray casting for GeoJSON rings (coordinates are [lon, lat]).
  let inside = false;
  if (!Array.isArray(ring) || ring.length < 3) return false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i += 1) {
    const xi = ring[i][0];
    const yi = ring[i][1];
    const xj = ring[j][0];
    const yj = ring[j][1];

    const intersects = (yi > lat) !== (yj > lat) &&
      lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-12) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function firstRingFromGeoGeometry(geom) {
  if (!geom || !geom.type) return null;
  if (geom.type === 'Polygon') return geom.coordinates?.[0] ?? null;
  if (geom.type === 'MultiPolygon') return geom.coordinates?.[0]?.[0] ?? null;
  return null;
}

function daysInRange(startDate, endDateExclusive) {
  const s = new Date(startDate);
  const e = new Date(endDateExclusive);
  return Math.max(1, Math.round((e - s) / (1000 * 60 * 60 * 24)));
}

function inferWindow() {
  const sm = Number(dom.startMonth.value) + 1;
  const sy = Number(dom.startYear.value);
  const em = Number(dom.endMonth.value) + 1;
  const ey = Number(dom.endYear.value);

  if (sy === ey && sm === 1 && em === 12) {
    return { baseline_mode: 'yearly', year: sy, label: 'kWh / month', kind: 'yearly' };
  }
  if (sy === ey && em - sm === 2 && [1, 4, 7, 10].includes(sm)) {
    return { baseline_mode: 'quarterly', year: sy, quarter: Math.floor((sm - 1) / 3) + 1, label: 'kWh / month', kind: 'quarterly' };
  }
  if (sy === ey && sm === em) {
    return { baseline_mode: 'monthly', year: sy, month: sm, label: 'kWh / week', kind: 'monthly' };
  }

  showToast('For exact backend windows, use same month (monthly), quarter, or Jan-Dec (yearly). Falling back to start month.');
  return { baseline_mode: 'monthly', year: sy, month: sm, label: 'kWh / week', kind: 'monthly' };
}

function computePvStages(yieldData) {
  // Trust the backend's stage numbers when they're there; the fallback below is only
  // for old/partial responses.
  const backendBaseline = Number(yieldData?.baseline_yield_kwh);
  const backendAfterShadow = Number(yieldData?.after_shadow_yield_kwh);
  const backendAfterSvf = Number(yieldData?.after_svf_yield_kwh);
  const backendAfterUhi = Number(yieldData?.after_uhi_yield_kwh);
  const backendAfterSoiling = Number(yieldData?.after_soiling_yield_kwh);
  const backendNet = Number(yieldData?.period_yield_kwh);

  const hasBackendStages = [backendBaseline, backendAfterShadow, backendAfterSvf, backendAfterUhi, backendAfterSoiling, backendNet]
    .every((v) => Number.isFinite(v));

  const baselinePv = hasBackendStages ? backendBaseline : 0;

  const meanShadowRetention = Number(yieldData?.mean_shadow_retention ?? 1);
  const meanSkyViewFactor = Number(yieldData?.mean_sky_view_factor ?? 1);
  const diffuseFraction = Number(yieldData?.diffuse_fraction ?? 0);
  const uhiDerateFactor = Number(yieldData?.uhi_derate_factor ?? 1);
  const soilingRetentionFactor = Number(yieldData?.soiling_retention_factor ?? 1);

  const afterShadowPv = hasBackendStages ? backendAfterShadow : baselinePv * meanShadowRetention;
  // sky-view fallback: diffuse loss is diffuse_fraction * (1 - SVF) off the shadow stage
  const afterSvfPv = hasBackendStages
    ? backendAfterSvf
    : afterShadowPv * (1 - diffuseFraction * (1 - meanSkyViewFactor));
  const afterUhiPv = hasBackendStages ? backendAfterUhi : afterSvfPv * uhiDerateFactor;
  const afterSoilingPv = hasBackendStages ? backendAfterSoiling : afterUhiPv * soilingRetentionFactor;

  const netPv = hasBackendStages ? backendNet : Number(yieldData?.period_yield_kwh ?? afterSoilingPv);

  const totalLossPv = Math.max(0, baselinePv - netPv);
  const shadowLossPv = Math.max(0, baselinePv - afterShadowPv);
  const svfLossPv = Math.max(0, afterShadowPv - afterSvfPv);
  const uhiLossPv = Math.max(0, afterSvfPv - afterUhiPv);
  const soilingLossPv = Math.max(0, afterUhiPv - afterSoilingPv);

  const pctOfLoss = (pv) => (totalLossPv > 0 ? (pv / totalLossPv) * 100 : 0);

  // %s come straight off the stage yields so the number next to each row is literally
  // the drop in the kWh you see. (The old shadow % used a mean-retention figure that
  // didn't line up with its own kWh — confusing, so gone.)
  const shadowPenaltyPercent = baselinePv > 0 ? (1 - afterShadowPv / baselinePv) * 100 : 0;
  const svfPenaltyPercent = afterShadowPv > 0 ? (1 - afterSvfPv / afterShadowPv) * 100 : 0;
  const uhiPenaltyPercent = afterSvfPv > 0 ? (1 - afterUhiPv / afterSvfPv) * 100 : 0;
  const soilingPenaltyPercent = afterUhiPv > 0 ? (1 - afterSoilingPv / afterUhiPv) * 100 : 0;

  return {
    baselinePv,
    afterShadowPv,
    afterSvfPv,
    afterUhiPv,
    afterSoilingPv,
    netPv,
    totalLossPv,
    totalLossPct: baselinePv > 0 ? (totalLossPv / baselinePv) * 100 : 0,
    shadowContributionPct: pctOfLoss(shadowLossPv),
    svfContributionPct: pctOfLoss(svfLossPv),
    uhiContributionPct: pctOfLoss(uhiLossPv),
    soilingContributionPct: pctOfLoss(soilingLossPv),
    shadowPenaltyPercent,
    svfPenaltyPercent,
    uhiPenaltyPercent,
    soilingPenaltyPercent,
  };
}

function ensurePenaltyChart() {
  if (state.penaltyChart) return state.penaltyChart;
  const ctxCanvas = dom.penaltyChart;
  if (!ctxCanvas) return null;

  state.penaltyChart = new Chart(ctxCanvas, {
    type: 'bar',
    data: {
      labels: ['Shadow', 'Sky View', 'Urban Heat', 'Soiling'],
      datasets: [
        {
          data: [0, 0, 0, 0],
          backgroundColor: ['#f97316', '#38bdf8', '#22c55e', '#ffb347'],
          borderColor: ['#f97316', '#38bdf8', '#22c55e', '#ffb347'],
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#9ab0d4', font: { size: 10 } }, grid: { display: false } },
        y: {
          min: 0,
          max: 100,
          ticks: {
            color: '#9ab0d4',
            font: { size: 10 },
            callback: (v) => `${v}%`,
          },
          grid: { color: 'rgba(90,118,170,0.25)' },
        },
      },
    },
  });

  return state.penaltyChart;
}

function renderResultBox(yieldData) {
  const stages = computePvStages(yieldData);
  dom.pvBaseline.textContent = fmtNum(stages.baselinePv, 0);
  dom.pvAfterShadow.textContent =
    stages.baselinePv > 0 ? `${fmtNum(stages.afterShadowPv, 0)} (-${stages.shadowPenaltyPercent.toFixed(1)}%)` : '-';
  if (dom.pvAfterSvf) dom.pvAfterSvf.textContent =
    stages.afterShadowPv > 0 ? `${fmtNum(stages.afterSvfPv, 0)} (-${stages.svfPenaltyPercent.toFixed(1)}%)` : '-';
  dom.pvAfterUhi.textContent =
    stages.afterSvfPv > 0 ? `${fmtNum(stages.afterUhiPv, 0)} (-${stages.uhiPenaltyPercent.toFixed(1)}%)` : '-';
  dom.pvAfterSoiling.textContent =
    stages.afterUhiPv > 0 ? `${fmtNum(stages.afterSoilingPv, 0)} (-${stages.soilingPenaltyPercent.toFixed(1)}%)` : '-';
  dom.pvNet.textContent =
    stages.baselinePv > 0 ? `${fmtNum(stages.netPv, 0)} (-${stages.totalLossPct.toFixed(1)}%)` : '-';

  const backendLossKwh = Number(yieldData?.penalty_loss_kwh);
  const backendLossPct = Number(yieldData?.penalty_loss_pct);
  const hasBackendLoss = Number.isFinite(backendLossKwh) && Number.isFinite(backendLossPct);

  const lossLabel = hasBackendLoss
    ? `${fmtNum(backendLossKwh, 0)} kWh loss (${backendLossPct.toFixed(1)}%)`
    : (stages.baselinePv > 0 ? `${fmtNum(stages.totalLossPv, 0)} kWh loss (${stages.totalLossPct.toFixed(1)}%)` : '-');
  if (dom.penaltyLossTotal) dom.penaltyLossTotal.textContent = lossLabel;

  const chart = ensurePenaltyChart();
  if (!chart) return;

  const pc = yieldData?.penalty_contribution || {};
  const sPct = Number(pc.shadow_contribution_pct);
  const svPct = Number(pc.svf_contribution_pct);
  const uPct = Number(pc.uhi_contribution_pct);
  const soPct = Number(pc.soiling_contribution_pct);
  const hasBackendContribution = [sPct, svPct, uPct, soPct].every((v) => Number.isFinite(v));

  chart.data.datasets[0].data = hasBackendContribution
    ? [sPct, svPct, uPct, soPct]
    : [stages.shadowContributionPct, stages.svfContributionPct, stages.uhiContributionPct, stages.soilingContributionPct];
  chart.update();
}

function renderKpis(yieldData, temporalWindow) {
  const potential = Number(yieldData.period_yield_kwh ?? 0);
  const days = daysInRange(temporalWindow.start_date, temporalWindow.end_date_exclusive);
  const dailyAvg = potential / days;

  // Estimated DC nameplate: usable roof * packing * module efficiency (at 1 kW/m^2 STC).
  // This is the actual size the yield is built on -- the old "energy/(days*5h)" number
  // was neither a peak nor a capacity, so it's gone.
  const roofArea = Number(yieldData.roof_area_m2 ?? 0);
  const packing = Number(yieldData.packing_factor ?? BASE_CONFIG.packing_factor);
  const panelEff = Number(yieldData.panel_efficiency ?? BASE_CONFIG.panel_efficiency);
  const capacityKwp = roofArea * packing * panelEff;
  const co2Kg = potential * GRID_EMISSION_FACTOR;

  // "TOTAL ENERGY" should match the PV output (net) shown elsewhere.
  dom.totalEnergy.textContent = fmtNum(potential, 0);
  dom.dailyAvg.textContent = fmtNum(dailyAvg, 1);
  dom.peakPower.textContent = fmtNum(capacityKwp, 2);
  dom.co2Avoided.textContent = fmtNum(co2Kg, 0);
}

function renderRooftopAnalysis(yieldData) {
  const roofArea = Number(yieldData?.roof_area_m2 ?? 0);
  const shadeFraction = Number(yieldData?.mean_shadow_fraction ?? 0);
  const shadeArea = roofArea * shadeFraction;
  const shadePct = shadeFraction * 100;

  if (dom.roofAreaM2) dom.roofAreaM2.textContent = fmtNum(roofArea, 0);
  if (dom.roofShadeAreaM2) dom.roofShadeAreaM2.textContent = fmtNum(shadeArea, 0);
  if (dom.roofShadePct) dom.roofShadePct.textContent = `${shadePct.toFixed(1)}`;
}

function renderShadeMatrix(yieldData) {
  if (!dom.shadeMatrix) return;
  const intervals = yieldData?.shade_intervals || [];
  if (!intervals.length) {
    dom.shadeMatrix.innerHTML = '<div class="muted">No shade data.</div>';
    return;
  }

  dom.shadeMatrix.innerHTML = intervals
    .map(
      (it) => `
        <div class="shade-row">
          <span>${it.label ?? '-'}</span>
          <div style="text-align:right">
            <b>${it.shade_percent != null ? Number(it.shade_percent).toFixed(1) : '-' }%</b>
            <small>${it.shade_area_m2 != null ? fmtNum(it.shade_area_m2, 0) : '-' } m2</small>
          </div>
        </div>
      `,
    )
    .join('');
}

function setChart(labels, values, label) {
  dom.seriesLabel.textContent = label;
  if (state.chart) {
    state.chart.data.labels = labels;
    state.chart.data.datasets[0].data = values;
    state.chart.update();
    return;
  }
  state.chart = new Chart(dom.generationChart, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: '#ffb347',
        backgroundColor: 'rgba(255,179,71,0.16)',
        fill: true,
        tension: 0.35,
        pointRadius: 2.5,
      }],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: true },
      scales: {
        x: { ticks: { color: '#9ab0d4', font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { color: '#9ab0d4', font: { size: 10 } }, grid: { color: 'rgba(90,118,170,0.25)' } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(10,16,30,0.95)',
          borderColor: 'rgba(255,179,71,0.45)',
          borderWidth: 1,
          titleColor: '#d9e7ff',
          bodyColor: '#d9e7ff',
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed?.y;
              return ` ${fmtNum(v, 0)} kWh`;
            },
          },
        },
      },
      onClick: (_evt, elements) => {
        const el = elements?.[0];
        if (!el) return;
        const idx = el.index;
        const x = state.chart?.data?.labels?.[idx];
        const y = state.chart?.data?.datasets?.[0]?.data?.[idx];
        if (x == null || y == null) return;
        showToast(`${x}: ${fmtNum(y, 0)} kWh`, 2200);
      },
    },
  });
}

function renderSavedMarkers() {
  if (!mapCtrl) return;
  const points = state.savedCalcs.map((c, idx) => ({
    id: c.id,
    lat: c.lat,
    lon: c.lon,
    label: String(idx + 1),
  }));
  mapCtrl.renderSavedPoints(points, state.activeSavedId);
}

function saveCalculation({ yieldData, temporalWindow, temporal, series }) {
  const id = String(Date.now());
  const entry = {
    id,
    lat: state.lat,
    lon: state.lon,
    half_size_deg: state.half_size_deg,
    createdAt: new Date().toISOString(),
    temporal,
    temporalWindow,
    yieldData,
    series,
  };
  state.savedCalcs.push(entry);
  if (state.savedCalcs.length > 5) state.savedCalcs = state.savedCalcs.slice(-5);
  state.activeSavedId = id;
  renderSavedMarkers();
}

function loadSavedCalculation(id) {
  const entry = state.savedCalcs.find((e) => String(e.id) === String(id));
  if (!entry) return;
  state.activeSavedId = String(id);
  state.lat = entry.lat;
  state.lon = entry.lon;
  state.half_size_deg = entry.half_size_deg ?? BASE_CONFIG.half_size_deg;

  dom.aoiLat.textContent = state.lat.toFixed(6);
  dom.aoiLon.textContent = state.lon.toFixed(6);
  mapCtrl?.drawAOI(state.lat, state.lon, state.half_size_deg);

  renderResultBox(entry.yieldData);
  renderRooftopAnalysis(entry.yieldData);
  renderShadeMatrix(entry.yieldData);
  renderKpis(entry.yieldData, entry.temporalWindow ?? {
    start_date: entry.yieldData?.start_date,
    end_date_exclusive: entry.yieldData?.end_date_exclusive,
  });
  if (mapCtrl) mapCtrl.renderBuildingLayer(entry.yieldData.geojson);
  if (entry.series) setChart(entry.series.labels, entry.series.values, entry.temporal?.label ?? 'kWh');

  renderSavedMarkers();
  setStatus(`Loaded saved result #${state.savedCalcs.findIndex((e) => e.id === id) + 1} for comparison.`);
}

async function showOverlayForLayer(layer) {
  if (!mapCtrl) return;
  if (state.lat == null || state.lon == null) return;

  // Keep map overlays minimal: only Temperature is exposed in the UI.
  if (layer !== 'temperature_delta') return;

  const temporal = state.lastCompute?.temporal ?? inferWindow();
  const payloadBase =
    state.lastCompute?.payloadBase ??
    {
      lat: state.lat,
      lon: state.lon,
      half_size_deg: state.half_size_deg ?? BASE_CONFIG.half_size_deg,
      roof_year: BASE_CONFIG.roof_year,
      presence_threshold: BASE_CONFIG.presence_threshold,
      min_height_m: BASE_CONFIG.min_height_m,
      panel_efficiency: BASE_CONFIG.panel_efficiency,
      performance_ratio: BASE_CONFIG.performance_ratio,
      packing_factor: BASE_CONFIG.packing_factor,
      building_confidence: BASE_CONFIG.building_confidence,
    };

  const overlayButtonsByOverlayId = {
    temperature: dom.btnTemperature,
  };

  const cfg = { overlayId: 'temperature', opacity: 0.55 };

  // Toggle off if the same overlay is selected.
  if (state.currentOverlayId === cfg.overlayId) {
    mapCtrl.removeRasterOverlay(cfg.overlayId);
    state.currentOverlayId = null;
    Object.values(overlayButtonsByOverlayId).forEach((b) => b?.classList.remove('active'));
    return;
  }

  // Remove previous overlay so the map stays readable.
  if (state.currentOverlayId) {
    mapCtrl.removeRasterOverlay(state.currentOverlayId);
    state.currentOverlayId = null;
  }
  Object.values(overlayButtonsByOverlayId).forEach((b) => b?.classList.remove('active'));

  const res = await fetchTiles({ ...payloadBase, ...temporal, layer });
  if (res.status === 'ok') {
    mapCtrl.addRasterOverlay(cfg.overlayId, res.urlTemplate, cfg.opacity, 'aoi-line');
    state.currentOverlayId = cfg.overlayId;
    overlayButtonsByOverlayId[cfg.overlayId]?.classList.add('active');
  } else {
    showToast(res.message || 'Failed to fetch overlay tiles.');
  }
}

async function runCompute() {
  if (state.lat == null || state.lon == null) {
    showToast('Select rooftop location on map first.');
    return;
  }
  const temporal = inferWindow();
  state.lastCompute = { payloadBase: null, temporal };
  const payloadBase = {
    lat: state.lat,
    lon: state.lon,
    half_size_deg: state.half_size_deg ?? BASE_CONFIG.half_size_deg,
    roof_year: BASE_CONFIG.roof_year,
    presence_threshold: BASE_CONFIG.presence_threshold,
    min_height_m: BASE_CONFIG.min_height_m,
    panel_efficiency: BASE_CONFIG.panel_efficiency,
    performance_ratio: BASE_CONFIG.performance_ratio,
    packing_factor: BASE_CONFIG.packing_factor,
    building_confidence: BASE_CONFIG.building_confidence,
  };
  state.lastCompute.payloadBase = payloadBase;

  try {
    dom.runComputeBtn.disabled = true;
    setStatus('Running backend calculations...');
    const yieldData = await fetchYield({ ...payloadBase, ...temporal });
    if (yieldData.status !== 'ok') {
      setStatus(yieldData.message || 'No rooftop found at selected point.');
      return;
    }
    state.lastYield = yieldData;
    // dates come off the yield response now -- no separate /api/baseline call
    const temporalWindow = {
      start_date: yieldData.start_date,
      end_date_exclusive: yieldData.end_date_exclusive,
    };
    renderResultBox(yieldData);
    renderRooftopAnalysis(yieldData);
    renderShadeMatrix(yieldData);
    renderKpis(yieldData, temporalWindow);
    if (mapCtrl) {
      mapCtrl.renderBuildingLayer(yieldData.geojson);
    }

    if (yieldData.selection_warning) {
      showToast(yieldData.selection_warning);
    }

    if (dom.btnTemperature?.classList.contains('active')) {
      await showOverlayForLayer('temperature_delta');
    }

    setStatus('Computing trend curve...');
    // one call for the whole curve now, not one /api/yield per point
    const series = await fetchSeries({ ...payloadBase, ...temporal });
    const hasSeries = series && series.status === 'ok'
      && Array.isArray(series.values) && series.values.length > 0;
    if (hasSeries) {
      setChart(series.labels, series.values, temporal.label);
      setStatus('Done. Dashboard updated.');
    } else {
      setChart([], [], temporal.label);
      setStatus('Dashboard updated (trend curve unavailable).');
      showToast(`Trend curve failed: ${series?.detail || series?.message || 'unknown error'}`);
    }

    // Keep last 5 computations for quick comparison.
    saveCalculation({ yieldData, temporalWindow, temporal, series: hasSeries ? series : null });
  } catch (e) {
    setStatus(`Failed: ${e.message}`);
    showToast('Computation failed. Check backend/server logs.');
  } finally {
    dom.runComputeBtn.disabled = false;
  }
}

function initYearMonthSelectors() {
  for (let m = 0; m < 12; m += 1) {
    dom.startMonth.add(new Option(MONTHS[m], String(m)));
    dom.endMonth.add(new Option(MONTHS[m], String(m)));
  }
  const thisYear = new Date().getFullYear();
  for (let y = 2022; y <= thisYear; y += 1) {
    dom.startYear.add(new Option(String(y), String(y)));
    dom.endYear.add(new Option(String(y), String(y)));
  }
  dom.startMonth.value = '0';
  dom.endMonth.value = '11';
  dom.startYear.value = String(thisYear - 1);
  dom.endYear.value = String(thisYear - 1);
}

function initClock() {
  const tick = () => {
    dom.clock.textContent = new Date().toLocaleString();
  };
  tick();
  setInterval(tick, 1000);
}

function initIntro() {
  const canvas = dom.introCanvas;
  const ctx = canvas?.getContext('2d');
  if (!canvas || !ctx || !dom.introOverlay || !dom.skipIntro) return;
  let done = false;
  let raf = 0;
  const introStart = performance.now();
  const MAX_PCT = 99;
  const STEP_MS = 42;
  let lastPct = 1;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  function draw(t) {
    if (done) return;
    const elapsed = performance.now() - introStart;
    const nextPct = Math.min(MAX_PCT, 1 + Math.floor(elapsed / STEP_MS));
    if (nextPct !== lastPct && dom.introLoader) {
      lastPct = nextPct;
      dom.introLoader.textContent = `${lastPct}%`;
    }
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(3,6,13,1)';
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 300; i += 1) {
      const x = (i * 53 + t * 0.05) % (w + 40) - 20;
      const y = (i * 97) % h;
      const a = 0.15 + ((i % 9) / 10);
      ctx.fillStyle = `rgba(255,190,110,${a * 0.3})`;
      ctx.fillRect(x, y, 1.2, 1.2);
    }
    const r = 58 + Math.sin(t * 0.0012) * 8;
    const gx = w * 0.5;
    const gy = h * 0.5;
    const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, r * 3);
    grad.addColorStop(0, 'rgba(255,215,150,0.9)');
    grad.addColorStop(0.3, 'rgba(255,145,40,0.45)');
    grad.addColorStop(1, 'rgba(255,100,20,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(gx, gy, r * 3, 0, Math.PI * 2);
    ctx.fill();
    raf = requestAnimationFrame(draw);
  }
  raf = requestAnimationFrame(draw);

  function finishIntro() {
    if (done) return;
    done = true;
    cancelAnimationFrame(raf);
    try {
      if (dom.introLoader) dom.introLoader.textContent = '100%';
      dom.introOverlay.classList.add('done');
      setTimeout(() => dom.introOverlay.remove(), 520);
    } catch {
      // keep UI unblocked even if DOM cleanup fails
    }
  }
  dom.skipIntro.addEventListener('click', finishIntro, { once: true });
  setTimeout(finishIntro, 4200);
}

async function initPresets() {
  try {
    const presets = await fetchPresets();
    const y = String(presets?.baseline?.year_bounds?.default || Number(dom.startYear.value));
    dom.startYear.value = y;
    dom.endYear.value = y;
  } catch {
    // optional presets
  }
}

function bindEvents() {
  dom.runComputeBtn.addEventListener('click', runCompute);
  dom.btnStreet.addEventListener('click', () => mapCtrl?.setBasemap('street'));
  dom.btnSatellite.addEventListener('click', () => mapCtrl?.setBasemap('satellite'));
  dom.btnHybrid.addEventListener('click', () => mapCtrl?.setBasemap('hybrid'));

  dom.btnTemperature?.addEventListener('click', () => showOverlayForLayer('temperature_delta'));

  if (mapCtrl) {
    mapCtrl.onMapClick(async (event) => {
      const clickedLat = Number(event.latlng.lat);
      const clickedLon = Number(event.latlng.lng);

      state.lat = clickedLat;
      state.lon = clickedLon;
      dom.aoiLat.textContent = state.lat.toFixed(6);
      dom.aoiLon.textContent = state.lon.toFixed(6);

      const mPerDegLat = 111320;
      const mPerDegLon = 111320 * Math.cos((state.lat * Math.PI) / 180);

      const PICK_HALF_DEG = 0.003; // small search window around click
      const MIN_HALF_DEG = 0.0015;
      const MAX_HALF_DEG = BASE_CONFIG.half_size_deg;

      let nextHalf = state.half_size_deg ?? BASE_CONFIG.half_size_deg;
      let pickedBuildingAreaM2 = null;

      try {
        const buildingsRes = await fetchBuildings({
          lat: state.lat,
          lon: state.lon,
          half_size_deg: PICK_HALF_DEG,
          building_confidence: BASE_CONFIG.building_confidence,
          limit: 400,
        });

        const features = buildingsRes?.geojson?.features || [];
        for (const f of features) {
          const ring = firstRingFromGeoGeometry(f?.geometry);
          if (!ring) continue;
          if (!pointInRing(state.lon, state.lat, ring)) continue;

          const areaM2 = Number(f?.properties?.area_in_meters ?? 0);
          if (areaM2 > 0) {
            pickedBuildingAreaM2 = areaM2;
            const radiusM = Math.sqrt(areaM2 / Math.PI);
            // AOI is drawn as a square, so convert radius (m) to half-size in degrees (lat).
            const halfDeg = radiusM / mPerDegLat;
            nextHalf = Math.max(MIN_HALF_DEG, Math.min(MAX_HALF_DEG, halfDeg));
          }
          break;
        }
      } catch {
        // Fall back to previous half_size_deg if selection fails.
      }

      state.half_size_deg = nextHalf;
      const areaRectM2 =
        (state.half_size_deg * 2 * mPerDegLat) * (state.half_size_deg * 2 * mPerDegLon);
      dom.aoiArea.textContent = `${fmtNum(areaRectM2, 0)} m2`;
      mapCtrl.drawAOI(state.lat, state.lon, state.half_size_deg);

      setStatus(pickedBuildingAreaM2 ? 'Building selected. Click COMPUTE.' : 'Click registered. Click COMPUTE.');
    });

    mapCtrl.onSavedPointClick(({ id }) => {
      loadSavedCalculation(id);
    });
  } else {
    setStatus('Map failed to initialize. Refresh once; if issue persists, check internet for tile scripts.');
  }

  // Mapbox 3D token-gated mode removed from the UI to keep the map uncluttered.
}

function startDashboard() {
  if (startDashboard.started) return;
  startDashboard.started = true;

  initClock();
  try {
    mapCtrl = createMapController();
  } catch (error) {
    showToast(`Map init failed: ${error.message}`);
  }
  initYearMonthSelectors();
  initPresets();
  bindEvents();
}

function startIntroThenDashboard() {
  if (!dom.introOverlay) {
    startDashboard();
    return;
  }

  let done = false;
  let unmount = null;
  const finish = () => {
    if (done) return;
    done = true;
    try {
      unmount?.();
    } catch {
      // ignore
    }
    try {
      dom.introOverlay.classList.add('done');
      setTimeout(() => dom.introOverlay?.remove(), 520);
    } catch {
      // ignore
    }
    startDashboard();
  };

  // Safety valve: always unblocks the UI even if intro fails.
  const safety = setTimeout(finish, 7_000);

  // User-facing fallback: if the React intro fails, this still works.
  dom.skipIntroFallback?.addEventListener('click', () => finish(), { once: true });

  // Also listen for clicks on the React skip button (id="skipIntro") if it renders.
  try {
    dom.skipIntro?.addEventListener('click', () => finish(), { once: true });
  } catch {
    // ignore
  }

  // Extra robustness: delegate clicks by id so we don't depend on element presence timing.
  document.addEventListener(
    'click',
    (e) => {
      const el = e.target?.closest?.('#skipIntro,#skipIntroFallback,#skipIntroFallback');
      if (el) finish();
    },
    { capture: true },
  );

  try {
    const api = window.pvIntro;
    if (api?.startIntro) {
      unmount = api.startIntro({ mountId: 'intro-root', onComplete: finish });
      return;
    }
  } catch {
    // fall through to finish()
  }

  clearTimeout(safety);
  finish();
}

function start() {
  startIntroThenDashboard();
}

start();
