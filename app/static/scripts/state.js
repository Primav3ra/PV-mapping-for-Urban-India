export const BUILDING_CAP = 7;

export function fmtEnergy(kwh, energyUnit, suffix = '/yr') {
  if (kwh == null) return '-';
  if (energyUnit === 'MWh') return `${(kwh / 1000).toFixed(1)} MWh${suffix}`;
  if (Math.abs(kwh) >= 1e6) return `${(kwh / 1e6).toFixed(2)} GWh${suffix}`;
  if (Math.abs(kwh) >= 1e3) return `${Math.round(kwh).toLocaleString()} kWh${suffix}`;
  return `${kwh.toFixed(0)} kWh${suffix}`;
}

export function fmtIrr(value, perYear = true) {
  if (value == null) return '-';
  return `${value.toFixed(0)} kWh/m\u00b2${perYear ? '/yr' : ''}`;
}

export function fmtArea(m2) {
  if (m2 == null) return '-';
  if (m2 >= 1e6) return `${(m2 / 1e6).toFixed(2)} km\u00b2`;
  if (m2 >= 1e4) return `${(m2 / 1e4).toFixed(2)} ha`;
  return `${Math.round(m2).toLocaleString()} m\u00b2`;
}

// India grid emission factor (kg CO2 / kWh) -- CEA CO2 Baseline Database FY2023-24
// weighted average (0.727 tCO2/MWh). Keep in sync with GRID_EMISSION_FACTOR in main.js.
const GRID_EMISSION_FACTOR = 0.727;

export function fmtCo2(kwhYear) {
  if (!kwhYear) return '-';
  const kg = kwhYear * GRID_EMISSION_FACTOR;
  if (kg >= 1000) return `${(kg / 1000).toFixed(1)} tCO\u2082/yr`;
  return `${kg.toFixed(0)} kg CO\u2082/yr`;
}

function polygonCentroid(ring) {
  let sx = 0;
  let sy = 0;
  const n = Math.max(0, ring.length - 1);
  if (!n) return [0, 0];
  for (let i = 0; i < n; i += 1) {
    sx += ring[i][0];
    sy += ring[i][1];
  }
  return [sx / n, sy / n];
}

function hashStr(input) {
  let h = 0;
  for (let i = 0; i < input.length; i += 1) {
    h = Math.imul(31, h) + input.charCodeAt(i) | 0;
  }
  return h >>> 0;
}

export function buildingKeyFromGeojson(geojson) {
  if (!geojson?.features?.[0]) return `k:${Date.now()}`;

  const feature = geojson.features[0];
  const props = feature.properties || {};
  if (props.full_id != null && String(props.full_id).length) return `id:${String(props.full_id)}`;
  if (props.OGB_ID != null && String(props.OGB_ID).length) return `ogb:${String(props.OGB_ID)}`;
  if (feature.id != null && String(feature.id).length) return `fid:${String(feature.id)}`;

  const geom = feature.geometry;
  if (geom?.type === 'Polygon' && geom?.coordinates?.[0]) {
    const [lon, lat] = polygonCentroid(geom.coordinates[0]);
    const area = props.area_in_meters != null ? Number(props.area_in_meters).toFixed(0) : '';
    return `geo:${lat.toFixed(5)},${lon.toFixed(5)}:${area}`;
  }

  return `h:${hashStr(JSON.stringify(geom || {})).toString(36)}`;
}

export class BuildingLru {
  constructor(max = BUILDING_CAP) {
    this.max = max;
    this.order = [];
    this.store = new Map();
  }

  touch(key) {
    const idx = this.order.indexOf(key);
    if (idx >= 0) {
      this.order.splice(idx, 1);
      this.order.push(key);
    }
  }

  add(key, payload) {
    if (this.store.has(key)) {
      this.store.set(key, payload);
      this.touch(key);
      return { isDuplicate: true, evictedKey: null };
    }

    let evictedKey = null;
    if (this.order.length >= this.max) {
      evictedKey = this.order.shift();
      this.store.delete(evictedKey);
    }

    this.order.push(key);
    this.store.set(key, payload);
    return { isDuplicate: false, evictedKey };
  }

  get(key) {
    return this.store.get(key);
  }

  list() {
    return this.order.map((key) => ({ key, entry: this.store.get(key) }));
  }
}