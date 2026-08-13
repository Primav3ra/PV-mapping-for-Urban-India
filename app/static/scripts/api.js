export async function fetchPresets() {
  const response = await fetch('/api/presets');
  if (!response.ok) throw new Error(`Presets request failed (${response.status})`);
  return response.json();
}

export async function fetchYield(payload) {
  const response = await fetch('/api/yield', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return response.json();
}

export async function fetchTiles(payload) {
  const response = await fetch('/api/tiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return response.json();
}

export async function fetchSeries(payload) {
  const response = await fetch('/api/series', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return response.json();
}

export async function fetchBuildings(payload) {
  const response = await fetch('/api/buildings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return response.json();
}