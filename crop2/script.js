// script.js
import CROP_DATA from './data.js';

/** ========= Helpers (stats & scaling) ========= **/

const FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"];

function mean(arr){ return arr.reduce((a,b)=>a+b,0)/arr.length; }
function std(arr){
  const m = mean(arr); const v = mean(arr.map(x => (x-m)**2));
  return Math.sqrt(v || 1e-9);
}
function zscore(x, m, s){ return (x - m) / (s || 1e-9); }

const STATS = (() => {
  const m = {}, s = {};
  FEATURES.forEach(f => {
    const col = CROP_DATA.map(r => Number(r[f]));
    m[f] = mean(col); s[f] = std(col);
  });
  return {mean:m, std:s};
})();

/** ========= Strategy-aware distance weighting ========= **/
function featureWeights(segment){
  // Default equal weights
  const w = { N:1, P:1, K:1, temperature:1, humidity:1, ph:1, rainfall:1 };
  if(segment === 'water_scarce'){
    // Emphasize rainfall proximity so we avoid crops that need very different rainfall
    w.rainfall = 2.0; w.humidity = 1.3;
  }
  if(segment === 'ph_sensitive'){
    // Emphasize pH match where soil prep is costly
    w.ph = 2.0; w.N = 0.8; w.P = 0.8; w.K = 0.8;
  }
  return w;
}

/** ========= kNN (k=7) ========= **/
function knnPredict(x, k=7, segment='default'){
  const w = featureWeights(segment);
  // Compute distances in normalized space with weights
  const dist = CROP_DATA.map((row, idx) => {
    let d2 = 0;
    FEATURES.forEach(f => {
      const zx = zscore(Number(x[f]), STATS.mean[f], STATS.std[f]);
      const zy = zscore(Number(row[f]), STATS.mean[f], STATS.std[f]);
      const diff = zx - zy;
      d2 += (w[f] || 1) * diff * diff;
    });
    return {i: idx, label: row.label, d: Math.sqrt(d2)};
  });
  dist.sort((a,b)=>a.d-b.d);
  const top = dist.slice(0, Math.max(1,k));
  const vote = {};
  top.forEach(t => { vote[t.label] = (vote[t.label] || 0) + 1/(t.d+1e-6); });
  const ranked = Object.entries(vote).sort((a,b)=>b[1]-a[1]).map(([label,score])=>({label,score}));
  return {ranked, neighbors: top};
}

/** ========= Quick CV (holdout) ========= **/
function quickCV(segment='default'){
  // Simple 80/20 split + kNN eval
  const idx = [...CROP_DATA.keys()];
  // Shuffle
  for(let i=idx.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [idx[i],idx[j]]=[idx[j],idx[i]]; }
  const cut = Math.floor(0.8*idx.length);
  const trainIdx = new Set(idx.slice(0,cut));
  const testIdx = idx.slice(cut);

  // Build lightweight train array
  const TRAIN = idx.slice(0,cut).map(i => CROP_DATA[i]);

  // Predict on test by kNN on TRAIN only
  function knnOnTrain(x, k=7){
    const w = featureWeights(segment);
    const dist = TRAIN.map((row) => {
      let d2 = 0;
      FEATURES.forEach(f => {
        const zx = zscore(Number(x[f]), STATS.mean[f], STATS.std[f]);
        const zy = zscore(Number(row[f]), STATS.mean[f], STATS.std[f]);
        const diff = zx - zy;
        d2 += (w[f] || 1) * diff * diff;
      });
      return {label: row.label, d: Math.sqrt(d2)};
    });
    dist.sort((a,b)=>a.d-b.d);
    const top = dist.slice(0,7);
    const vote = {};
    top.forEach(t => { vote[t.label] = (vote[t.label] || 0) + 1/(t.d+1e-6); });
    return Object.entries(vote).sort((a,b)=>b[1]-a[1])[0][0];
  }

  let correct=0;
  testIdx.forEach(i => {
    const x = CROP_DATA[i];
    const yhat = knnOnTrain(x, 7);
    if(yhat === x.label) correct++;
  });
  return {acc: correct/testIdx.length, n:testIdx.length};
}

/** ========= UI wiring ========= **/

function $(sel){ return document.querySelector(sel); }
function $all(sel){ return document.querySelectorAll(sel); }

function readInputs(){
  const values = {};
  FEATURES.forEach(f => values[f] = Number($(`#${f}`).value));
  const seg = [...$all('input[name="segment"]')].find(r => r.checked)?.value || 'default';
  return {x: values, segment: seg};
}

function autoFillSuggested(){
  // Fill inputs with dataset medians to speed demo
  const med = {
    N: 67, P: 47, K: 50,
    temperature: 25, humidity: 65, ph: 6.4, rainfall: 95
  };
  FEATURES.forEach(f => { const el = $(`#${f}`); if(el && !el.value) el.value = med[f]; });
}

function renderRecs(result){
  const recs = $("#recs");
  recs.innerHTML = "";
  result.ranked.slice(0,6).forEach((r, idx) => {
    const el = document.createElement('div');
    el.className = 'rec';
    el.innerHTML = `
      <h4>${idx+1}. ${r.label}</h4>
      <div class="score">Score: ${r.score.toFixed(3)}</div>
      <div class="muted">Strategy-aware: ${document.querySelector('input[name="segment"]:checked')?.value}</div>
    `;
    recs.appendChild(el);
  });

  // neighbor evidence
  const nn = $("#neighbors");
  nn.innerHTML = `
    <div class="nn-card">
      <div style="margin-bottom:6px;font-weight:600">Nearest Neighbors (evidence)</div>
      <div class="nn-grid">
        ${result.neighbors.map(n => `
          <div>
            <div style="font-weight:700">${n.label}</div>
            <div>dist=${n.d.toFixed(3)}</div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

/** ========= EDA Charts ========= **/
function makeBar(ctx, labels, values, title){
  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: title, data: values }]},
    options: { responsive:true, plugins:{ legend:{display:false} } }
  });
}
function makeHist(values, bins=20){
  const min = Math.min(...values), max = Math.max(...values);
  const step = (max - min) / bins;
  const edges = Array.from({length: bins}, (_,i)=>min + i*step);
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    let b = Math.floor((v - min)/step);
    if(b >= bins) b = bins-1;
    counts[b] += 1;
  });
  const labels = edges.map((e,i)=> i===bins-1 ? `${e.toFixed(1)}+` : `${e.toFixed(1)}–${(e+step).toFixed(1)}`);
  return {labels, counts};
}
function buildEDA(){
  // Label distribution
  const labelCounts = {};
  CROP_DATA.forEach(r => { labelCounts[r.label] = (labelCounts[r.label] || 0) + 1; });
  const labels = Object.keys(labelCounts).sort();
  const counts = labels.map(l => labelCounts[l]);
  makeBar(document.getElementById('chart-labels'), labels, counts, 'Class counts');

  // Temperature hist
  const temp = CROP_DATA.map(r => Number(r.temperature));
  let hist = makeHist(temp, 18);
  makeBar(document.getElementById('chart-temp'), hist.labels, hist.counts, 'Temperature histogram');

  // Rainfall hist
  const rain = CROP_DATA.map(r => Number(r.rainfall));
  hist = makeHist(rain, 18);
  makeBar(document.getElementById('chart-rain'), hist.labels, hist.counts, 'Rainfall histogram');

  // pH hist
  const ph = CROP_DATA.map(r => Number(r.ph));
  hist = makeHist(ph, 16);
  makeBar(document.getElementById('chart-ph'), hist.labels, hist.counts, 'pH histogram');
}

/** ========= Bootstrap ========= **/
window.addEventListener('DOMContentLoaded', () => {
  autoFillSuggested();
  buildEDA();

  $('#btn-recommend').addEventListener('click', () => {
    const {x, segment} = readInputs();
    const result = knnPredict(x, 7, segment);
    renderRecs(result);
  });

  $('#btn-cv').addEventListener('click', () => {
    const seg = document.querySelector('input[name="segment"]:checked')?.value || 'default';
    const {acc, n} = quickCV(seg);
    const el = document.createElement('div');
    el.className = 'rec';
    el.innerHTML = `<h4>Quick Holdout Accuracy</h4>
      <div class="score">${(acc*100).toFixed(2)}%</div>
      <div class="muted">Test size: ${n} · Segment: ${seg}</div>`;
    const recs = $("#recs");
    recs.prepend(el);
  });
});
