
// ---- Utilities & State ----
const dom = id => document.getElementById(id);
const logEl = dom('log');
let DATA = null; // {X: tf.Tensor2D, y: tf.Tensor1D, labels: string[], mean: number[], std: number[]}
let MODEL = null;
let TRAINED = false;

// Charts
let rocChart = null;
let cmChart = null;

function log(msg){
  const time = new Date().toLocaleTimeString();
  logEl.innerHTML += `[${time}] ${msg}<br/>`;
  logEl.scrollTop = logEl.scrollHeight;
}

// ---- Data Prep ----
function parseCSV(text){
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(',').map(s=>s.trim());
  const idx = {
    N: header.indexOf('N'), P: header.indexOf('P'), K: header.indexOf('K'),
    temperature: header.indexOf('temperature'), humidity: header.indexOf('humidity'),
    ph: header.indexOf('ph'), rainfall: header.indexOf('rainfall'),
    label: header.findIndex(h => /^(label|crop)$/i.test(h))
  };
  const rows = [];
  const labels = [];
  for(const line of lines){
    if(!line.trim()) continue;
    const cols = line.split(',');
    const x = [idx.N,idx.P,idx.K,idx.temperature,idx.humidity,idx.ph,idx.rainfall].map(i=>Number(cols[i]));
    const y = String(cols[idx.label]).trim();
    if(x.some(v=>Number.isNaN(v)) || !y) continue;
    rows.push(x);
    labels.push(y);
  }
  const uniq = Array.from(new Set(labels)).sort();
  const yEncoded = labels.map(y=>uniq.indexOf(y));
  const X = tf.tensor2d(rows);
  const {mean, variance} = tf.moments(X, 0);
  const std = variance.sqrt();
  const Xstd = X.sub(mean).div(std);
  const data = {
    X: Xstd, y: tf.tensor1d(yEncoded, 'int32'), labels: uniq,
    mean: Array.from(mean.dataSync()), std: Array.from(std.dataSync())
  };
  X.dispose(); mean.dispose(); variance.dispose(); std.dispose();
  return data;
}

function standardize(x){
  const mu = DATA.mean, sd = DATA.std;
  return x.map((v,i)=> (v - mu[i]) / (sd[i] || 1));
}

function readInputs(){
  const v = id=> Number(dom(id).value);
  return [v('f_N'), v('f_P'), v('f_K'), v('f_temp'), v('f_hum'), v('f_ph'), v('f_rain')];
}

function trainValSplit(n, frac=0.8){
  const idx = tf.util.createShuffledIndices(n);
  const nTrain = Math.floor(n*frac);
  const tr = Array.from(idx.slice(0,nTrain));
  const va = Array.from(idx.slice(nTrain));
  return {tr, va};
}

function subset(t, idxArr){
  const idx = tf.tensor1d(idxArr, 'int32');
  const out = tf.gather(t, idx);
  idx.dispose();
  return out;
}

// ---- Model ----
function buildModel(inputDim, numClasses){
  const m = tf.sequential();
  m.add(tf.layers.dense({units:96, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.dropout({rate:0.15}));
  m.add(tf.layers.dense({units:48, activation:'relu'}));
  m.add(tf.layers.dense({units:numClasses, activation:'softmax'}));
  m.compile({optimizer: tf.train.adam(0.005), loss:'sparseCategoricalCrossentropy', metrics:[tf.metrics.sparseCategoricalAccuracy]});
  return m;
}

// ---- Metrics: ROC (micro) & Confusion ----
function computeRocMicro(yTrue, yScore, numClasses){
  // yTrue: Int32Array of shape [N] with class ids
  // yScore: Float32Array of shape [N, C] flattened row-major
  const N = yTrue.length;
  const C = numClasses;
  // Build flattened arrays for micro-avg (one-vs-all, flattened)
  const scores = [];
  const truths = [];
  for(let n=0;n<N;n++){
    for(let c=0;c<C;c++){
      scores.push(yScore[n*C + c]);
      truths.push( yTrue[n] === c ? 1 : 0 );
    }
  }
  // Sort by descending score
  const idx = scores.map((s,i)=>i).sort((a,b)=>scores[b]-scores[a]);
  let P = truths.reduce((a,b)=>a+b,0); // positives
  let Nneg = truths.length - P;        // negatives
  let tp=0, fp=0;
  const tpr=[], fpr=[];
  let lastScore = Infinity;
  for(const i of idx){
    if(scores[i] !== lastScore){
      // record point before moving threshold
      tpr.push(P? tp/P : 0);
      fpr.push(Nneg? fp/Nneg : 0);
      lastScore = scores[i];
    }
    if(truths[i]===1) tp++; else fp++;
  }
  // final point
  tpr.push(P? tp/P : 0); fpr.push(Nneg? fp/Nneg : 0);
  // Ensure (0,0) and (1,1)
  tpr.unshift(0); fpr.unshift(0);
  tpr.push(1); fpr.push(1);
  // AUC via trapezoidal rule
  let auc=0;
  for(let i=1;i<fpr.length;i++){
    const dx = fpr[i]-fpr[i-1];
    const yavg = (tpr[i]+tpr[i-1])/2;
    auc += dx*yavg;
  }
  return {fpr, tpr, auc: Math.max(0, Math.min(1, auc))};
}

function confusionMatrix(yTrue, yPred, numClasses){
  const cm = Array.from({length:numClasses},()=>Array(numClasses).fill(0));
  for(let i=0;i<yTrue.length;i++) cm[yTrue[i]][yPred[i]]++;
  return cm;
}

// ---- Charts ----
function ensureRocChart(){
  const ctx = document.getElementById('rocChart');
  if(rocChart) rocChart.destroy();
  rocChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'ROC (micro-average)',
        data: [],
        fill: false,
        tension: 0.15
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: {display:true, text:'False Positive Rate'} , min:0, max:1 },
        y: { title: {display:true, text:'True Positive Rate'} , min:0, max:1 }
      },
      plugins: { legend: {display:true} }
    }
  });
}

function renderRoc(fpr, tpr, auc){
  ensureRocChart();
  const pts = fpr.map((x,i)=>({x, y:tpr[i]}));
  rocChart.data.datasets[0].data = pts;
  rocChart.data.labels = fpr;
  rocChart.update();
  const note = dom('aucNote');
  note.textContent = `ROC (micro-average) · AUC = ${auc.toFixed(3)}`;
}

function ensureCmChart(numClasses, labels){
  const ctx = document.getElementById('cmChart');
  if(cmChart) cmChart.destroy();
  cmChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels.slice(0, Math.min(labels.length, 10)), // show first 10 classes for readability
      datasets: [{
        label: 'Correct (diag) counts',
        data: new Array(Math.min(labels.length, 10)).fill(0)
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: { y: { beginAtZero:true } },
      plugins: { legend: {display:false} }
    }
  });
}

function renderCm(cm, labels){
  // Show diagonal counts for readability (top-1 correct per class)
  const diag = cm.map((row,i)=>row[i]);
  const L = Math.min(labels.length, 10);
  ensureCmChart(L, labels);
  cmChart.data.labels = labels.slice(0,L);
  cmChart.data.datasets[0].data = diag.slice(0,L);
  cmChart.update();
}

// ---- Train / Predict ----
async function train(){
  try{
    if(!DATA){ log('⚠️ Dataset missing.'); return; }
    if(MODEL){ MODEL.dispose(); MODEL = null; }
    const n = DATA.X.shape[0];
    const {tr, va} = trainValSplit(n, 0.8);
    const Xtr = subset(DATA.X, tr); const ytr = subset(DATA.y, tr);
    const Xva = subset(DATA.X, va); const yva = subset(DATA.y, va);
    MODEL = buildModel(DATA.X.shape[1], DATA.labels.length);
    dom('modelState').textContent = 'model: training...';
    dom('predictBtn').disabled = true;
    log(`🚀 Training on ${tr.length} rows, validating on ${va.length} rows…`);
    let best = 0;
    await MODEL.fit(Xtr, ytr, {
      epochs: 20, batchSize: 32, validationData: [Xva, yva], shuffle: true,
      callbacks: { onEpochEnd: (e, logs)=>{
        const accKey = ['acc','accuracy','sparseCategoricalAccuracy'].find(k=>k in logs) || 'accuracy';
        const valAccKey = ['val_acc','val_accuracy','val_sparseCategoricalAccuracy'].find(k=>k in logs) || 'val_accuracy';
        const va = logs[valAccKey] ?? 0;
        best = Math.max(best, va);
        dom('k_epochs').textContent = String(e+1);
        dom('k_acc').textContent = (best*100).toFixed(1) + '%';
        log(`epoch ${e+1}: loss=${(logs.loss??0).toFixed(3)} · acc=${(logs[accKey]||0).toFixed(3)} · val_acc=${(va).toFixed(3)}`)
      }}
    });
    // Validation predictions for diagnostics
    const yva_true = Array.from(await yva.data());
    const logits = MODEL.predict(Xva);
    const yva_prob = await logits.data(); // flattened [N, C]
    const yva_pred = [];
    {
      const N = yva_true.length, C = DATA.labels.length;
      for(let i=0;i<N;i++){
        let bestI=0, bestP=-1;
        for(let c=0;c<C;c++){
          const p = yva_prob[i*C + c];
          if(p>bestP){ bestP=p; bestI=c; }
        }
        yva_pred.push(bestI);
      }
    }
    // ROC micro
    const roc = computeRocMicro(new Int32Array(yva_true), new Float32Array(yva_prob), DATA.labels.length);
    renderRoc(roc.fpr, roc.tpr, roc.auc);
    // Confusion snapshot
    const cm = confusionMatrix(yva_true, yva_pred, DATA.labels.length);
    renderCm(cm, DATA.labels);
    // Cleanup
    logits.dispose?.(); Xtr.dispose(); ytr.dispose(); Xva.dispose(); yva.dispose();
    TRAINED = true;
    dom('modelState').textContent = 'model: trained';
    dom('predictBtn').disabled = false;
    log('✅ Training complete. Ready to recommend.');
  }catch(err){
    console.error(err);
    log('❌ Training error: ' + err.message);
    dom('modelState').textContent = 'model: error';
  }
}

function topK(probs, k=3){
  const arr = Array.from(probs).map((p,i)=>({i,p}));
  arr.sort((a,b)=>b.p-a.p);
  return arr.slice(0,k);
}

async function predict(){
  if(!TRAINED){ log('⚠️ Train the model first.'); return; }
  const xs = standardize(readInputs());
  const t = tf.tensor2d([xs]);
  const p = MODEL.predict(t);
  const probs = (await p.data());
  const top = topK(probs, 3);
  dom('topk').innerHTML = top.map(({i,p})=>`<li><strong>${DATA.labels[i]}</strong> — ${(p*100).toFixed(1)}%</li>`).join('');
  p.dispose(); t.dispose();
}

function resetAll(){
  if(DATA){ DATA.X.dispose?.(); DATA.y.dispose?.(); }
  if(MODEL){ MODEL.dispose(); }
  DATA = parseCSV(window.CROP_CSV); // reload from embedded CSV
  dom('k_rows').textContent = String(DATA.X.shape[0]);
  dom('k_classes').textContent = String(DATA.labels.length);
  dom('k_acc').textContent = '–';
  dom('k_epochs').textContent = '–';
  dom('topk').innerHTML = '';
  dom('modelState').textContent = 'model: data loaded';
  if(rocChart){ rocChart.destroy(); rocChart=null; }
  if(cmChart){ cmChart.destroy(); cmChart=null; }
  log('🔁 State reset.');
}

// ---- Init ----
(function init(){
  try{
    if(!window.CROP_CSV){ log('❌ data.js did not load.'); return; }
    DATA = parseCSV(window.CROP_CSV);
    dom('k_rows').textContent = String(DATA.X.shape[0]);
    dom('k_classes').textContent = String(DATA.labels.length);
    dom('modelState').textContent = 'model: data loaded';
    log(`📦 Loaded ${DATA.X.shape[0]} rows · ${DATA.labels.length} classes`);
  }catch(err){
    log('❌ Failed to parse embedded CSV: ' + err.message);
  }
  dom('trainBtn').addEventListener('click', train);
  dom('predictBtn').addEventListener('click', predict);
  dom('resetBtn').addEventListener('click', resetAll);
})();
