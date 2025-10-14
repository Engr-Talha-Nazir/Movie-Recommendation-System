
// ===== State & Utilities =====
const dom = id => document.getElementById(id);
const logEl = dom('log');
let DATA = null; // {X: tf.Tensor2D(float32), yOne: tf.Tensor2D(float32), yIdx: Int32Array, labels: string[], mean: number[], std: number[]}
let MODEL = null;
let TRAINED = false;
let rocChart = null;
let cmChart = null;

function log(msg){
  const time = new Date().toLocaleTimeString();
  logEl.innerHTML += `[${time}] ${msg}<br/>`;
  logEl.scrollTop = logEl.scrollHeight;
}

// ===== CSV → Tensors (float32 only) =====
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
  const yIdx = labels.map(y=>uniq.indexOf(y)); // JS number array
  // Float32 feature matrix
  const X = tf.tensor2d(rows, undefined, 'float32'); // [N,7] float32
  const {mean, variance} = tf.moments(X, 0);
  const std = variance.sqrt();
  const Xstd = X.sub(mean).div(std);
  const n = rows.length, c = uniq.length;
  // One-hot labels in float32
  const yOneData = new Float32Array(n * c);
  for(let i=0;i<n;i++){ yOneData[i*c + yIdx[i]] = 1; }
  const yOne = tf.tensor2d(yOneData, [n,c], 'float32');
  const data = {
    X: Xstd, yOne, yIdx: Int32Array.from(yIdx), labels: uniq,
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

// ===== Model (categorical, float32 pipeline) =====
function buildModel(inputDim, numClasses){
  const m = tf.sequential();
  m.add(tf.layers.dense({units:96, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.dropout({rate:0.15}));
  m.add(tf.layers.dense({units:48, activation:'relu'}));
  m.add(tf.layers.dense({units:numClasses, activation:'softmax'}));
  m.compile({optimizer: tf.train.adam(0.005), loss:'categoricalCrossentropy'}); // no metrics
  return m;
}

// ===== Manual evaluation: accuracy, ROC micro, confusion =====
async function manualValEval(model, Xva, yvaOne, yvaIdx){
  const logits = model.predict(Xva);
  const probs = await logits.array(); // [N][C], float
  logits.dispose?.();
  const yTrue = Array.from(await yvaIdx.data()); // int32 array
  const yPred = [];
  let correct = 0;
  for(let i=0;i<probs.length;i++){
    const row = probs[i]; let bi=0, bp=-1;
    for(let c=0;c<row.length;c++){ if(row[c]>bp){ bp=row[c]; bi=c; } }
    yPred.push(bi); if(bi===yTrue[i]) correct++;
  }
  // Flatten probs for ROC
  const C = probs[0].length;
  const probsFlat = new Float32Array(probs.length * C);
  let k=0; for(const row of probs){ for(const p of row){ probsFlat[k++]=p; } }
  return { acc: correct / yTrue.length, preds: yPred, probsFlat, yTrue };
}

// ===== ROC & Confusion rendering =====
function computeRocMicro(yTrue, yScore, numClasses){
  const N = yTrue.length, C = numClasses;
  const scores = []; const truths = [];
  for(let n=0;n<N;n++){
    for(let c=0;c<C;c++){
      scores.push(yScore[n*C + c]);
      truths.push( yTrue[n] === c ? 1 : 0 );
    }
  }
  const idx = scores.map((s,i)=>i).sort((a,b)=>scores[b]-scores[a]);
  let P = truths.reduce((a,b)=>a+b,0), Nneg = truths.length - P;
  let tp=0, fp=0; const tpr=[], fpr=[]; let lastScore = Infinity;
  for(const i of idx){
    if(scores[i] !== lastScore){
      tpr.push(P? tp/P : 0); fpr.push(Nneg? fp/Nneg : 0); lastScore = scores[i];
    }
    if(truths[i]===1) tp++; else fp++;
  }
  tpr.push(P? tp/P : 0); fpr.push(Nneg? fp/Nneg : 0);
  tpr.unshift(0); fpr.unshift(0); tpr.push(1); fpr.push(1);
  let auc=0; for(let i=1;i<fpr.length;i++){ const dx=fpr[i]-fpr[i-1]; const yavg=(tpr[i]+tpr[i-1])/2; auc += dx*yavg; }
  return {fpr, tpr, auc: Math.max(0, Math.min(1, auc))};
}

function confusionMatrix(yTrue, yPred, numClasses){
  const cm = Array.from({length:numClasses},()=>Array(numClasses).fill(0));
  for(let i=0;i<yTrue.length;i++) cm[yTrue[i]][yPred[i]]++;
  return cm;
}

function ensureRocChart(){
  const ctx = document.getElementById('rocChart');
  if(rocChart) rocChart.destroy();
  rocChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'ROC (micro-average)', data: [], fill:false, tension:0.15 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { x:{ title:{display:true,text:'False Positive Rate'}, min:0, max:1 }, y:{ title:{display:true,text:'True Positive Rate'}, min:0, max:1 } },
      plugins: { legend:{display:true} }
    }
  });
}

function renderRoc(fpr, tpr, auc){
  ensureRocChart();
  const pts = fpr.map((x,i)=>({x, y:tpr[i]}));
  rocChart.data.datasets[0].data = pts;
  rocChart.data.labels = fpr;
  rocChart.update();
  dom('aucNote').textContent = `ROC (micro-average) · AUC = ${auc.toFixed(3)}`;
}

function ensureCmChart(labels){
  const ctx = document.getElementById('cmChart');
  if(cmChart) cmChart.destroy();
  const L = Math.min(labels.length, 10);
  cmChart = new Chart(ctx, {
    type: 'bar',
    data: { labels: labels.slice(0,L), datasets: [{ label: 'Correct (diag) counts', data: new Array(L).fill(0) }] },
    options: { responsive:true, maintainAspectRatio:false, scales:{ y:{ beginAtZero:true } }, plugins:{ legend:{display:false} } }
  });
}

function renderCm(cm, labels){
  const diag = cm.map((row,i)=>row[i]);
  const L = Math.min(labels.length, 10);
  ensureCmChart(labels);
  cmChart.data.labels = labels.slice(0,L);
  cmChart.data.datasets[0].data = diag.slice(0,L);
  cmChart.update();
}

// ===== Train & Predict =====
async function train(){
  try{
    if(!DATA){ log('⚠️ Dataset missing.'); return; }
    if(MODEL){ MODEL.dispose(); MODEL = null; }
    const n = DATA.X.shape[0];
    const {tr, va} = trainValSplit(n, 0.8);
    const Xtr = subset(DATA.X, tr);
    const ytr = subset(DATA.yOne, tr);
    const Xva = subset(DATA.X, va);
    const yvaOne = subset(DATA.yOne, va);
    const yvaIdx = tf.tensor1d(Array.from(DATA.yIdx).filter((_,i)=>va.includes(i)), 'int32'); // gather yIdx by va

    MODEL = buildModel(DATA.X.shape[1], DATA.labels.length);
    dom('modelState').textContent = 'model: training...';
    dom('predictBtn').disabled = true;
    log(`🚀 Training on ${tr.length} rows, validating on ${va.length} rows…`);

    let lastEva = null;
    await MODEL.fit(Xtr, ytr, {
      epochs: 20, batchSize: 32, shuffle: true,
      callbacks: {
        onEpochEnd: async (e, logs) => {
          lastEva = await manualValEval(MODEL, Xva, yvaOne, yvaIdx);
          dom('k_epochs').textContent = String(e+1);
          dom('k_acc').textContent = (lastEva.acc*100).toFixed(1) + '%';
          log(`epoch ${e+1}: loss=${(logs.loss??0).toFixed(3)} · val_acc=${(lastEva.acc).toFixed(3)}`);
        }
      }
    });

    // Diagnostics
    const eva = await manualValEval(MODEL, Xva, yvaOne, yvaIdx);
    const roc = computeRocMicro(eva.yTrue, eva.probsFlat, DATA.labels.length);
    renderRoc(roc.fpr, roc.tpr, roc.auc);
    const cm = confusionMatrix(eva.yTrue, eva.preds, DATA.labels.length);
    renderCm(cm, DATA.labels);

    // Cleanup
    Xtr.dispose(); ytr.dispose(); Xva.dispose(); yvaOne.dispose(); yvaIdx.dispose();
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
  const t = tf.tensor2d([xs], undefined, 'float32');
  const p = MODEL.predict(t);
  const probs = (await p.data());
  const top = topK(probs, 3);
  dom('topk').innerHTML = top.map(({i,p})=>`<li><strong>${DATA.labels[i]}</strong> — ${(p*100).toFixed(1)}%</li>`).join('');
  p.dispose(); t.dispose();
}

function resetAll(){
  if(DATA){ DATA.X.dispose?.(); DATA.yOne.dispose?.(); }
  if(MODEL){ MODEL.dispose(); }
  DATA = parseCSV(window.CROP_CSV);
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

// ===== Init =====
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
