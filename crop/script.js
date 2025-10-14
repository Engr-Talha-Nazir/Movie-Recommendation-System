
const dom = id => document.getElementById(id);
const logEl = dom('log');
let DATA = null; // {X: tf.Tensor2D, y: tf.Tensor1D, labels: string[], mean: number[], std: number[]}
let MODEL = null;
let TRAINED = false;

function log(msg){
  const time = new Date().toLocaleTimeString();
  logEl.innerHTML += `[${time}] ${msg}<br/>`;
  logEl.scrollTop = logEl.scrollHeight;
}

function parseCSV(text){
  // header: N,P,K,temperature,humidity,ph,rainfall,label (or crop)
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

function trainValSplit(n, frac=0.8){
  const idx = tf.util.createShuffledIndices(n);
  const nTrain = Math.floor(n*frac);
  const tr = Array.from(idx.slice(0,nTrain));
  const va = Array.from(idx.slice(nTrain));
  return {tr, va};
}

function buildModel(inputDim, numClasses){
  const m = tf.sequential();
  m.add(tf.layers.dense({units:64, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.dropout({rate:0.1}));
  m.add(tf.layers.dense({units:32, activation:'relu'}));
  m.add(tf.layers.dense({units:numClasses, activation:'softmax'}));
  m.compile({optimizer: tf.train.adam(0.01), loss:'sparseCategoricalCrossentropy', metrics:['accuracy']});
  return m;
}

function subset(t, idxArr){
  const idx = tf.tensor1d(idxArr, 'int32');
  const out = tf.gather(t, idx);
  idx.dispose();
  return out;
}

async function train(){
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
    epochs: 25, batchSize: 32, validationData: [Xva, yva], shuffle: true,
    callbacks: { onEpochEnd: (e, logs)=>{
      best = Math.max(best, logs.val_accuracy || 0);
      dom('k_epochs').textContent = String(e+1);
      dom('k_acc').textContent = (best*100).toFixed(1) + '%';
      log(`epoch ${e+1}: loss=${logs.loss.toFixed(3)} · acc=${(logs.acc||logs.accuracy).toFixed(3)} · val_acc=${(logs.val_acc||logs.val_accuracy).toFixed(3)}`)
    }}
  });
  Xtr.dispose(); ytr.dispose(); Xva.dispose(); yva.dispose();
  TRAINED = true;
  dom('modelState').textContent = 'model: trained';
  dom('predictBtn').disabled = false;
  log('✅ Training complete. Ready to recommend.');
}

function standardize(x){
  const mu = DATA.mean, sd = DATA.std;
  return x.map((v,i)=> (v - mu[i]) / (sd[i] || 1));
}

function readInputs(){
  const v = id=> Number(dom(id).value);
  return [v('f_N'), v('f_P'), v('f_K'), v('f_temp'), v('f_hum'), v('f_ph'), v('f_rain')];
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
  DATA = null; MODEL = null; TRAINED = false;
  dom('k_rows').textContent = '–';
  dom('k_classes').textContent = '–';
  dom('k_acc').textContent = '–';
  dom('k_epochs').textContent = '–';
  dom('topk').innerHTML = '';
  dom('modelState').textContent = 'model: not trained';
  log('🔁 State reset.');
}

// Initialize: parse embedded CSV
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
