// ===== Settings =====
const MODEL_NAME = "MLP(128-64-32, ReLU, Dropout 0.2) · Adam lr=0.001 · CE";
const EPOCHS = 80;
const BATCH_SIZE = 32;

// ===== State & Utilities =====
const dom = id => document.getElementById(id);
const logEl = dom('log');
let DATA = null; // {X: tf.Tensor2D, yOne: tf.Tensor2D, yIdx: tf.Tensor1D, labels: string[], mean: number[], std: number[]}
let MODEL = null;
let TRAINED = false;

// Charts
let rocChart=null, cmChart=null, lossChart=null, valAccChart=null;
const lossHistory = [];
const valAccHistory = [];

function log(msg){
  const time = new Date().toLocaleTimeString();
  logEl.innerHTML += `[${time}] ${msg}<br/>`;
  logEl.scrollTop = logEl.scrollHeight;
}

// ===== CSV decode + parse =====
function decodeEmbeddedCSV(){
  try{
    const csv = atob(window.CROP_CSV_BASE64 || "");
    if(!csv || csv.trim().length === 0) throw new Error("Embedded CSV empty.");
    return csv;
  }catch(err){
    throw new Error("Failed to decode embedded CSV (base64): "+err.message);
  }
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
  if (Object.values(idx).some(v => v < 0)) {
    throw new Error("CSV headers must include N,P,K,temperature,humidity,ph,rainfall,label(crop)");
  }
  const rows = [], labels = [];
  for(const line of lines){
    if(!line.trim()) continue;
    const cols = line.split(',');
    const x = [idx.N,idx.P,idx.K,idx.temperature,idx.humidity,idx.ph,idx.rainfall].map(i=>Number(cols[i]));
    const y = String(cols[idx.label]).trim();
    if(x.some(v=>Number.isNaN(v)) || !y) continue;
    rows.push(x); labels.push(y);
  }
  const uniq = Array.from(new Set(labels)).sort();
  const yIdxArr = labels.map(y=>uniq.indexOf(y));
  const X = tf.tensor2d(rows, undefined, 'float32');
  const {mean, variance} = tf.moments(X, 0);
  const std = variance.sqrt();
  const Xstd = X.sub(mean).div(std);
  const n = rows.length, c = uniq.length;
  const yOneData = new Float32Array(n * c);
  for(let i=0;i<n;i++){ yOneData[i*c + yIdxArr[i]] = 1; }
  const yOne = tf.tensor2d(yOneData, [n,c], 'float32');
  const yIdx = tf.tensor1d(Int32Array.from(yIdxArr), 'int32');
  const data = { X: Xstd, yOne, yIdx, labels: uniq, mean: Array.from(mean.dataSync()), std: Array.from(std.dataSync()) };
  X.dispose(); mean.dispose(); variance.dispose(); std.dispose();
  return data;
}

function standardize(x){ const mu=DATA.mean, sd=DATA.std; return x.map((v,i)=> (v - mu[i]) / (sd[i] || 1)); }
function readInputs(){ const v=id=>Number(dom(id).value); return [v('f_N'),v('f_P'),v('f_K'),v('f_temp'),v('f_hum'),v('f_ph'),v('f_rain')]; }
function trainValSplit(n, frac=0.8){ const idx=tf.util.createShuffledIndices(n); const nTr=Math.floor(n*frac); return {tr:Array.from(idx.slice(0,nTr)), va:Array.from(idx.slice(nTr))}; }
function subset(t, idxArr){ const idx=tf.tensor1d(idxArr,'int32'); const out=tf.gather(t,idx); idx.dispose(); return out; }

// ===== Model =====
function buildModel(inputDim, numClasses){
  const m = tf.sequential();
  m.add(tf.layers.dense({units:128, activation:'relu', inputShape:[inputDim]}));
  m.add(tf.layers.dropout({rate:0.2}));
  m.add(tf.layers.dense({units:64, activation:'relu'}));
  m.add(tf.layers.dense({units:32, activation:'relu'}));
  m.add(tf.layers.dense({units:numClasses, activation:'softmax'}));
  m.compile({optimizer: tf.train.adam(0.001), loss:'categoricalCrossentropy'});
  return m;
}

// ===== Manual evaluation =====
async function manualValEval(model, Xva, yvaIdx){
  const logits = model.predict(Xva);
  const probs = await logits.array(); // [N][C]
  logits.dispose?.();
  const yTrue = Array.from(await yvaIdx.data());
  const yPred = [];
  let correct = 0;
  for(let i=0;i<probs.length;i++){
    const row = probs[i]; let bi=0, bp=-1;
    for(let c=0;c<row.length;c++){ if(row[c]>bp){ bp=row[c]; bi=c; } }
    yPred.push(bi); if(bi===yTrue[i]) correct++;
  }
  const C = probs[0].length;
  const probsFlat = new Float32Array(probs.length * C);
  let k=0; for(const row of probs){ for(const p of row){ probsFlat[k++]=p; } }
  return { acc: correct / yTrue.length, preds: yPred, probsFlat, yTrue };
}

// ===== Charts =====
function ensureRocChart(){ const ctx=document.getElementById('rocChart'); if(rocChart) rocChart.destroy(); rocChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label:'ROC (micro-average)',data:[],fill:false,tension:0.15}]},options:{responsive:true,maintainAspectRatio:false,scales:{x:{title:{display:true,text:'False Positive Rate'},min:0,max:1},y:{title:{display:true,text:'True Positive Rate'},min:0,max:1}},plugins:{legend:{display:true}}}); }
function renderRoc(fpr,tpr,auc){ ensureRocChart(); rocChart.data.datasets[0].data=fpr.map((x,i)=>({x,y:tpr[i]})); rocChart.data.labels=fpr; rocChart.update(); dom('aucNote').textContent=`ROC (micro-average) · AUC = ${auc.toFixed(3)}`; }

function ensureCmChart(labels){ const ctx=document.getElementById('cmChart'); if(cmChart) cmChart.destroy(); const L=Math.min(labels.length,10); cmChart=new Chart(ctx,{type:'bar',data:{labels:labels.slice(0,L),datasets:[{label:'Correct (diag) counts',data:new Array(L).fill(0)}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{beginAtZero:true}},plugins:{legend:{display:false}}}); }
function renderCm(cm,labels){ const diag=cm.map((row,i)=>row[i]); const L=Math.min(labels.length,10); ensureCmChart(labels); cmChart.data.labels=labels.slice(0,L); cmChart.data.datasets[0].data=diag.slice(0,L); cmChart.update(); }

function ensureLossChart(){ const ctx=document.getElementById('lossChart'); if(lossChart) lossChart.destroy(); lossChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label:'Train Loss',data:[],fill:false,tension:0.15}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true}}}); }
function ensureValAccChart(){ const ctx=document.getElementById('valAccChart'); if(valAccChart) valAccChart.destroy(); valAccChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label:'Val Accuracy',data:[],fill:false,tension:0.15}]},options:{responsive:true,maintainAspectRatio:false,scales:{y:{min:0,max:1}},plugins:{legend:{display:true}}}); }
function updateLoss(epoch, loss){ ensureLossChart(); lossHistory.push({x:epoch,y:loss}); lossChart.data.labels=lossHistory.map(p=>p.x); lossChart.data.datasets[0].data=lossHistory.map(p=>p.y); lossChart.update(); }
function updateValAcc(epoch, acc){ ensureValAccChart(); valAccHistory.push({x:epoch,y:acc}); valAccChart.data.labels=valAccHistory.map(p=>p.x); valAccChart.data.datasets[0].data=valAccHistory.map(p=>p.y); valAccChart.update(); }

// ===== ROC & CM helpers =====
function computeRocMicro(yTrue, yScore, numClasses){
  const N=yTrue.length, C=numClasses; const scores=[]; const truths=[];
  for(let n=0;n<N;n++){ for(let c=0;c<C;c++){ scores.push(yScore[n*C+c]); truths.push(yTrue[n]===c?1:0); } }
  const idx=scores.map((s,i)=>i).sort((a,b)=>scores[b]-scores[a]);
  let P=truths.reduce((a,b)=>a+b,0), Nneg=truths.length-P, tp=0, fp=0; const tpr=[],fpr=[]; let last=Infinity;
  for(const i of idx){ if(scores[i]!==last){ tpr.push(P?tp/P:0); fpr.push(Nneg?fp/Nneg:0); last=scores[i]; } if(truths[i]===1) tp++; else fp++; }
  tpr.push(P?tp/P:0); fpr.push(Nneg?fp/Nneg:0); tpr.unshift(0); fpr.unshift(0); tpr.push(1); fpr.push(1);
  let auc=0; for(let i=1;i<fpr.length;i++){ const dx=fpr[i]-fpr[i-1]; const yavg=(tpr[i]+tpr[i-1])/2; auc+=dx*yavg; }
  return {fpr,tpr,auc:Math.max(0,Math.min(1,auc))};
}
function confusionMatrix(yTrue,yPred,numClasses){ const cm=Array.from({length:numClasses},()=>Array(numClasses).fill(0)); for(let i=0;i<yTrue.length;i++) cm[yTrue[i]][yPred[i]]++; return cm; }

// ===== Train & Predict =====
async function train(){
  try{
    if(!DATA){ log('⚠️ Dataset missing.'); return; }
    if(MODEL){ MODEL.dispose(); MODEL=null; }
    lossHistory.length=0; valAccHistory.length=0;
    const n = DATA.X.shape[0];
    const {tr, va} = trainValSplit(n, 0.8);
    const Xtr = subset(DATA.X, tr);
    const ytr = subset(DATA.yOne, tr);
    const Xva = subset(DATA.X, va);
    const idxVa = tf.tensor1d(va, 'int32');
    const yvaIdx = tf.gather(DATA.yIdx, idxVa); // exact gather
    idxVa.dispose();

    MODEL = buildModel(DATA.X.shape[1], DATA.labels.length);
    dom('modelState').textContent = 'model: training...';
    dom('predictBtn').disabled = true;
    log(`🚀 Training on ${tr.length} rows, validating on ${va.length} rows…`);

    let lastEva=null;
    await MODEL.fit(Xtr, ytr, {
      epochs: EPOCHS, batchSize: BATCH_SIZE, shuffle: true,
      callbacks: {
        onEpochEnd: async (e, logs) => {
          const epoch = e+1;
          const loss = (logs.loss??0);
          updateLoss(epoch, loss);
          lastEva = await manualValEval(MODEL, Xva, yvaIdx);
          updateValAcc(epoch, lastEva.acc);
          dom('k_epochs').textContent = String(epoch);
          dom('k_acc').textContent = (lastEva.acc*100).toFixed(1) + '%';
          log(`epoch ${epoch}: loss=${loss.toFixed(3)} · val_acc=${(lastEva.acc).toFixed(3)}`);
        }
      }
    });

    // Diagnostics
    const eva = await manualValEval(MODEL, Xva, yvaIdx);
    const roc = computeRocMicro(eva.yTrue, eva.probsFlat, DATA.labels.length);
    ensureRocChart(); renderRoc(roc.fpr, roc.tpr, roc.auc);
    const cm = confusionMatrix(eva.yTrue, eva.preds, DATA.labels.length);
    renderCm(cm, DATA.labels);

    // Cleanup
    Xtr.dispose(); ytr.dispose(); Xva.dispose(); yvaIdx.dispose();
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

function topK(probs,k=3){ const arr=Array.from(probs).map((p,i)=>({i,p})); arr.sort((a,b)=>b.p-a.p); return arr.slice(0,k); }
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
  if(DATA){ DATA.X.dispose?.(); DATA.yOne.dispose?.(); DATA.yIdx.dispose?.(); }
  if(MODEL){ MODEL.dispose(); }
  DATA = null;
  dom('k_rows').textContent = '–';
  dom('k_classes').textContent = '–';
  dom('k_acc').textContent = '–';
  dom('k_epochs').textContent = '–';
  dom('topk').innerHTML = '';
  dom('modelState').textContent = 'model: not trained';
  if(rocChart){ rocChart.destroy(); rocChart=null; }
  if(cmChart){ cmChart.destroy(); cmChart=null; }
  if(lossChart){ lossChart.destroy(); lossChart=null; }
  if(valAccChart){ valAccChart.destroy(); valAccChart=null; }
  lossHistory.length=0; valAccHistory.length=0;
  log('🔁 State reset. Click Train model again.');
}

// ===== Init =====
(function init(){
  dom('model_name').textContent = `Model: ${MODEL_NAME}`;
  try{
    const csv = decodeEmbeddedCSV();
    DATA = parseCSV(csv);
    dom('k_rows').textContent = String(DATA.X.shape[0]);
    dom('k_classes').textContent = String(DATA.labels.length);
    dom('modelState').textContent = 'model: data loaded';
    log(`📦 Loaded ${DATA.X.shape[0]} rows · ${DATA.labels.length} classes`);
  }catch(err){
    log('❌ Failed to load embedded CSV: ' + err.message);
  }
  dom('trainBtn').addEventListener('click', train);
  dom('predictBtn').addEventListener('click', predict);
  dom('resetBtn').addEventListener('click', resetAll);
})();
