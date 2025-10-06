// app.js
// Glue: data loading, batching, UI, training loop, projection, test UI
// Depends on two-tower.js (TwoTowerModel) and tf.js

// Globals
let interactions = []; // {userId, itemId, rating, ts}
let items = new Map(); // itemId -> {title, year, genres: [0/1 array]}
let usersToItems = new Map(); // userId -> [{itemId, rating, ts}]
let userIdToIndex = {}, itemIdToIndex = {}, indexToUserId = [], indexToItemId = [];
let numUsers=0, numItems=0;
let genreDim = 19;
let model = null;
let genreMatrix = null; // array of arrays [numItems][genreDim]
let logEl, statusEl;
let lossHistory = [];

// UI bindings
document.addEventListener('DOMContentLoaded', ()=> {
  logEl = document.getElementById('log');
  statusEl = document.getElementById('status');
  document.getElementById('btn-load').onclick = loadData;
  document.getElementById('btn-train').onclick = trainHandler;
  document.getElementById('btn-test').onclick = testHandler;
  document.getElementById('btn-clear-log').onclick = ()=>{ logEl.textContent=''; };
  drawLossChart(); // initialize
});

// UTIL: log message
function log(msg){
  const t = new Date().toLocaleTimeString();
  logEl.textContent += `[${t}] ${msg}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

// ----------- DATA LOADING -------------
async function loadData(){
  try {
    statusEl.innerText = "Loading data...";
    log("Fetching /data/u.item and /data/u.data ...");
    // fetch items
    const itemResp = await fetch('data/u.item');
    if(!itemResp.ok) throw new Error("Failed to fetch data/u.item");
    const itemText = await itemResp.text();
    parseItems(itemText);

    const dataResp = await fetch('data/u.data');
    if(!dataResp.ok) throw new Error("Failed to fetch data/u.data");
    const dataText = await dataResp.text();
    parseInteractions(dataText);

    buildIndexers();
    buildGenreMatrix();

    populateSampleRow();
    populateConfigDropdowns();

    statusEl.innerText = `Loaded ${interactions.length} interactions, ${numUsers} users, ${numItems} items. Ready.`;
    log(`Loaded interactions=${interactions.length}, users=${numUsers}, items=${numItems}`);
    document.getElementById('btn-train').disabled = false;
  } catch (err) {
    statusEl.innerText = "Load failed: " + err.message;
    log("Error: " + err.stack || err);
    console.error(err);
  }
}

function parseItems(text){
  items = new Map();
  const genreNames = [
    "Unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy",
    "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
  ];
  const lines = text.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length < 2) continue;
    const id = parseInt(parts[0],10);
    let title = parts[1];
    // attempt to extract year from title "Movie (1995)"
    let year = null;
    const m = title.match(/\((\d{4})\)/);
    if (m) { year = parseInt(m[1],10); }
    // genre flags are last 19 fields
    const flags = parts.slice(-19);
    const genres = flags.map(f => f === '1' ? 1 : 0);
    items.set(id, {id, title, year, genres});
  }
}

function parseInteractions(text){
  interactions = [];
  usersToItems = new Map();
  const lines = text.split('\n');
  for (const line of lines) {
    if (!line.trim()) continue;
    const p = line.split('\t');
    if (p.length < 3) continue;
    const userId = parseInt(p[0],10);
    const itemId = parseInt(p[1],10);
    const rating = parseFloat(p[2]);
    const ts = p[3] ? parseInt(p[3],10) : 0;
    interactions.push({userId, itemId, rating, ts});
    if (!usersToItems.has(userId)) usersToItems.set(userId, []);
    usersToItems.get(userId).push({itemId, rating, ts});
  }
}

// build index maps (0-based) using users/items present
function buildIndexers(){
  // users from interactions
  const userSet = new Set(interactions.map(r=>r.userId));
  const itemSet = new Set(interactions.map(r=>r.itemId));
  // ensure items in items map also included
  for (const k of items.keys()) itemSet.add(k);

  indexToUserId = Array.from(userSet).sort((a,b)=>a-b);
  indexToItemId = Array.from(itemSet).sort((a,b)=>a-b);
  userIdToIndex = {}; itemIdToIndex = {};
  indexToUserId.forEach((u,i)=> userIdToIndex[u]=i);
  indexToItemId.forEach((m,i)=> itemIdToIndex[m]=i);
  numUsers = indexToUserId.length;
  numItems = indexToItemId.length;
}

function buildGenreMatrix(){
  genreMatrix = new Array(numItems);
  for (let i=0;i<numItems;i++){
    const origId = indexToItemId[i];
    const rec = items.get(origId);
    if (rec && rec.genres) genreMatrix[i] = rec.genres.slice(0, genreDim).map(x=> x?1:0);
    else genreMatrix[i] = new Array(genreDim).fill(0);
  }
}

// populate sample row visuals
function populateSampleRow(){
  const row = document.getElementById('sample-row'); row.innerHTML='';
  const sample = indexToItemId.slice(0,20);
  for (const mid of sample){
    const it = items.get(mid);
    const card = document.createElement('div'); card.className='movie-card';
    card.style.minWidth='130px';
    card.style.marginRight='8px';
    const img = document.createElement('img'); img.src=`https://picsum.photos/200/260?random=${mid}`; img.style.width='100%';
    const div = document.createElement('div'); div.className='movie-info';
    const h = document.createElement('h4'); h.textContent = it ? it.title : `#${mid}`;
    div.appendChild(h);
    card.appendChild(img); card.appendChild(div);
    row.appendChild(card);
  }
}

// populate some config UIs (if needed)
function populateConfigDropdowns(){
  // nothing for now; placeholder if future UI needs
}

// ----------- TRAINING / BATCHING -------------
function buildBatches(maxInteractions){
  // Shuffle interactions, limit to maxInteractions
  const maxN = Math.min(maxInteractions || 80000, interactions.length);
  const arr = interactions.slice(0, maxN);
  // shuffle
  for (let i=arr.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [arr[i],arr[j]]=[arr[j],arr[i]]; }
  // map to indices
  const userIdx = arr.map(r=> userIdToIndex[r.userId]);
  const itemIdx = arr.map(r=> itemIdToIndex[r.itemId]);
  const ratingsArr = arr.map(r=> r.rating);
  // return arrays
  return {userIdx, itemIdx, ratings: ratingsArr};
}

// Higher-level train handler
async function trainHandler(){
  try {
    document.getElementById('btn-train').disabled = true;
    const maxI = parseInt(document.getElementById('cfg-max').value || '80000',10);
    const embDim = parseInt(document.getElementById('cfg-dim').value || '32',10);
    const epochs = parseInt(document.getElementById('cfg-epochs').value || '6',10);
    const bsz = parseInt(document.getElementById('cfg-bsz').value || '256',10);
    const lossType = document.getElementById('cfg-loss').value || 'inbatch';
    const useMLP = document.getElementById('cfg-mlp').checked;

    statusEl.innerText = 'Preparing batches...';
    log(`Training config: maxI=${maxI}, embDim=${embDim}, epochs=${epochs}, bsz=${bsz}, loss=${lossType}, MLP=${useMLP}`);

    const {userIdx, itemIdx} = buildBatches(maxI);

    // Build array of batches (simple segmentation)
    const N = userIdx.length;
    const batches = [];
    for (let i=0;i<N;i+=bsz){
      const j = Math.min(N, i+bsz);
      batches.push({
        userIdx: userIdx.slice(i,j),
        posIdx: itemIdx.slice(i,j)
      });
    }
    log(`Built ${batches.length} batches (${N} interactions)`);

    // Create model
    model = new TwoTowerModel({
      numUsers, numItems, embDim, useMLP, genreDim, lr:0.001, lossType
    });

    // UI: enable Test only after training finishes
    document.getElementById('btn-test').disabled = true;
    lossHistory = [];
    drawLossChart();

    // Training loop
    for (let e=0;e<epochs;e++){
      let epochLoss = 0;
      let count = 0;
      statusEl.innerText = `Training epoch ${e+1}/${epochs} ...`;
      for (let b=0;b<batches.length;b++){
        const batch = batches[b];
        // prepare genre features for positives if MLP used
        let genrePos = null;
        if (useMLP){
          genrePos = batch.posIdx.map(pi => genreMatrix[pi]);
        }
        // For BPR, sample negatives per positive
        let negIdx = null;
        if (lossType === 'bpr'){
          negIdx = batch.posIdx.map(()=> Math.floor(Math.random()*numItems));
        }

        const loss = await model.trainStep({userIdx: batch.userIdx, posIdx: batch.posIdx, negIdx}, {genrePos, genreDim});
        epochLoss += loss; count++;
        lossHistory.push({x: e + b/batches.length, y: loss});
        if ((b % Math.max(1,Math.floor(batches.length/6)))===0) {
          drawLossChart();
          log(`Epoch ${e+1} batch ${b+1}/${batches.length} loss=${loss.toFixed(4)}`);
          statusEl.innerText = `Epoch ${e+1}/${epochs} — batch ${b+1}/${batches.length} — loss ${loss.toFixed(4)}`;
          await tf.nextFrame();
        }
      }
      const avg = epochLoss/count;
      log(`Epoch ${e+1} completed. avg loss=${avg.toFixed(4)}`);
      drawLossChart();
    }

    statusEl.innerText = "Training complete.";
    document.getElementById('btn-test').disabled = false;
    log('Training finished. Preparing embedding projection...');
    await projectAndDraw(); // projection after training
  } catch (err) {
    statusEl.innerText = 'Training error: ' + err.message;
    log('Error during training: ' + (err.stack || err));
    console.error(err);
  } finally {
    document.getElementById('btn-train').disabled = false;
  }
}

// ----------- PREDICTION / TEST -------------
async function testHandler(){
  if (!model) { statusEl.innerText='Model not ready.'; return; }
  statusEl.innerText = 'Selecting random user with >=20 ratings...';
  // choose qualified users
  const qualified = [];
  for (const [u, arr] of usersToItems.entries()){
    if (arr.length >= 20) qualified.push(u);
  }
  if (qualified.length===0){ statusEl.innerText='No user with >=20 ratings found.'; return; }
  const userId = qualified[Math.floor(Math.random()*qualified.length)];
  const userIdx = userIdToIndex[userId];

  // Left: top-10 historically rated movies by rating then recency
  const hist = usersToItems.get(userId).slice();
  hist.sort((a,b)=> b.rating - a.rating || b.ts - a.ts);
  const topHist = hist.slice(0,20).map(h => ({itemId:h.itemId, rating:h.rating, ts:h.ts}));

  // Right: compute scores for all items and take top-10 excluding already rated
  statusEl.innerText = 'Computing recommendations (this may take a moment)...';
  log(`Computing scores for user ${userId} (idx ${userIdx})`);
  // model.scoresForUserIndex returns Float32Array scores
  const scores = await model.scoresForUserIndex(userIdx, genreMatrix, 1024);
  const ratedSet = new Set(usersToItems.get(userId).map(x=>x.itemId));
  const scoredList = [];
  for (let i=0;i<scores.length;i++){
    const origId = indexToItemId[i];
    if (ratedSet.has(origId)) continue;
    scoredList.push({itemIndex:i, score: scores[i]});
  }
  scoredList.sort((a,b)=> b.score - a.score);
  const topRec = scoredList.slice(0,10).map(s=> ({itemId: indexToItemId[s.itemIndex], score:s.score}));

  // Optional DL comparison: here our model already includes MLP if selected; for comparison we can compute
  // baseline dot-product using raw embeddings (without MLP) to compare. We'll compute top-10 baseline.
  const baselineScores = await baselineScoresForUser(userIdx);
  const baselineList = [];
  for (let i=0;i<baselineScores.length;i++){
    const origId = indexToItemId[i];
    if (ratedSet.has(origId)) continue;
    baselineList.push({itemIndex:i, score: baselineScores[i]});
  }
  baselineList.sort((a,b)=> b.score - a.score);
  const topBaseline = baselineList.slice(0,10).map(s=> ({itemId: indexToItemId[s.itemIndex], score:s.score}));

  // Render side-by-side table: left historic top 10, middle model recs, right baseline or DL comparison
  renderComparisonTable(topHist.slice(0,10), topRec, topBaseline);
  statusEl.innerText = `Test completed for user ${userId}.`;
}

async function baselineScoresForUser(userIdx){
  // compute simple dot using embedding matrices (no MLP)
  // user embedding
  const uIdxT = tf.tensor1d([userIdx],'int32');
  const uEmb = model.userForward(uIdxT); // [1,d]
  const itemEmbAll = model.itemEmb; // tf.variable [numItems,d]
  const logits = tf.matMul(uEmb, itemEmbAll, false, true); // [1, numItems]
  const arr = await logits.data();
  uIdxT.dispose(); uEmb.dispose(); logits.dispose();
  return Array.from(arr);
}

function renderComparisonTable(histList, recList, baselineList){
  const wrap = document.getElementById('tables-wrap'); wrap.innerHTML = '';
  const tbl = document.createElement('table');
  const header = document.createElement('tr');
  header.innerHTML = `<th>Top-10 Historically Rated</th><th>Model Recommendations (DL)</th><th>Baseline (Dot Only)</th>`;
  tbl.appendChild(header);
  for (let i=0;i<10;i++){
    const tr = document.createElement('tr');
    const left = histList[i] ? items.get(histList[i].itemId).title + ` (${histList[i].rating})` : '';
    const mid = recList[i] ? items.get(recList[i].itemId).title + ` (${recList[i].score.toFixed(2)})` : '';
    const right = baselineList[i] ? items.get(baselineList[i].itemId).title + ` (${baselineList[i].score.toFixed(2)})` : '';
    tr.innerHTML = `<td>${escapeHtml(left)}</td><td>${escapeHtml(mid)}</td><td>${escapeHtml(right)}</td>`;
    tbl.appendChild(tr);
  }
  wrap.appendChild(tbl);
}

// ----------- PROJECTION (PCA) -------------
async function projectAndDraw(){
  statusEl.innerText = 'Projecting item embeddings (PCA) for visualization...';
  // sample up to 1000 items
  const sampleN = Math.min(1000, numItems);
  const indices = [];
  // sample uniformly
  for (let i=0;i<sampleN;i++) indices.push(Math.floor(i*(numItems/sampleN)));
  // build item embedding matrix for these indices (apply MLP processing if used)
  // We'll gather embeddings via model.itemForward for a batch of indices
  const idxT = tf.tensor1d(indices,'int32');
  let itemEmb;
  if (model.useMLP){
    // need genre features slice
    const genreSlice = indices.map(i=> genreMatrix[i]);
    const genreT = tf.tensor2d(genreSlice);
    itemEmb = model.itemForward(idxT, genreT); // [sampleN, d]
    genreT.dispose();
  } else {
    itemEmb = model.itemForward(idxT, null); // [sampleN,d]
  }
  // Perform SVD on centered embeddings (X = U S V^T). We'll use tf.svd
  const mean = itemEmb.mean(0);
  const centered = itemEmb.sub(mean);
  // compute SVD (tf.svd exists in tfjs). fallback: use simple power iteration on covariance if not available.
  let U, S, V;
  try {
    const svdRes = tf.svd(centered, true);
    U = svdRes.u; S = svdRes.s; V = svdRes.v;
    // project: scores = centered @ V[:,0:2]
    const V2 = V.slice([0,0],[V.shape[0],2]); // [d,2]
    const proj = tf.matMul(centered, V2); // [sampleN,2]
    const arr = await proj.array();
    drawProjection(indices, arr);
    proj.dispose(); V2.dispose(); U.dispose(); S.dispose(); V.dispose();
  } catch (err){
    // fallback to simple PCA via covariance eigenvectors computed with power iteration on covariance matrix (dxd)
    log('SVD not available; using fallback PCA (may be slower).');
    const X = centered; // [n,d]
    const Xt = X.transpose(); // [d,n]
    const cov = tf.matMul(Xt, X).div(sampleN-1); // [d,d]
    // power iteration to find top 2 eigenvectors
    const d = cov.shape[0];
    let eigs = [];
    let covT = cov;
    for (let k=0;k<2;k++){
      let v = tf.randomNormal([d,1]);
      for (let iter=0;iter<60;iter++){
        v = tf.matMul(covT, v);
        const norm = v.norm();
        v = v.div(norm);
      }
      eigs.push(v.reshape([d]));
      // deflate
      const lambda = tf.dot(v.reshape([d]), tf.matMul(covT, v).reshape([d]));
      covT = covT.sub(tf.matMul(v, v.transpose()).mul(lambda));
    }
    const Vmat = tf.stack(eigs,1); // [d,2]
    const proj = tf.matMul(centered, Vmat); // [n,2]
    const arr = await proj.array();
    drawProjection(indices, arr);
    proj.dispose(); cov.dispose(); Xt.dispose();
  }
  itemEmb.dispose(); idxT.dispose(); mean.dispose(); centered.dispose();
  statusEl.innerText = 'Projection drawn.';
}

function drawProjection(indices, coords){
  const canvas = document.getElementById('proj-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // normalize coords to canvas
  const xs = coords.map(p=>p[0]); const ys = coords.map(p=>p[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  for (let i=0;i<coords.length;i++){
    const x = 20 + ((coords[i][0]-minx)/(maxx-minx || 1))*(canvas.width-40);
    const y = 20 + ((coords[i][1]-miny)/(maxy-miny || 1))*(canvas.height-40);
    ctx.fillStyle = '#e50914';
    ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
    // hover not implemented here; for simplicity we don't add hover listeners on canvas
  }
  log(`Plotted ${coords.length} item embeddings.`);
}

// ----------- LOSS CHART (canvas simple plot) -------------
function drawLossChart(){
  const canvas = document.getElementById('loss-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#0b0b0b'; ctx.fillRect(0,0,canvas.width,canvas.height);
  if (lossHistory.length===0) return;
  const padding = 30;
  const w = canvas.width - 2*padding;
  const h = canvas.height - 2*padding;
  const xs = lossHistory.map((p,i)=> padding + (i/(lossHistory.length-1))*w);
  const ysvals = lossHistory.map(p=>p.y);
  const miny = Math.min(...ysvals), maxy = Math.max(...ysvals);
  // draw axis
  ctx.strokeStyle='#444'; ctx.beginPath(); ctx.moveTo(padding, padding); ctx.lineTo(padding, padding+h); ctx.lineTo(padding+w, padding+h); ctx.stroke();
  // draw line
  ctx.strokeStyle='#46d369'; ctx.lineWidth=2; ctx.beginPath();
  for (let i=0;i<lossHistory.length;i++){
    const x = xs[i];
    const y = padding + h - ((lossHistory[i].y - miny)/(maxy-miny || 1))*h;
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
  // draw last loss value
  ctx.fillStyle='#ccc'; ctx.font='12px Arial';
  ctx.fillText('Loss: ' + lossHistory[lossHistory.length-1].y.toFixed(4), padding+6, padding+12);
}

// ------------- Utilities -------------
function escapeHtml(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
