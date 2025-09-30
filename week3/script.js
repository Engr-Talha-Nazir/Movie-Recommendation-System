// script.js
// Uses TensorFlow.js to build, train, and use a Matrix Factorization model in-browser.
// Depends on data.js (movies, ratings, mappings, numUsers, numMovies)

let model = null;     // trained tf.Model
const latentDim = 32; // embedding size (tunable)
let isTraining = false;

window.onload = async function () {
  try {
    // Load dataset
    await loadData();

    // Populate dropdowns
    populateUserDropdown();
    populateMovieDropdown();
    populateSampleRow();

    // Start training
    await trainModel();

  } catch (err) {
    console.error(err);
    const r = document.getElementById("result");
    if (r) r.innerText = "Error initializing app: " + err.message;
  }

  // Attach handlers
  document.getElementById("predict-btn").addEventListener("click", predictRating);
  const searchInput = document.getElementById("search-input");
  if (searchInput) searchInput.addEventListener("input", handleSearch);
};

/* ----------------------
   UI population helpers
   ---------------------- */

function populateUserDropdown() {
  const select = document.getElementById("user-select");
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.disabled = true;
  placeholder.selected = true;
  placeholder.textContent = "-- Select user --";
  select.appendChild(placeholder);

  // Use indexToUserId from data.js (if available)
  if (typeof indexToUserId !== "undefined" && indexToUserId.length) {
    for (let idx = 0; idx < indexToUserId.length; idx++) {
      const opt = document.createElement("option");
      opt.value = idx; // store zero-based index
      opt.textContent = `User ${indexToUserId[idx]} (idx ${idx})`;
      select.appendChild(opt);
    }
  } else {
    // fallback: list distinct users from ratings
    const users = Array.from(new Set(ratings.map(r => r.userId))).sort((a,b)=>a-b);
    users.forEach((u, i) => {
      const opt = document.createElement("option");
      opt.value = i;
      opt.textContent = `User ${u}`;
      select.appendChild(opt);
    });
  }
}

function populateMovieDropdown() {
  const select = document.getElementById("movie-select");
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.disabled = true;
  placeholder.selected = true;
  placeholder.textContent = "-- Select a movie --";
  select.appendChild(placeholder);

  // Sort movies alphabetically
  const sorted = [...movies].sort((a,b) => a.title.localeCompare(b.title));
  for (const m of sorted) {
    // map original movie id to its index
    const movieIndex = (typeof movieIdToIndex !== "undefined" && movieIdToIndex[m.id] !== undefined) ? movieIdToIndex[m.id] : m.id;
    const opt = document.createElement("option");
    opt.value = movieIndex;
    opt.textContent = m.title;
    select.appendChild(opt);
  }
}

function populateSampleRow() {
  const row = document.getElementById("sample-row");
  if (!row) return;
  row.innerHTML = "";
  const sample = movies.slice(0, 12);
  sample.forEach(m => {
    const idx = movieIdToIndex[m.id];
    const card = makeMovieCard({ ...m, idx }, null);
    row.appendChild(card);
  });
}

/* ----------------------
   Model creation
   ---------------------- */

/**
 * createModel(numUsers, numMovies, latentDim)
 *
 * Architecture:
 * - Two inputs: userInput and movieInput (shape [1])
 * - Two embeddings: userEmbedding (numUsers x latentDim), movieEmbedding (numMovies x latentDim)
 * - Flatten embeddings to get userVec and movieVec
 * - Dot product between userVec and movieVec (axis=1) -> scalar prediction
 * - Optionally add user & movie biases and a final dense layer for slight adjustment
 */
function createModel(numUsersLocal, numMoviesLocal, latentDimLocal) {
  // Inputs: integer id for user and movie
  const userInput = tf.input({shape: [1], dtype: 'int32', name: 'userInput'});
  const movieInput = tf.input({shape: [1], dtype: 'int32', name: 'movieInput'});

  // Embedding layers
  // inputDim must be >= maximum index + 1. We used contiguous zero-based indices in data.js.
  const userEmbedding = tf.layers.embedding({
    inputDim: numUsersLocal,
    outputDim: latentDimLocal,
    inputLength: 1,
    name: 'userEmbedding'
  });

  const movieEmbedding = tf.layers.embedding({
    inputDim: numMoviesLocal,
    outputDim: latentDimLocal,
    inputLength: 1,
    name: 'movieEmbedding'
  });

  // Bias embeddings (optional) to learn per-user and per-movie offsets
  const userBiasEmbedding = tf.layers.embedding({
    inputDim: numUsersLocal,
    outputDim: 1,
    inputLength: 1,
    name: 'userBias'
  });

  const movieBiasEmbedding = tf.layers.embedding({
    inputDim: numMoviesLocal,
    outputDim: 1,
    inputLength: 1,
    name: 'movieBias'
  });

  // Get latent vectors
  const userVec = tf.layers.flatten().apply(userEmbedding.apply(userInput));   // shape [batch, latentDim]
  const movieVec = tf.layers.flatten().apply(movieEmbedding.apply(movieInput)); // shape [batch, latentDim]

  // Dot product between user and movie vectors
  const dot = tf.layers.dot({ axes: 1 }).apply([userVec, movieVec]); // shape [batch, 1]

  // Get biases and flatten
  const userBias = tf.layers.flatten().apply(userBiasEmbedding.apply(userInput)); // [batch,1]
  const movieBias = tf.layers.flatten().apply(movieBiasEmbedding.apply(movieInput)); // [batch,1]

  // Sum dot + userBias + movieBias
  const added = tf.layers.add().apply([dot, userBias, movieBias]); // [batch,1]

  // Final dense transform (linear) to allow the model to shift/scale output a little
  const output = tf.layers.dense({ units: 1, activation: 'linear', name: 'rating' }).apply(added);

  const mfModel = tf.model({
    inputs: [userInput, movieInput],
    outputs: output,
    name: 'matrixFactorizationModel'
  });

  return mfModel;
}

/* ----------------------
   Training
   ---------------------- */

async function trainModel() {
  const resultEl = document.getElementById("result");
  if (isTraining) {
    resultEl.innerText = "Training already in progress...";
    return;
  }

  resultEl.innerText = "Preparing training data...";
  await tf.nextFrame();

  // Build tensors for training
  // We need arrays of user indices (zero-based), movie indices (zero-based), and ratings (float)
  const userIdx = [];
  const movieIdx = [];
  const ratingVals = [];

  for (const r of ratings) {
    // r.userId and r.itemId are original ids; map to indices
    const uidx = userIdToIndex[r.userId];
    const midx = movieIdToIndex[r.itemId];
    if (uidx === undefined || midx === undefined) continue;
    userIdx.push(uidx);
    movieIdx.push(midx);
    ratingVals.push(r.rating);
  }

  if (userIdx.length === 0) {
    resultEl.innerText = "No training data found.";
    return;
  }

  // Convert to tensors
  const userTensor = tf.tensor2d(userIdx, [userIdx.length, 1], 'int32');
  const movieTensor = tf.tensor2d(movieIdx, [movieIdx.length, 1], 'int32');
  const ratingTensor = tf.tensor2d(ratingVals, [ratingVals.length, 1], 'float32');

  resultEl.innerText = `Creating model with ${numUsers} users and ${numMovies} movies...`;
  await tf.nextFrame();

  model = createModel(numUsers, numMovies, latentDim);

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError'
  });

  resultEl.innerText = "Training model in browser (this may take a while)...";
  isTraining = true;

  const epochs = 8;
  const batchSize = 64;

  // Train with onEpochEnd callback to update UI
  await model.fit(
    [userTensor, movieTensor],
    ratingTensor,
    {
      epochs,
      batchSize,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          resultEl.innerText = `Training: epoch ${epoch+1}/${epochs} — loss=${(logs.loss || 0).toFixed(4)}`;
          await tf.nextFrame();
        }
      }
    }
  );

  isTraining = false;
  resultEl.innerText = `Training complete. Model ready — you can predict ratings now.`;
  await tf.nextFrame();

  // dispose tensors used for training to free memory
  userTensor.dispose();
  movieTensor.dispose();
  ratingTensor.dispose();
}

/* ----------------------
   Prediction
   ---------------------- */

async function predictRating() {
  const resultEl = document.getElementById("result");
  if (!model) {
    resultEl.innerText = "Model is not ready yet. Please wait for training to complete.";
    return;
  }

  const userSelect = document.getElementById("user-select");
  const movieSelect = document.getElementById("movie-select");

  const userVal = userSelect.value;
  const movieVal = movieSelect.value;

  if (!userVal || !movieVal) {
    resultEl.innerHTML = "<span style='color:#f6c0c0;'>Please select both a user and a movie.</span>";
    return;
  }

  const userIdx = parseInt(userVal, 10);
  const movieIdx = parseInt(movieVal, 10);
  if (Number.isNaN(userIdx) || Number.isNaN(movieIdx)) {
    resultEl.innerText = "Invalid selection.";
    return;
  }

  // Build input tensors (shape [1,1], int32)
  const uT = tf.tensor2d([userIdx], [1,1], 'int32');
  const mT = tf.tensor2d([movieIdx], [1,1], 'int32');

  // Predict
  const predTensor = model.predict([uT, mT]);
  const predArray = await predTensor.data();
  let predicted = predArray[0];

  // Optionally clip to rating range (MovieLens ratings are 1..5)
  if (predicted < 1) predicted = 1;
  if (predicted > 5) predicted = 5;

  // Show result
  const movieOriginalId = indexToMovieId[movieIdx];
  // find movie title
  const mv = movies.find(x => x.id === movieOriginalId) || {};
  const movieTitle = mv.title || `movie id ${movieOriginalId}`;

  resultEl.innerHTML = `<strong>Predicted rating for user ${indexToUserId[userIdx]} → "${escapeHtml(movieTitle)}":</strong> <span style="color:#46d369;">${predicted.toFixed(2)}</span> (scale 1-5)`;

  // Dispose tensors
  uT.dispose();
  mT.dispose();
  predTensor.dispose();
}

/* ----------------------
   Utilities & small UI helpers
   ---------------------- */

function makeMovieCard(movie, similarity = null) {
  const card = document.createElement('div');
  card.className = 'movie-card';

  // Poster placeholder using movie id (deterministic)
  const posterUrl = `https://picsum.photos/300/420?random=${encodeURIComponent(movie.id || movie.idx || Math.random())}`;

  card.innerHTML = `
    <img class="movie-poster" src="${posterUrl}" alt="${escapeHtml(movie.title)}">
    <div class="movie-info">
      <h3>${escapeHtml(movie.title)}</h3>
      ${similarity !== null ? `<div class="match">${similarity}% Match</div>` : ''}
      <div class="genres">${(movie.genres||[]).slice(0,4).map(g=>`<span class="genre-tag">${escapeHtml(g)}</span>`).join(' ')}</div>
    </div>
  `;
  return card;
}

// search handler updates the sample row
function handleSearch(e) {
  const q = String(e.target.value || '').trim().toLowerCase();
  const row = document.getElementById('sample-row');
  row.innerHTML = '';
  const found = movies.filter(m => m.title.toLowerCase().includes(q)).slice(0, 20);
  (q ? found : movies.slice(0, 12)).forEach(m => {
    const idx = movieIdToIndex[m.id];
    row.appendChild(makeMovieCard({ ...m, idx }, null));
  });
}

function escapeHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
