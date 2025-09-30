// data.js
// Responsible for loading and parsing u.item and u.data
// Exposes globals: movies, ratings, numUsers, numMovies, userIdToIndex, movieIdToIndex

let movies = [];    // { id: Number, title: String, genres: [String] }
let ratings = [];   // { userId: Number, itemId: Number, rating: Number, timestamp: Number }

// Mappings and counts (filled after parsing)
let userIdToIndex = {};   // original userId -> zero-based index
let movieIdToIndex = {};  // original movieId -> zero-based index
let indexToUserId = [];   // reverse mapping
let indexToMovieId = [];  // reverse mapping
let numUsers = 0;
let numMovies = 0;

/**
 * Load data asynchronously from u.item and u.data (assumed in same folder)
 */
async function loadData() {
  try {
    // Load u.item
    const itemResp = await fetch("u.item");
    if (!itemResp.ok) throw new Error("Failed to load u.item");
    const itemText = await itemResp.text();
    parseItemData(itemText);

    // Load u.data
    const dataResp = await fetch("u.data");
    if (!dataResp.ok) throw new Error("Failed to load u.data");
    const dataText = await dataResp.text();
    parseRatingData(dataText);

    // Build user/movie index mappings from rating data (ensure all users in ratings are included)
    buildMappings();

    console.log(`Loaded ${movies.length} movies and ${ratings.length} ratings`);
  } catch (err) {
    console.error("loadData error:", err);
    const resultEl = document.getElementById && document.getElementById("result");
    if (resultEl) resultEl.innerText = "Error loading data: " + err.message;
  }
}

/**
 * Parse u.item
 * Format: movieId | title | release_date | ... | 19 genre flags
 */
function parseItemData(text) {
  movies = [];

  const genreNames = [
    "Unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
  ];

  const lines = text.split("\n");
  for (const line of lines) {
    if (!line.trim()) continue;
    // u.item uses '|' delimiter
    const parts = line.split("|");
    if (parts.length < 24) continue; // skip malformed
    const id = parseInt(parts[0], 10);
    const title = parts[1];
    // last 19 fields are genre flags
    const genreFlags = parts.slice(-19);
    const genres = [];
    for (let i = 0; i < genreFlags.length; i++) {
      if (genreFlags[i] === "1") genres.push(genreNames[i]);
    }
    movies.push({ id, title, genres });
  }

  // store raw number of movies found
  numMovies = movies.length;
}

/**
 * Parse u.data
 * Format: userId \t itemId \t rating \t timestamp
 */
function parseRatingData(text) {
  ratings = [];

  const lines = text.split("\n");
  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split("\t");
    if (parts.length < 4) continue;
    const userId = parseInt(parts[0], 10);
    const itemId = parseInt(parts[1], 10);
    const rating = parseFloat(parts[2]);
    const timestamp = parseInt(parts[3], 10);
    ratings.push({ userId, itemId, rating, timestamp });
  }
}

/**
 * Build mappings userId<->index and movieId<->index based on ratings & movies
 * This ensures embeddings input indices are contiguous and zero-based.
 */
function buildMappings() {
  userIdToIndex = {};
  movieIdToIndex = {};
  indexToUserId = [];
  indexToMovieId = [];

  const userSet = new Set();
  const movieSet = new Set();

  for (const r of ratings) {
    userSet.add(r.userId);
    movieSet.add(r.itemId);
  }

  // ensure movies array movie ids also present; include any movie ids not in ratings (optional)
  for (const m of movies) movieSet.add(m.id);

  // create user mapping
  const sortedUsers = Array.from(userSet).sort((a, b) => a - b);
  sortedUsers.forEach((uid, idx) => {
    userIdToIndex[uid] = idx;
    indexToUserId[idx] = uid;
  });
  numUsers = sortedUsers.length;

  // create movie mapping
  const sortedMovies = Array.from(movieSet).sort((a, b) => a - b);
  sortedMovies.forEach((mid, idx) => {
    movieIdToIndex[mid] = idx;
    indexToMovieId[idx] = mid;
  });
  numMovies = sortedMovies.length;

  // Log for debugging
  console.log(`numUsers=${numUsers}, numMovies=${numMovies}`);
}
