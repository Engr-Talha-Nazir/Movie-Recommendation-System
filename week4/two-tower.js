// two-tower.js
// Two-Tower model core for TF.js
// Implements embedding tables, optional MLP for item tower, scoring, and training step.
// Comments above blocks explain design decisions.

// TwoTowerModel class encapsulates parameters and operations.
// - userEmbedding: variable [numUsers, embDim]
// - itemEmbedding: variable [numItems, embDim]
// - optional item MLP: to incorporate item genre features (concat with embedding and pass through dense layers)
// - scoring: dot product between userEmb and itemEmb (or processed item features)
// - training supports two losses: in-batch softmax (sampled softmax using batch items as negatives) and BPR.

class TwoTowerModel {
  constructor(config) {
    this.numUsers = config.numUsers;
    this.numItems = config.numItems;
    this.embDim = config.embDim || 32;
    this.useMLP = !!config.useMLP;
    this.mlpHidden = config.mlpHidden || 64;
    this.lr = config.lr || 0.001;
    this.lossType = config.lossType || 'inbatch'; // 'inbatch' or 'bpr'
    this.optimizer = tf.train.adam(this.lr);

    // Initialize embeddings: small random normal
    this.userEmb = tf.variable(tf.randomNormal([this.numUsers, this.embDim], 0, 0.05), true, 'userEmb');
    this.itemEmb = tf.variable(tf.randomNormal([this.numItems, this.embDim], 0, 0.05), true, 'itemEmb');

    // Optional biases
    this.userBias = tf.variable(tf.zeros([this.numUsers, 1]), true, 'userBias');
    this.itemBias = tf.variable(tf.zeros([this.numItems, 1]), true, 'itemBias');

    // MLP for item features (genres). We'll create weights if useMLP true.
    if (this.useMLP) {
      // input: embDim + genreDim -> hidden -> output embDim (project back to embDim)
      // We'll create simple dense layers with variables
      this.genreDim = config.genreDim || 19;
      const inDim = this.embDim + this.genreDim;
      this.W1 = tf.variable(tf.randomNormal([inDim, this.mlpHidden], 0, 0.05));
      this.b1 = tf.variable(tf.zeros([this.mlpHidden]));
      this.W2 = tf.variable(tf.randomNormal([this.mlpHidden, this.embDim], 0, 0.05));
      this.b2 = tf.variable(tf.zeros([this.embDim]));
    }
  }

  // Gather user embeddings for indices tensor shape [batch,1] or [batch]
  userForward(userIdx) {
    // userIdx: int32 tensor shape [batch] or [batch,1]
    const idx = userIdx.reshape([-1]).toInt();
    return tf.gather(this.userEmb, idx); // shape [batch, embDim]
  }

  // Gather item embeddings (base) and optionally process with MLP using genre features
  // itemIdx: [batch] int tensor. genreFeat: optional float tensor [batch, genreDim]
  itemForward(itemIdx, genreFeat = null) {
    const idx = itemIdx.reshape([-1]).toInt();
    let emb = tf.gather(this.itemEmb, idx); // [batch, embDim]
    if (this.useMLP && genreFeat) {
      // concatenate embedding and genres
      const concat = tf.concat([emb, genreFeat], 1); // [batch, embDim+genreDim]
      const h = tf.relu(tf.add(tf.matMul(concat, this.W1), this.b1)); // [batch, mlpHidden]
      const out = tf.add(tf.matMul(h, this.W2), this.b2); // [batch, embDim]
      emb.dispose();
      return out; // [batch, embDim]
    } else {
      return emb;
    }
  }

  // Compute dot scores between userEmb [batch,embDim] and itemEmb [batch,embDim] or itemEmbedding matrix
  // If itemEmbMatrix provided [numItems, embDim], result [batch, numItems]; otherwise dot per row -> [batch,1]
  scorePairwise(userEmb, itemEmb) {
    // userEmb [batch, d], itemEmb [batch, d] -> elementwise dot and sum -> [batch,1]
    const prod = tf.mul(userEmb, itemEmb);
    const s = tf.sum(prod, 1).reshape([-1, 1]); // [batch,1]
    prod.dispose();
    return s;
  }

  // Compute batch logits U @ I^T (for in-batch softmax negatives)
  // userEmb [B, d], itemEmbBatch [B, d] -> logits [B, B]
  batchLogits(userEmb, itemEmbBatch) {
    return tf.matMul(userEmb, itemEmbBatch, false, true); // [B, B]
  }

  // Compute user bias and item bias lookups
  userBiasLookup(userIdx) {
    const idx = userIdx.reshape([-1]).toInt();
    return tf.gather(this.userBias, idx).reshape([-1,1]); // [B,1]
  }
  itemBiasLookup(itemIdx) {
    const idx = itemIdx.reshape([-1]).toInt();
    return tf.gather(this.itemBias, idx).reshape([-1,1]); // [B,1]
  }

  // Single training step: accepts batch tensors (userIdx [B], posItemIdx [B], genreFeatPos [B,gd])
  // If lossType=='inbatch' will compute logits = U@I^T + biases and compute softmax crossentropy with labels = diagonal
  // If lossType=='bpr' will sample negative items idxNeg [B] and compute BPR loss
  async trainStep(batch, extra) {
    // batch: {userIdx: Int32Array, posIdx: Int32Array, negIdx?: Int32Array}
    // extra: {genrePos?: Float32Array2D, genreNeg?: Float32Array2D, itemEmbAll?: tf.Tensor}
    const { userIdx, posIdx, negIdx } = batch;
    const B = userIdx.length;
    const userT = tf.tensor1d(userIdx, 'int32');
    const posT = tf.tensor1d(posIdx, 'int32');

    // convert genre features if provided
    const genrePosT = extra && extra.genrePos ? tf.tensor2d(extra.genrePos, [B, extra.genreDim]) : null;
    const genreNegT = extra && extra.genreNeg ? (negIdx ? tf.tensor2d(extra.genreNeg, [B, extra.genreDim]) : null) : null;

    const self = this;
    const lossScalar = await this.optimizer.minimize(() => {
      // Forward pass
      const uEmb = self.userForward(userT);      // [B, d]
      const iPosEmb = self.itemForward(posT, genrePosT); // [B, d]
      const uBias = self.userBiasLookup(userT); // [B,1]
      const iPosBias = self.itemBiasLookup(posT); // [B,1]

      if (self.lossType === 'inbatch') {
        // logits [B,B] = U @ I_pos^T + uBias + iBias^T
        const logits = self.batchLogits(uEmb, iPosEmb); // [B,B]
        // Add biases: uBias broadcast + iPosBias transposed
        const uBiasMat = tf.add(logits, uBias); // broadcasts [B,1] -> [B,B]
        const iBiasRow = iPosBias.transpose(); // [1,B]
        const logitsWithBias = tf.add(uBiasMat, iBiasRow); // [B,B]

        // labels: identity matrix (each example's positive is diagonal)
        // For tf.losses.softmaxCrossEntropy we provide oneHot labels of shape [B,B]
        const labels = tf.oneHot(tf.range(0, B, 1, 'int32'), B); // [B,B]
        const soft = tf.losses.softmaxCrossEntropy(labels, logitsWithBias);
        // cleanup
        labels.dispose();
        uBiasMat.dispose();
        iBiasRow.dispose();
        logitsWithBias.dispose();
        logits.dispose();

        // free embeddings/biases
        uEmb.dispose(); iPosEmb.dispose(); uBias.dispose(); iPosBias.dispose();
        return soft;
      } else {
        // BPR-style: need negative samples
        if (!negIdx || negIdx.length !== B) {
          // fallback: sample random negatives inside loss (not ideal) - here we pick random ints
          // but we prefer passing negIdx in batch
          console.warn('BPR requested but no negIdx provided; sampling random negatives.');
        }
        const negT = negIdx ? tf.tensor1d(negIdx, 'int32') : tf.randomUniform([B], 0, self.numItems, 'int32');
        const iNegEmb = self.itemForward(negT, genreNegT); // [B,d]
        const iNegBias = self.itemBiasLookup(negT); // [B,1]

        const posScores = tf.add(self.scorePairwise(uEmb, iPosEmb), tf.add(uBias, iPosBias)); // [B,1]
        const negScores = tf.add(self.scorePairwise(uEmb, iNegEmb), tf.add(uBias, iNegBias));     // [B,1]

        // BPR loss = -log(sigmoid(pos - neg))
        const x = tf.sub(posScores, negScores);
        const losses = tf.neg(tf.log(tf.sigmoid(x).add(1e-8)));
        const meanLoss = tf.mean(losses);

        // cleanup
        uEmb.dispose(); iPosEmb.dispose(); uNegEmb = null;
        iNegEmb.dispose(); posScores.dispose(); negScores.dispose();
        uBias.dispose(); iPosBias.dispose(); iNegBias.dispose();
        negT.dispose();
        return meanLoss;
      }
    }, true);

    // dispose temp tensors
    userT.dispose(); posT.dispose();
    if (genrePosT) genrePosT.dispose();
    if (genreNegT) genreNegT.dispose();

    const lossVal = lossScalar.dataSync ? lossScalar.dataSync()[0] : (await lossScalar.data())[0];
    lossScalar.dispose();
    return lossVal;
  }

  // Utility: compute scores vs all items for a given user embedding (batched to limit memory)
  // uIdx: integer index single user or array. returns JS Float32Array scores length numItems (may be streamed)
  async scoresForUserIndex(uIdx, genreMatrixAll=null, batchSize=1024) {
    // get user embedding
    const uIdxT = tf.tensor1d([uIdx],'int32');
    const uEmb = this.userForward(uIdxT); // [1,d]
    uIdxT.dispose();

    // We'll compute dot with itemEmb or itemEmb processed via MLP + optional genre features
    // For memory, do in batches over items
    const N = this.numItems;
    const scores = new Float32Array(N);
    for (let start=0; start<N; start+=batchSize) {
      const end = Math.min(N, start+batchSize);
      const idx = tf.tensor1d(Array.from({length:end-start}, (_,i)=>i+start),'int32');
      let itemPart = tf.gather(this.itemEmb, idx); // [b,d]
      if (this.useMLP && genreMatrixAll) {
        const genreSlice = tf.tensor2d(genreMatrixAll.slice(start, end)); // expects array of arrays
        const concat = tf.concat([itemPart, genreSlice],1);
        const h = tf.relu(tf.add(tf.matMul(concat, this.W1), this.b1));
        const out = tf.add(tf.matMul(h, this.W2), this.b2); // [b,d]
        itemPart.dispose();
        concat.dispose(); h.dispose();
        itemPart = out;
        genreSlice.dispose();
      }
      // dot: uEmb [1,d] x itemPart [b,d]^T => [1,b]
      const logits = tf.matMul(uEmb, itemPart, false, true); // [1,b]
      const biasSlice = tf.gather(this.itemBias, idx).reshape([1, end-start]); // [1,b]
      const withBias = tf.add(logits, biasSlice);
      const arr = await withBias.data();
      for (let i=0;i<arr.length;i++) scores[start+i]=arr[i];
      idx.dispose(); itemPart.dispose(); logits.dispose(); biasSlice.dispose(); withBias.dispose();
    }
    uEmb.dispose();
    return scores;
  }
}
