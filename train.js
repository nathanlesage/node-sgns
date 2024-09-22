// Training functions

import { SGNSModel } from "./sgns.js"
import { binaryCrossEntropyWithLogits, zeros } from "./sgns_util.js"
import { randomInt } from "crypto"
import { binomial } from "./binomial.js"

/**
 * The purpose of this function is to generate training examples for the
 * training of the SGNS model. It yields tuples containing a vector of training
 * word indices (based on the vocabulary) and a matrix of context word indices
 * (based on the vocabulary), where the first index is always a correct context
 * word and the remaining indices in the matrix are incorrect context words.
 *
 * @param   {Record<string, number>}  vocab        The vocabulary
 * @param   {number[]}                counts       The word frequency counts
 * @param   {DataLoader}              loader       The DataLoader instance to use
 * @param   {number}                  window       The sliding window size (default 5)
 * @param   {number}                  num_ns       The amount of negative samples to produce (default 5)
 * @param   {number}                  batch_size   The batch size (default 100)
 * @param   {number}                  ns_exponent  The exponent for the negative sampling (default 0.75)
 *
 * @return  {AsyncGenerator<[number[], number[][]]>}    A tuple of target words and context words
 */
async function *trainingExamples (
  vocab,
  counts,
  loader,
  window = 5,
  num_ns = 5,
  batch_size = 100,
  ns_exponent = 0.75
) {
  // First, we need to prepare the cumsum vector that will be used for sampling
  // the negatives in the context matrices. This can be calculated once and then
  // remains the same.
  const cumsum = counts.map(c => c ** ns_exponent)
  for (let i = 1; i < cumsum.length; i++) {
    cumsum[i] = cumsum[i - 1] + cumsum[i]
  }

  // Also, calculate the dropout probability for each word once here. This will
  // reduce the likelihood of very frequent words of being yielded as a positive
  // example.
  const DROPOUT_THRESHOLD = 0.001
  const N = counts.reduce((prev, cur) => prev + cur, 0)
  const dropout = zeros(counts.length)
  for (const idx of Object.values(vocab)) {
    const prob = 1 - Math.sqrt(DROPOUT_THRESHOLD * N / counts[idx])
    dropout[idx] = Math.max(0, prob)
  }

  // Preset the target vector and context matrix
  let targetVec = zeros(batch_size)
  let contextMat = zeros([batch_size, 1 + num_ns])
  let batchIdx = 0

  for await (const rawSentence of loader.iter('sentence')) {
    const sentence = loader.word2idx(rawSentence)
      // Dropout words at this point using a binomial draw
      .filter(idx => binomial(0, 1, dropout[idx]) !== 1)

    // Slide the window over the text
    for (let i = 0; i < sentence.length; i++) {
      const targetWord = sentence[i]
      const contextWords = []
      for (let slide = i - window; slide <= i + window; slide++) {
        if (slide < 0 || slide >= sentence.length || slide === i) {
          continue // No out of bounds, and not the target word (i)
        }
        contextWords.push(sentence[slide])
      }

      // Now we have the target and context words. The target word goes into the
      // targetVec immediately, as go the positive context words.
      for (const contextWord of contextWords) {
        targetVec[batchIdx] = targetWord
        contextMat[batchIdx][0] = contextWord // First index is always the positive sample
        batchIdx++
        if (batchIdx === batch_size) {
          yield [targetVec, buildNegativeSamples(contextMat, cumsum, batch_size, num_ns)]
          batchIdx = 0
          targetVec = zeros(batch_size)
          contextMat = zeros([batch_size, 1 + num_ns])
        }
      }
    }
  }

  // Yield a final partial batch
  if (batchIdx > 0) {
    yield [targetVec.slice(0, batchIdx), buildNegativeSamples(contextMat, cumsum, batch_size, num_ns).slice(0, batchIdx)]
  }
}

/**
 * Fills all columns other than the first with randomly sampled negative samples
 * and returns it.
 *
 * @param   {number[][]}  contextMat  Context word matrix
 * @param   {number[]}    cumsum      Cumulative sum vector
 * @param   {number}      batchSize   Batch size
 * @param   {number}      numNS       Number of negative samples to generate
 *
 * @return  {number[][]}              A reference to the contextMat
 */
function buildNegativeSamples (contextMat, cumsum, batchSize, numNS) {
  const min = Math.floor(cumsum[0])
  const max = Math.floor(cumsum[cumsum.length - 1])
  const randomSample = zeros(batchSize * numNS).map(x => randomInt(min, max))

  // Now we need this weird implementation of searchsorted (cf. https://pytorch.org/docs/stable/generated/torch.searchsorted.html)
  // Seems straightforward.
  const negativeIdx = randomSample
    .map(r => cumsum.findIndex(x => x >= r) - 1)
    // Due to the way the mapper works, if the cumsum value is smaller than the
    // first index of randomSample, it will be -1.
    .map(idx => idx < 0 ? 0 : idx)
  for (let row = 0; row < batchSize; row++) {
    for (let col = 1; col <= numNS; col++) { // Account for the positive example
      // NOTE that negativeIdx is a long vector of batch size TIMES numMS, not a
      // matrix!
      contextMat[row][col] = negativeIdx[row + col - 1]
    }
  }
  return contextMat
}

/**
 * Implement the primary training loop for the SGNS model. This will take the
 * provided data and the hyperparameters and return a trained SGNS model.
 *
 * @param {DataLoader} loader       The primary data loader that provides the training data
 * @param {number}     embeddingDim The embedding dimensionality (default: 50)
 * @param {number}     window       The context window size (default: 5)
 * @param {number}     numNS        The ratio of negative samples to positives (must be integer)
 * @param {number}     batchSize    How many examples per forward run of the model (default: 100k)
 * @param {number}     nEpochs      How many runs through the training data (default: 1)
 * @param {number}     lr           The learning rate (default: 0.1 or 1e-1)
 *
 * @return {SGNSModel}              The learned model
 */
export async function train (
  loader,
  embeddingDim = 50,
  window = 5,
  numNS = 5,
  batchSize = 100_000,
  nEpochs = 1,
  lr = 1e-1
) {
  const vocab = loader.getVocab()
  const counts = await loader.getFrequencyCounts()

  const model = new SGNSModel(vocab, embeddingDim)

  console.log('Starting training with hyperparameters:')
  console.log({ embeddingDim, window, numNS, batchSize, nEpochs, lr })
  console.log('')
  let totalBatches = 0
  let errors = 0
  const allLossItems = []

  for (let epoch = 1; epoch <= nEpochs; epoch++) {
    let runningLoss = 0.0
    let batches = 0

    for await (const [targetVec, contextMat] of trainingExamples(vocab, counts, loader, window, numNS, batchSize)) {
      batches++
      totalBatches++

      // Step 1: Forward pass through the model, retrieve D
      const [D, t, c] = model.forward(targetVec, contextMat)

      // Step 2: Calculate the true data (Same shape as D with column 1 set to 1)
      const labels = zeros([D.length, 1 + numNS])
      for (let i = 0; i < D.length; i++) {
        labels[i][0] = 1
      }

      // Step 3: Calculate the binary cross entropy loss with logits
      const loss = binaryCrossEntropyWithLogits(D, labels)
      const lossItem = loss.reduce((prev, cur) => prev + cur, 0)
      runningLoss += lossItem
      allLossItems.push(lossItem)
      process.stdout.write(`\rE${epoch}/B${batches} | Avg.loss ${runningLoss/batches} | Current loss ${lossItem}`)

      // Step 4: Perform the gradient descent
      // It is important to understand that the gradient descent is independent
      // from the loss in terms of calculation. The loss that we calculate here
      // is merely useful for us to determine how good or bad the classifier
      // actually is. However, we do need the loss for calculating the gradient
      // descent, as it describes our target or goal. We define as the target
      // the minimization of the binary cross entropy, which means that whatever
      // the gradient descent does, it should minimize whatever the cross
      // entropy calculates for us. The connection thus comes to how we define
      // the function that the gradient descent should use. The gradient descent
      // is nothing but the partial derivative of the loss function. In the case
      // of a binary cross entropy, it is literally defined as predicted labels
      // minus true labels times the input matrix. See here: https://www.python-unleashed.com/post/derivation-of-the-binary-cross-entropy-loss-gradient

      // Since we are using a very basic classifier without hidden layers, the
      // gradient has only exactly one step and not multiple ones. In addition,
      // this means that the word embeddings and context embeddings are
      // interchangeable with regard to the gradient. This means that, while
      // the partial derivative of the BCE is literally diff times input, what
      // the input is depends on the aim: Both word and context embeddings form
      // the "input" to the other, which means that, to calculate the numbers by
      // which we have to adapt each of those is entirely determined by the diff
      // between predicted and true label as well as the other embeddings.

      // In short, the gradient for the context word embeddings is:
      // diff times word embeddings times learning rate -> substract that.
      // Likewise, for the word embeddings, it is:
      // diff times context embedds times learning rate -> substract that.

      // This also explains how this dude here has done it: https://betterprogramming.pub/word2vec-embeddings-from-the-ground-up-for-the-ml-adjacent-8d8c484e7cb5#09ec
      // He calculates the diffs (p0), then the context word gradient (a0), then
      // calculates the word gradient (p1). The most likely reason I can imagine
      // for why he calls the word gradient p1 and then adds a step to call it
      // a1 is his way of (probably more efficiently) extracting the word
      // embeddings.
      // In any case, he extracts both gradients after reducing it by the
      // learning rate and calls it a day.

      // So, step one: Calculate a diff matrix
      const diff = zeros([D.length, D[0].length])
      for (let i = 0; i < D.length; i++) {
        for (let j = 0; j < D[0].length; j++) {
          // True label = 0 -> positive diff
          // True label = 1 -> negative diff
          // Greater difference = greater descent
          diff[i][j] = D[i][j] - labels[i][j]
        }
      }

      // Now calculate the WORD embedding gradient. For this, we have to bmm the
      // context word matrix with the diff and attenuate using the lr.
      // context shape = batch x num_ns x embedding
      // diff shape    = batch x num_ns
      const a1 = zeros([D.length, embeddingDim])
      for (let i = 0; i < D.length; i++) {         // For each batch
        for (let j = 0; j < diff[i].length; j++) { // For each num_ns/loss diff
          for (let k = 0; k < embeddingDim; k++) { // For each embedding scalar
            a1[i][k] = c[i][j][k] * diff[i][j] * lr // Embedding scalar times num_ns diff times lr
          }
        }
      } // TODO: This looks horrifying

      // Do the same for the CONTEXT embedding gradient.
      // word shape = batch x embedding
      // diff shape = batch x num_ns
      // Required output = batch x num_ns x embedding
      const a0 = zeros([D.length, numNS + 1, embeddingDim])
      for (let i = 0; i < D.length; i++) {         // For each batch
        for (let j = 0; j < diff[i].length; j++) { // For each num_ns/loss diff
          for (let k = 0; k < embeddingDim; k++) { // For each embedding scalar
            a0[i][j][k] = t[i][k] * diff[i][j] * lr // Embedding scalar times num_ns diff times lr
          }
        }
      } // TODO: This looks horrifying

      // Now subtract a0 and a1 from the corresponding matrix elements. NOTE that
      // here we must ensure that we modify those things in-place.
      for (let i = 0; i < D.length; i++) {
        for (let j = 0; j < diff[i].length; j++) {
          for (let k = 0; k < embeddingDim; k++) {
            c[i][j][k] = c[i][j][k] - a0[i][j][k]
          }
        }
      }

      for (let i = 0; i < D.length; i++) {
        for (let k = 0; k < embeddingDim; k++) {
          t[i][k] = t[i][k] - a1[i][k]
        }
      }
    }
  }
  console.log('')

  console.log(`Training done. ${errors} error batches. Total: ${totalBatches} (${Math.round(errors/totalBatches * 100)}%)`)

  // Calculate the loss's slope to see if the model got better or worse.
  // Formula comes from https://math.stackexchange.com/questions/204020/what-is-the-equation-used-to-calculate-a-linear-trendline
  const y = [...Array(allLossItems.length).keys()]
  const sumX = allLossItems.reduce((prev, cur) => prev + cur, 0)
  const sumY = y.reduce((prev, cur) => prev + cur, 0)
  const sumX2N = allLossItems.map(x => x ** 2).reduce((prev, cur) => prev + cur, 0) * allLossItems.length
  const sumX2 = sumX ** 2
  let sumXYN = 0
  for (let i = 0; i < allLossItems.length; i++) {
    sumXYN += allLossItems[i] * y[i]
  }
  sumXYN = sumXYN * allLossItems.length

  const slope = (sumXYN - sumX * sumY) / sumX2N - sumX2

  console.log(`The slope of the loss items is ${slope}. It should be negative.`)

  return model
}
