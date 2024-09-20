////////////////////////////////////////////////////////////////////////////////
// NUMERICAL OPERATIONS UTILITY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

import assert from "assert"
import { randomInt } from "crypto"

/**
 * Presets an array or matrix with the provided shape, filling it up with zeros.
 *
 * @param   {number|number[]}  shape  The dimensionality of the array. Tuple of numbers
 *
 * @return  {number[]}                An n-dimensional array of zeros
 */
export function zeros (shape) {
  if (Array.isArray(shape)) {
    // Use recursion
    const thisDim = shape.shift()
    const emptyArr = [...Array(thisDim).keys()]
    if (shape.length > 0) {
      return emptyArr.map(x => zeros([...shape])) // We have to clone the array
    } else {
      return emptyArr.map(x => 0)
    }
  } else {
    return [...Array(shape).keys()].map(x => 0)
  }
}

/**
   * Calculates the batched dot product between two matrices, specific for the
   * SGNS model we implement here.
   *
   * @param   {number[][][]}  context  First matrix; shape n_batch x num_ns x embedding_dim
   * @param   {number[][]}    target   Second matrix; shape n_batch x embedding_dim
   *
   * @return  {number[][]}             Output matrix of shape n_batch x num_ns
   */
export function bmm (context, target) {
  const batchSize = target.length
  // Basically, we have to calculate the dot product of each target word
  // embedding with each of its corresponding context words (1 true and the
  // rest negative samples).

  // Preset D (n_batch x num_ns) with zeroes
  const D = zeros([context.length, context[0].length])

  for (let i = 0; i < batchSize; i++) {
    const contextWords = context[i]
    for (let j = 0; j < contextWords.length; j++) {
      try {
        D[i][j] = dot(target[i], contextWords[j])
      } catch (e) {
        console.log(target[i], contextWords)
        throw e
      }
    }
  }

  return D
}

/**
 * Calculates the dot product between two vectors.
 *
 * @param   {number[]}  vec1  First vector
 * @param   {number[]}  vec2  Second vector
 *
 * @return  {number}          Dot product
 */
export function dot (vec1, vec2) {
  let result = 0
  for (let i = 0; i < vec1.length; i++) {
    result += vec1[i] * vec2[i]
  }
  return result
}

/**
   * Creates a new, random embedding vector with EMBEDDING_DIM dimensions.
   *
   * @param   {number}    dim The optional number of embeddings
   * @return  {number[]}      The embedding vector.
   */
export function makeRandomEmbedding (dim) {
  // TODO: Better initialization; good is a He et al.'s distribution with µ = 0
  // and std = sqrt(2/n_in) where n_in is the embedding dimension
  // (cf. https://betterprogramming.pub/word2vec-embeddings-from-the-ground-up-for-the-ml-adjacent-8d8c484e7cb5)
  return zeros(dim).map(x => Math.random() - 0.5)
}

/**
 * Calculates the softmax over a vector of logits
 *
 * @param   {number[]}  vec  A vector of logits
 *
 * @return  {number[]}       A vector of probabilities
 */
export function softmax (vec) {
  // JavaScript is fun: Very large numbers for vec[i] (a bit more than 700) will
  // cause the Engine to assign Infinity instead of a real number. To avoid this
  // and reach a somewhat better numerical stability, we do something super
  // stupid I read on the internet and subtract the max value from each element
  // prior to calculating the softmax.
  vec = vec.map(x => x - Math.max(...vec))
  let sum = 0
  for (let i = 0; i < vec.length; i++) {
    sum += Math.E ** vec[i]
  }

  return vec.map(x => (Math.E ** x) / sum)
}

/**
 * Calculates the cosine similarity between two numeric vectors
 *
 * @param   {number[]}  vec1  Vector 1
 * @param   {number[]}  vec2  Vector 2
 *
 * @return  {number}          The similarity; bound between -1 and 1.
 */
export function cosine (vec1, vec2) {
  assert(vec1.length === vec2.length, "Cannot calculate cosine similarity of vectors with different dimensions.")

  let ab = 0
  for (let i = 0; i < vec1.length; i++) {
    ab += vec1[i] * vec2[i]
  }

  const a2 = vec1.map(x => x ** 2).reduce((prev, cur) => prev + cur, 0)
  const b2 = vec2.map(x => x ** 2).reduce((prev, cur) => prev + cur, 0)

  return ab / (a2 * b2)
}

/**
 * Calculates the binary cross entropy with logits from a distribution of true
 * samples (q) and calculated samples (p). NOTE: Assumes a batched input, so
 * ensure that you pass at least a 1-element-array [p], [q]
 *
 * @param   {number[][]}  p  A series of predicted labels [0.0; 1.0]
 * @param   {number[][]}  q  A series of true labels [0; 1]
 *
 * @return  {number[]}     The cross entropy loss between the two distributions.
 */
export function binaryCrossEntropyWithLogits (predicted, trueLabels) {
  // TODO: This does not yet account for the sigmoids required in the "withLogits" part!
  assert(predicted.length === trueLabels.length, "Cannot calculate BCE: Prediction and trueLabels had different shape.")
  // Implements https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
  // Formula: L = -1/N sum(1..N) of y_i * log(p(y_i)) + (1 - y_i) * log(1 - p(y_i))

  // So, we're receiving a batch, and we have to return an array of losses, one
  // per example.
  const loss = zeros(predicted.length)
  const batchSize = predicted.length
  for (let i = 0; i < batchSize; i++) {
    const p = predicted[i]
    const q = trueLabels[i]

    let tally = 0
    for (let j = 0; j < p.length; j++) {
      // Before + if true label = 1; after + if true label = 0
      tally += q[j] * Math.log(p[j] + 1e-6) + (1 - q[j]) * Math.log(1 - p[j] + 1e-6)
    }
    loss[i] = -tally / batchSize
  }

  return loss
}
