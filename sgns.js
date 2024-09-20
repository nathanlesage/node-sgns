import { makeRandomEmbedding, bmm, zeros, softmax, cosine } from "./sgns_util.js"
import { promises as fs } from 'fs'

////////////////////////////////////////////////////////////////////////////////
// SKIP-GRAM NEGATIVE SAMPLING (SGNS) MODEL
////////////////////////////////////////////////////////////////////////////////

export class SGNSModel {
  static EMBEDDING_DIM = 300 // Number of embedding dimensions

  /**
   * Makes a new SGNS model.
   *
   * @param   {Record<string, number>}  vocab  Vocab dict
   * @param   {number}                  dim    The number of embedding dimensions
   *
   * @return  {SGNSModel}                      The model
   */
  constructor (vocab, dim = SGNSModel.EMBEDDING_DIM) {
    // Save the vocab
    this.vocab = vocab
    this.embeddingDimensionality = dim
    // Word embeddings
    this.w = zeros(Object.keys(vocab).length).map(x => makeRandomEmbedding(dim))
    // Context word embeddings
    this.c = zeros(Object.keys(vocab).length).map(x => makeRandomEmbedding(dim))
  }

  /**
   * Forward model call. Takes a list of target words and context words, and returns
   * a list of dot products between the two embedding lists.
   *
   * @param   {number[]}   w  List of target word IDs
   * @param   {number[][]} c  Matrix of context word IDs
   *
   * @return  {[number[][], number[][], number[][][]]} Returns three matrices, D, target embeddings, and context embeddings
   */
  forward (w, c) {
    // So what we need to do here is: Get both the embedding vectors for the
    // target word w and the context word c, and then calculate the dot product
    // and write it to the corresponding position in the return array.

    // target shape: batch_size x embedding_dim
    const target_embeddings = w.map(idx => this.w[idx])
    // context shape: batch_size x num_ns x embedding_dim
    const context_embeddings = c.map(context_row => {
      return context_row.map(idx => this.c[idx])
    })

    // Now we have the embeddings. target_embeddings is a matrix of batch_size x
    // embedding_dim; the context_embeddings is a three-dimensional matrix of
    // batch_size x num_ns x embedding_dim.

    // Our goal here is to retrieve matrix D which has the same shape as the
    // input c, which involves reducing the context_embeddings from three to two
    // dimensions by creating the dot-product of those n-dimensions (the
    // embedding_dim) and receive a scalar value.

    // PyTorch offers the function torch.bmm, which will calculate this product
    // in a batched way, for JavaScript we have to implement it ourselves.
    const D = bmm(context_embeddings, target_embeddings)

    // NOTE: Difference to the Python implementation (as this one is bare metal)
    // is that here we have to manually apply a row-wise softmax.
    for (let i = 0; i < D.length; i++) {
      D[i] = softmax(D[i])
    }

    // Because we're not in Python, we don't have a wrong dimensionality and can
    // return the matrix as-is. However, since we don't have this cool graph-
    // building feature, we need to return the corresponding word embeddings so
    // that the gradient descent algorithm can adapt them
    return [D, target_embeddings, context_embeddings]
  }

  /**
   * Takes the target word and returns a list of n most similar words
   *
   * @param   {string}  targetWord  The target word.
   * @param   {number}  n           How many words to return (default: 10)
   *
   * @return  {[string, number]}    An array of most similar words (`[word, cosine]`).
   */
  similar (targetWord, n = 10) {
    const targetIdx = this.vocab[targetWord]
    if (targetIdx === undefined) {
      return []
    }

    const emb = this.w[targetIdx]
    return this.similarVec(emb, n)
  }

  /**
   * Returns a list of n most similar words to the provided vector.
   *
   * @param   {number[]}  vec       The target vector.
   * @param   {number}    n         How many words to return (default: 10)
   *
   * @return  {[string, number]}    An array of most similar words (`[word, cosine]`).
   */
  similarVec (vec, n = 10) {
    const cosines = []
    for (const [ word, idx ] of Object.entries(this.vocab)) {
      const compareEmb = this.w[idx]
      cosines.push([word, cosine(vec, compareEmb)])
    }

    cosines.sort((a, b) => b[1] - a[1]) // Sort descending

    return cosines.slice(0, Math.min(n, cosines.length))
  }

  /**
   * Writes a trained model to disk.
   *
   * @param   {string}  filePath  The file path
   */
  async toDisk (filePath) {
    const contents = ['== VOCABULARY']
    for (const [ word, idx ] of Object.entries(this.vocab)) {
      contents.push(`${word}\t${idx}`)
    }
    //this.embeddingDimensionality
    contents.push('== WORD EMBEDDINGS')
    for (let i = 0; i < this.w.length; i++) {
      contents.push(this.w[i].map(x => String(x)).join('\t'))
    }
    contents.push('== CONTEXT EMBEDDINGS')
    for (let i = 0; i < this.c.length; i++) {
      contents.push(this.c[i].map(x => String(x)).join('\t'))
    }

    await fs.writeFile(filePath, contents.join('\n'), 'utf-8')
  }

  /**
   * Loads an existing model from disk
   *
   * @param   {string}  filePath  The file path with the model data
   *
   * @return  {SGNSModel}            The rehydrated model
   */
  static async fromDisk (filePath) {
    const content = await fs.readFile(filePath, 'utf-8')

    const lines = content.split('\n')

    const vocabIdx = lines.findIndex(x => x === '== VOCABULARY')
    const wordIdx = lines.findIndex(x => x === '== WORD EMBEDDINGS')
    const ctxIdx = lines.findIndex(x => x === '== CONTEXT EMBEDDINGS')

    if (vocabIdx < 0 || wordIdx < 0 || ctxIdx < 0) {
      throw new Error('Corrupt file')
    }

    const vocab = {}
    let lineIdx = 1
    while (lineIdx < wordIdx) {
      const [word, idx] = lines[lineIdx].split('\t')
      vocab[word] = idx
      lineIdx++
    }

    lineIdx++ // Jump over separator

    const wordEmbeddings = []
    while (lineIdx < ctxIdx) {
      wordEmbeddings.push(lines[lineIdx].split('\t').map(x => parseFloat(x)))
      lineIdx++
    }

    lineIdx++ // Jump over separator

    const ctxEmbeddings = []
    while (lineIdx < lines.length) {
      ctxEmbeddings.push(lines[lineIdx].split('\t').map(x => parseFloat(x)))
      lineIdx++
    }

    // Now we can instantiate an empty model
    const model = new SGNSModel(vocab, wordEmbeddings[0].length)

    // Overwrite the word and context embeddings
    model.w = wordEmbeddings
    model.c = ctxEmbeddings

    return model
  }
}
