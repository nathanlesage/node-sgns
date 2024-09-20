import { promises as fs } from 'fs'
import { zeros } from './sgns_util.js'

////////////////////////////////////////////////////////////////////////////////
// TEXT PROCESSING UTILITIES
////////////////////////////////////////////////////////////////////////////////

/**
   * Removes (Markdown) syntax from a single word
   *
   * @param   {string}  word  Input
   *
   * @return  {string}        Output
   */
function removeSyntax (word) {
  return word.replaceAll(/[_*]+/g, '')
}

/**
 * Takes a doc and splits it into words.
 *
 * @param   {string}    doc  The files contents
 *
 * @return  {string[]}       A list of words in the document
 */
function docToWords (doc) {
  return doc.toLowerCase()
    // Split on non-word characters
    .split(/\W+/g)
    // Remove additional characters that are considered word-characters for
    // RegExp but we don't count as such, e.g., Markdown syntax
    .map(removeSyntax)
}

/**
 * Turns a document into sentences.
 *
 * @param   {string}    doc  The document
 *
 * @return  {string[]}       A list of sentences in the doc
 */
function docToSentences (doc) {
  return doc
    // Split into sentences (TODO: Potentially better algorithm). Currently
    // assumes period, whitespace, then uppercase letter
    .split(/\.\s+(?=[A-Z])/g)
}

////////////////////////////////////////////////////////////////////////////////
// DATA LOADER CLASS
////////////////////////////////////////////////////////////////////////////////

export class DataLoader {
  /**
   * Instantiate a new DataLoader. Provide a list of absolute paths for the data
   * loader to use.
   *
   * @param   {string[]}  textFilePaths  List of file paths
   */
  constructor (textFilePaths) {
    this.files = textFilePaths
    this.vocab = null
    this.i2w = null
  }

  /**
   * Iterates through the data, yielding lists of tokens (words). It supports
   * two modes; "sentences" where it yields lists of tokens per sentence, and
   * "document" where it yields lists of tokens per document.
   *
   * @param   {'sentence'|'document'}  mode  The mode, defaults to sentence
   *
   * @return  {string[]}                     Yields lists of tokens
   */
  async *iter (mode = 'sentence') {
    for (const filePath of this.files) {
        const contents = await fs.readFile(filePath, 'utf-8')
        if (mode === 'document') {
          yield docToWords(contents)
        } else {
          for (const sentence of docToSentences(contents)) {
            yield docToWords(sentence)
          }
        }
      }
  }

  /**
   * Takes a list of tokens/words and returns their indices in this instance's
   * vocab.
   *
   * @param   {string[]}  words  The list of tokens
   *
   * @return  {number[]}         The list of indices
   */
  wordsToIndices (words) {
    if (this.vocab === null) {
      throw new Error('Cannot turn words to indices: Vocab not yet initialized')
    }

    return words.map(w => this.vocab[w]).filter(w => w !== undefined)
  }

  idx2words (idx) {
    return idx.map(i => this.i2w[i]).filter(w => w !== undefined)
  }

  /**
   * Builds and returns a vocab based on the data in the loader.
   *
   * @param   {number}  minCount  Threshold below which to throw words away
   *
   * @return  {Promise<Record<string, number>>}  The vocab (word -> index)
   */
  async buildVocab (minCount = 5) {
    const v = new Set()

    for await (const words of this.iter('document')) {
      // Remove duplicates as early as possible by turning it into a Set
      for (const word of [... new Set(words)]) {
        v.add(word)
      }
    }

    // Convert the Set into an array, turn that into tuples of word -> index,
    // and then return a dictionary object from that.
    this.vocab = Object.fromEntries([...v].map((word, idx) => [word, idx]))

    // Now, prune the vocabulary. For this, first get the frequencies...
    const count = await this.getFrequencyCounts()
    for (const [word, idx] of Object.entries(this.vocab)) {
      if (count[idx] < minCount) {
        delete this.vocab[word]
      }
    }

    // Cleanup the counts the vocab refers to
    const words = Object.keys(this.vocab)
    this.vocab = {}
    for (let i = 0; i < words.length; i++) {
      this.vocab[words[i]] = Object.keys(this.vocab).length
    }

    // Now create the i2w with the correct information

    this.i2w = {}
    for (const [word, idx] of Object.entries(this.vocab)) {
      this.i2w[idx] = word
    }

    return this.vocab
  }

  /**
   * Returns the vocab.
   *
   * @return  {Record<string, number>}  The vocab (word -> index)
   */
  getVocab () {
    if (this.vocab === null) {
      throw new Error('Cannot access vocab: You need to build it first using buildVocab().')
    }
    return this.vocab
  }

  /**
   * Calculates frequency counts for each individual word in the vocabulary.
   *
   * @return  {Promise<number[]>}  The frequency counts (word -> count)
   */
  async getFrequencyCounts () {
    const vocab = this.getVocab()
    const frequencyCounts = zeros(Object.keys(vocab).length)

    for await (const words of this.iter('document')) {
      for (const word of words) {
        if (word in vocab) {
          frequencyCounts[vocab[word]] += 1
        }
      }
    }

    return frequencyCounts
  }
}
