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
    this.frequencyCounts = null
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
      // console.log(`Reading in file ${this.files.indexOf(filePath)+1}/${this.files.length}`)
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

  docCount () {
    return this.files.length
  }

  async sentenceCount () {
    let count = 0
    for await (const _ of this.iter('sentence')) {
      count++
    }
    return count
  }

  /**
   * Takes a list of tokens/words and returns their indices in this instance's
   * vocab.
   *
   * @param   {string[]}  words  The list of tokens
   *
   * @return  {number[]}         The list of indices
   */
  word2idx (words) {
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
    // Here we have to count the frequencies of any word. We can't use the
    // corresponding method.
    console.log('Counting word frequencies')
    const counts = {}
    const total = this.docCount()
    let i = 0
    for await (const words of this.iter('document')) {
      i++
      process.stdout.write(`\rProcessing ... ${Math.round(i/total*100)}%`)
      for (const word of words) {
        counts[word] = (counts[word] || 0) + 1
      }
    }
    process.stdout.write('\r')

    // Now we have the frequency counts and can build the vocabulary, removing
    // rare words
    console.log('Creating vocabulary')
    this.vocab = {}
    i = 0
    for (const word in counts) {
      process.stdout.write(`\rProcessing ... ${i}`)
      if (counts[word] >= minCount) {
        this.vocab[word] = i
        // We use the i to speed up the length calculation so we must increment
        // it here (but also only if we actually used the index of i)
        i++
      }
    }
    process.stdout.write('\r')

    // Now that we have the vocab, we can store those frequency counts of those
    // words that we actually use in the vocabulary for future reference.
    this.frequencyCounts = zeros(i + 1)
    for (const word in this.vocab) {
      this.frequencyCounts[this.vocab[word]] = counts[word]
    }

    // Finally, create the i2w with the correct information
    console.log('Creating i2w dictionary')
    this.i2w = {}
    i = 0
    for (const word in this.vocab) {
      i++
      process.stdout.write(`\rProcessing ... ${i}`)
      this.i2w[this.vocab[word]] = word
    }
    process.stdout.write('\r')

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
    if (this.frequencyCounts == null) {
      throw new Error('Cannot access frequency counts: You need to build it first using buildVocab().')
    }

    return this.frequencyCounts
  }
}
