import { promises as fs } from 'fs'
import path from 'path'
import { DataLoader } from './data_loader.js'
import { train } from './train.js'
import { SGNSModel } from './sgns.js'

const FOLDER_PATH = "/path/to/some/Markdown files.md"

////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
////////////////////////////////////////////////////////////////////////////////

async function trainNewModel (folderPath) {
  // First, read in the folder
  const filePaths = (await fs.readdir(folderPath))
    // Retain only Markdown files
    .filter(filename => filename.endsWith('.md'))
    // Map to absolute paths
    .map(filename => path.join(folderPath, filename))
    // DEBUG
    // .slice(0, 30)
  
  // Now, instantiate a data loader with these files
  console.log('Instantiating DataLoader...')
  const loader = new DataLoader(filePaths)
  console.log('Done. Calculating vocabulary...')
  const vocab = await loader.buildVocab()
  console.log(`Done. ${Object.entries(vocab).length} unique words in the vocabulary.`)

  // At this point, we have the data, a vocab, a model, and we are set to
  // training the model.

  const model = await train(loader, 50, 5, 15, 100_000, 1, 1e-1)
  return model
}

async function main () {
  const model = await trainNewModel(FOLDER_PATH)
  await model.toDisk('model.sgns')
  // const model = await SGNSModel.fromDisk('model.sgns')

  console.log('Testing words')
  console.log('Ten most similar words to jihadist:')
  console.log(model.similar('jihadist'))
  console.log('Ten most similar words to modernity:')
  console.log(model.similar('modernity'))
}

// MAIN CALL
main()
  .catch(err => {
    console.error(err)
  })
  .then(() => {
    console.log('main() finished successfully.')
  })
