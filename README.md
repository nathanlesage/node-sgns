# node-sgns

> A plain vanilla, zero-dependency bare metal, JavaScript implementation of the Skip-Gram Negative Sampling (SGNS) Word2Vec algorithm for Node.js.

I used Node.JS 20 for developing this script.

> [!WARNING]
> This is not a production-ready implementation. It is merely for demonstrative/learning purposes. There are likely errors I did make (because I'm not a computer scientist and/or mathematician and simply follow instructions) and optimizations I did not make.

## How To Run

To run this model on your computer, clone the repository, replace the folder path constant in `index.js` with a path to some folder full of Markdown files (`.md`).

Then, uncomment the training-related lines (and comment out the "read from disk" ones) in the `main` function, and run it using node.

Saving and loading the model will use a very barebones data format that I came up on the fly. Even though I tried to compress it as much as possible, the model files will still be fairly large (about 10MB using the default settings).

After having trained the model, you can do some basic operations such as calculating similarity scores for words.

## Acknowledgements

This repository is not at all my ideational work. This is a combination of internet resources and course work during my PhD. The main acknowledgement goes to Prof. Marco Kuhlmann (IDA, Linköping University) who has provided the lab Python notebook with most of the code that comprises this algorithm. What I did was merely translate the Python code into JavaScript.

Other, smaller acknowledgements go to various people who have written resources on the internet on the matter (see the source files for that).

The main part of the work that I provided in this repository was the general data loading algorithms, the Python-to-JavaScript translation, and the implementation of the backpropagation mechanism using gradient descent (which, tbh, was fairly anticlimactic).

## License

This code is not yet licensed. If you want to use this code, send me a message so that I can re-prioritize finding a suitable license for this thing.
