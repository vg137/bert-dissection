# BERT Dissection

A notebook-based walkthrough of the representations that make BERT work. The project starts with raw text and follows it through tokenization, learned word embeddings, positional embeddings, and token-type embeddings.

The notebooks are designed for learning by inspection: run a cell, look at the tensors or plots, and connect each result to the corresponding part of the Transformer input.

## Learning path

Work through the notebooks in order:

1. **[Tokenization](step_1_tokenization.ipynb)**
	- Compare character-, word-, and subword-based tokenization.
	- Inspect BERT's WordPiece tokens, token IDs, vocabulary, `[CLS]`, and `[SEP]` tokens.
	- Compare BERT's tokenizer with GPT-2's byte-pair encoding tokenizer.

2. **[Word embeddings](step_2_word_embedding.ipynb)**
	- Load `bert-base-uncased` with Hugging Face Transformers.
	- Inspect the model architecture and its input embedding lookup table.
	- Compare Euclidean distance and cosine similarity for related tokens such as `run`, `runs`, `ran`, and `running`.
	- Explore nearest neighbors in the learned embedding space.

3. **[Positional and type encoding](step_3_positional_and_type_encoding.ipynb)**
	- Inspect BERT's learned positional embedding matrix.
	- Examine BERT's two learned token-type (segment) embeddings.
	- Generate and visualize sinusoidal positional encodings for comparison.

## Getting started

### Requirements

- Python 3.9 or newer
- A few hundred megabytes of disk space for the downloaded Hugging Face models
- Jupyter Notebook, JupyterLab, or VS Code with the Jupyter extension

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter numpy matplotlib networkx torch transformers
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Open the repository in Jupyter or VS Code and run the notebooks from top to bottom. The first model-loading cells download these pretrained models from Hugging Face:

- `google/byt5-small` for character-based tokenization
- `bert-base-uncased` for BERT's WordPiece tokenizer and embeddings
- `gpt2` for byte-pair encoding

The notebooks contain **Open in Colab** links at the top if you prefer a hosted environment.

## Repository layout

| File | Purpose |
| --- | --- |
| `step_1_tokenization.ipynb` | Tokenization strategies and token IDs |
| `step_2_word_embedding.ipynb` | BERT's input embedding layer and semantic neighborhoods |
| `step_3_positional_and_type_encoding.ipynb` | Positional, segment, and sinusoidal encodings |
| `functions.py` | Experimental helper code for exploring nearby embedding vectors |

## Core idea

For a token sequence, BERT builds its input representation by combining three vectors at each position:

```text
token embedding + positional embedding + token-type embedding
```

The notebooks make each term concrete by inspecting the pretrained matrices directly instead of treating BERT as a black box.

## What's next

The next steps in understanding BERT are to dissect the remaining Transformer components, including self-attention, multi-head attention, feed-forward layers, residual connections, layer normalization, and the encoder stack. Together, these components transform the input embeddings into contextual representations.

## Notes

- The notebooks are exploratory lessons rather than a training pipeline or a reusable BERT implementation.
- Model downloads require internet access the first time the relevant cells are run.
- The examples use `bert-base-uncased`, whose maximum sequence length is 512 tokens and whose hidden size is 768.
