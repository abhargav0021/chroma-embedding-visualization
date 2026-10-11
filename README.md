# Chroma Embedding Visualization

A Jupyter notebook that builds an end-to-end document embedding pipeline — loading a PDF, chunking the text, storing embeddings in ChromaDB, and visualizing the embedding space interactively in 3D.

## What it does

1. **Load a PDF** — extracts raw text from any PDF using `pypdf`
2. **Chunk the text** — splits the document into overlapping windows (configurable size & overlap) so each chunk fits cleanly into an embedding model
3. **Embed & store in ChromaDB** — uses `sentence-transformers` (`all-MiniLM-L6-v2`) as the embedding function and persists vectors in a local ChromaDB collection
4. **Visualize in 3D** — reduces the high-dimensional embeddings to 3 components with PCA and renders an interactive Plotly scatter plot, with each point labelled by its chunk index and a preview of the text on hover

## Tech stack

| Tool | Purpose |
|------|---------|
| `pypdf` | PDF text extraction |
| `chromadb` | Vector store with persistent local storage |
| `sentence-transformers` | Embedding model (`all-MiniLM-L6-v2`) |
| `scikit-learn` | PCA dimensionality reduction |
| `plotly` | Interactive 3D visualization |
| `pandas` / `numpy` | Data wrangling |

## Getting started

```bash
pip install pypdf chromadb sentence-transformers scikit-learn plotly pandas numpy
```

Open the notebook and set `PDF_PATH` to point at any PDF on your machine:

```python
PDF_PATH = "your_document.pdf"
```

Run all cells — the final cell renders the 3D embedding plot inline in the notebook.

## How it works

```
PDF → raw text → overlapping chunks → sentence-transformer embeddings
                                              ↓
                                         ChromaDB
                                              ↓
                                    PCA (n_components=3)
                                              ↓
                                   Interactive Plotly 3D plot
```

Chunks that are semantically similar cluster together in the plot, making it easy to see the thematic structure of a document at a glance.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 | Characters per chunk |
| `overlap` | 50 | Overlap between consecutive chunks |
| `model_name` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `collection_name` | `doc_embeddings` | ChromaDB collection name |
