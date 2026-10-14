"""
Module 3: Indexer
Builds BM25 and FAISS indices for hybrid retrieval
Uses Test Set for indexing
"""

import json
import pickle
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import torch
from tqdm import tqdm
import argparse

def load_documents(filepath: str) -> list:
    """Load documents from preprocessed JSON
    
    Each document includes:
    - 'id': unique identifier
    - 'question': original question (for reference)
    - 'context': context passage
    - 'answers': list of answers (not used in indexing)

    Args:
        filepath (str): Path to preprocessed JSON file
    Returns:
        list: List of document dicts
        
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract documents with metadata
    documents = []
    for item in data:
        documents.append({
            'id': item['id'],
            'text': item['context'],
            'question': item['question']  # Stored for reference
        })
    
    return documents

def build_bm25_index(documents: list) -> BM25Okapi:
    """Build BM25 index
        Args:
            documents (list): List of document dicts with 'text' field
            Returns:
            BM25Okapi: BM25 index object
            """
    print("  Tokenizing documents for BM25...")
    
    # Simple tokenization (lowercase + split)
    tokenized_docs = []
    for doc in tqdm(documents, desc="  Tokenizing"):
        tokens = doc['text'].lower().split()
        tokenized_docs.append(tokens)
    
    print("  Building BM25 index...")
    bm25 = BM25Okapi(tokenized_docs, k1=1.2, b=0.75)
    
    return bm25

def build_faiss_index(documents: list, model_name: str, device: str) -> tuple:
    """Build FAISS index with dense embeddings
    Args:
        documents (list): List of document dicts with 'text' field
        model_name (str): SentenceTransformer model name
        device (str): Device to run embedding model on ('cpu', 'cuda', 'mps')
    Returns:
        tuple: (FAISS index object, numpy array of embeddings)
        """
    
    print(f"  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    model.to(device)
    
    print("  Creating embeddings...")
    texts = [doc['text'] for doc in documents]
    
    # Batch encoding
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding"):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            embeddings = model.encode(
                batch,
                convert_to_tensor=False,
                show_progress_bar=False
            )
        all_embeddings.append(embeddings)
    
    # Combine embeddings
    embeddings = np.vstack(all_embeddings).astype('float32')
    
    # Normalize for cosine similarity
    print("  Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    
    # Create FAISS index
    print("  Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
    index.add(embeddings)
    
    print(f"    Index dimension: {dimension}")
    print(f"    Indexed vectors: {index.ntotal}")
    
    return index, embeddings

def save_indices(bm25_index, faiss_index, embeddings, documents, output_dir: str):
    """Save all indices to disk

    Args:
        bm25_index (BM25Okapi): BM25 index object
        faiss_index: FAISS index object
        embeddings (np.ndarray): Numpy array of embeddings
        documents (list): List of document dicts
        output_dir (str): Directory to save indices
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save BM25
    bm25_file = output_path / "bm25_index.pkl"
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25_index, f)
    print(f"    Successfully Saved BM25 index: {bm25_file}")
    
    # Save FAISS
    faiss_file = output_path / "faiss_index.bin"
    faiss.write_index(faiss_index, str(faiss_file))
    print(f"    Successfully Saved FAISS index: {faiss_file}")
    
    # Save embeddings
    embeddings_file = output_path / "embeddings.npy"
    np.save(str(embeddings_file), embeddings)
    print(f"    Successfully Saved Embeddings: {embeddings_file}")
    
    # Save documents
    docs_file = output_path / "documents.json"
    with open(docs_file, 'w') as f:
        json.dump(documents, f, indent=2)
    print(f"    Successfully Created Documents: {docs_file}")

def main():
    """Main execution"""
    print("="*60)
    print("MODULE: INDEXING OF TEST SET")
    print("="*60)
    
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bioasq", help="Dataset name (pubmedqa or bioasq)")
    args = parser.parse_args()

    if args.dataset == "bioasq":
        input_file = "data/processed/bioasq_test.json"
    else:
        input_file = "data/processed/test_extractive.json"

    output_dir = "data/indices"
    embedding_model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load documents
    print(f"\n[1] Loading documents from: {input_file}")
    documents = load_documents(input_file)
    print(f"    Successfully Loaded {len(documents)} documents")
    
    # Build BM25 index
    print(f"\n[2] Building BM25 index...")
    bm25_index = build_bm25_index(documents)
    print(f"     BM25 index ready")
    
    # Build FAISS index
    print(f"\n[3] Building FAISS index...")
    print(f"    Device: {device}")
    faiss_index, embeddings = build_faiss_index(documents, embedding_model, device)
    print(f"     FAISS index ready")
    
    # Save indices
    print(f"\n[4] Saving indices to: {output_dir}/")
    save_indices(bm25_index, faiss_index, embeddings, documents, output_dir)
    
    # Summary
    print(f"\n[5] Summary:")
    print(f"    Documents indexed:    {len(documents)}")
    print(f"    BM25 corpus size:     {len(bm25_index.doc_freqs)}")
    print(f"    FAISS index size:     {faiss_index.ntotal}")
    print(f"    Embedding dimension:  {embeddings.shape[1]}")
    
    print("\n INDEXING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()