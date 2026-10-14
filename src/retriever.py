"""
Module 4:
Hybrid retrieval combining BM25 and FAISS using Reciprocal Rank Fusion (RRF)
Implement RRF fusion, verify it improves over single methods
Dynamic fusion pool size based on top_k parameter
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi



import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from scipy import stats
import pandas as pd


class HybridRetriever:
    """
    Hybrid Retriever using BM25 and FAISS with Reciprocal Rank Fusion (RRF)
    Args:
        index_dir: Directory where indices are stored
        rrf_k: RRF parameter k
        device: Device for embedding model ('cpu' or 'cuda')
    """
    def __init__(self, index_dir: str = "data/indices", rrf_k: int = 60, device="cpu"):
        self.index_dir = Path(index_dir)
        self.encoder = None
        self.bm25_index = None
        self.faiss_index = None
        self.device = device
        self.documents = None
        self.rrf_k = rrf_k
        
    def load_indices(self):
        """
        Load pre-built indices        
        """
        import pickle
        
        # Load BM25
        with open(self.index_dir / "bm25_index.pkl", 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        # Load FAISS
        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss_index.bin"))
        
        # Load documents
        with open(self.index_dir / "documents.json", 'r') as f:
            self.documents = json.load(f)
        
        # Load encoder
        self.encoder = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1", device=self.device)
        
        print(f" Loaded indices: {len(self.documents)} documents")
        print(f"[Retriever] Loading encoder on device: {self.device}")
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 sparse retrieval
        Args:
            query: The input query string
            top_k: Number of top documents to retrieve
        Returns:
            List of tuples (doc_id, score)
        
        """
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results
    
    def faiss_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """FAISS dense retrieval
        Args:
            query: The input query string
            top_k: Number of top documents to retrieve
        Returns:
            List of tuples (doc_id, score)"""
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        results = [(int(indices[0][i]), float(scores[0][i])) 
                   for i in range(len(indices[0]))]
        return results
    
    def reciprocal_rank_fusion(self, 
                                bm25_results: List[Tuple[int, float]],
                                faiss_results: List[Tuple[int, float]],
                                k: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF)
        Score = 1/(k + rank_bm25) + 1/(k + rank_faiss)
        
        Args:
            k: RRF parameter (default 60 from literature)
        Returns:
            List of tuples (doc_id, rrf_score) sorted by rrf_score descending
        """
        rrf_scores = {}
        
        # Add BM25 scores
        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Add FAISS scores
        for rank, (doc_id, _) in enumerate(faiss_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Hybrid search with RRF fusion
        
        Dynamic fusion pool size based on top_k

        Args:
            query: The input query string
            top_k: Number of top documents to retrieve
        
        Returns:
            List of documents with metadata and scores
        """
        fusion_pool_size = min(top_k * 2, len(self.documents))
        
        # Get results from both methods with dynamic pool size
        bm25_results = self.bm25_search(query, top_k=fusion_pool_size)
        faiss_results = self.faiss_search(query, top_k=fusion_pool_size)
        
        # Fuse rankings
        fused_results = self.reciprocal_rank_fusion(
            bm25_results, 
            faiss_results, 
            k=self.rrf_k
        )
        
        # Convert to document format
        retrieved_docs = []
        for doc_id, rrf_score in fused_results[:top_k]:
            doc = self.documents[doc_id].copy()
            doc['retrieval_score'] = rrf_score
            doc['doc_index'] = doc_id
            retrieved_docs.append(doc)
        
        return retrieved_docs


# ============ TEST MODULE ============
if __name__ == "__main__":
    print("="*60)
    print("MODULE TEST: Hybrid Retrieval with RRF Fusion")
    print("="*60)
    
    # Initialize retriever
    print("\n[Test 1] Loading indices...")
    retriever = HybridRetriever()
    retriever.load_indices()
    
    # Test query
    test_query = "Does the reformed primary care system have any pharmaceutical expenditure awareness?"
    
    # Test 2: Verify top_k parameter works with dynamic fusion pool size
    print(f"\n[Test 2] Testing Dynamic top_k")
    print(f"Query: {test_query}")
    
    for k in [3, 5, 10]:
        results = retriever.search(test_query, top_k=k)
        print(f"\n  top_k={k}: Retrieved {len(results)} documents")
        print(f"    Fusion pool size used: {k * 2}")
        print(f"    Top result: Doc {results[0]['doc_index']} (score: {results[0]['retrieval_score']:.4f})")
    
    # Test 3: BM25 only
    print(f"\n[Test 3] BM25 Search")
    bm25_results = retriever.bm25_search(test_query, top_k=5)
    print("Top 5 BM25 Results:")
    for rank, (doc_id, score) in enumerate(bm25_results, 1):
        print(f"  {rank}. Doc {doc_id:2d} | Score: {score:6.3f}")
    
    # Test 4: FAISS only
    print(f"\n[Test 4] FAISS Search")
    faiss_results = retriever.faiss_search(test_query, top_k=5)
    print("Top 5 FAISS Results:")
    for rank, (doc_id, score) in enumerate(faiss_results, 1):
        print(f"  {rank}. Doc {doc_id:2d} | Score: {score:6.3f}")
    
    # Test 5: Hybrid (RRF)
    print(f"\n[Test 5] Hybrid Search (RRF)")
    hybrid_results = retriever.search(test_query, top_k=5)
    print("Top 5 Hybrid Results:")
    for rank, doc in enumerate(hybrid_results, 1):
        print(f"  {rank}. Doc {doc['doc_index']:2d} | RRF Score: {doc['retrieval_score']:.4f}")
        print(f"      Q: {doc['question'][:80]}...")
    
    print("\n ALL TESTS PASSED")
    print("="*60)


# =================== Ablation Class (Added) ===================

class RetrievalAblation:
    """Ablation study comparing retrieval methods"""
    
    def __init__(self, index_dir: str = "data/indices"):
        self.index_dir = Path(index_dir)
        self.bm25_index = None
        self.faiss_index = None
        self.documents = None
        self.encoder = None
        
    def load_indices(self):
        """Load pre-built indices"""
        print("[1] Loading indices...")
        
        # Load BM25
        with open(self.index_dir / "bm25_index.pkl", 'rb') as f:
            self.bm25_index = pickle.load(f)
        print("    BM25 index loaded")
        
        # Load FAISS
        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss_index.bin"))
        print("    FAISS index loaded")
        
        # Load documents
        with open(self.index_dir / "documents.json", 'r') as f:
            self.documents = json.load(f)
        print(f"    {len(self.documents)} documents loaded")
        
        # Load encoder for FAISS queries
        self.encoder = SentenceTransformer(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )
        print("    Encoder loaded")
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[int]:
        """BM25 sparse retrieval - returns document indices"""
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return list(top_indices)
    
    def faiss_search(self, query: str, top_k: int = 10) -> List[int]:
        """FAISS dense retrieval - returns document indices"""
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        _, indices = self.faiss_index.search(query_embedding, top_k)
        return list(indices[0])
    
    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[int]:
        """Hybrid search with RRF fusion - returns document indices"""
        # Get more candidates for fusion
        pool_size = min(top_k * 2, len(self.documents))
        
        bm25_results = self.bm25_search(query, top_k=pool_size)
        faiss_results = self.faiss_search(query, top_k=pool_size)
        
        # RRF fusion
        rrf_scores = {}
        
        for rank, doc_id in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        
        for rank, doc_id in enumerate(faiss_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_id for doc_id, _ in sorted_results[:top_k]]
    
    def find_relevant_doc(self, question: str, answer: str) -> int:
        """Find the index of the document containing the answer"""
        answer_lower = answer.lower()
        
        for idx, doc in enumerate(self.documents):
            if answer_lower in doc['text'].lower():
                return idx
        
        return -1  # Not found
    
    def compute_mrr(self, retrieved: List[int], relevant: int) -> float:
        """Compute Mean Reciprocal Rank for a single query"""
        if relevant == -1:
            return 0.0
        
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id == relevant:
                return 1.0 / rank
        
        return 0.0
    
    def compute_recall_at_k(self, retrieved: List[int], relevant: int, k: int) -> float:
        """Compute Recall@K for a single query"""
        if relevant == -1:
            return 0.0
        
        return 1.0 if relevant in retrieved[:k] else 0.0
    
    def compute_ndcg(self, retrieved: List[int], relevant: int, k: int = 10) -> float:
        """Compute NDCG@K for a single query (binary relevance)"""
        if relevant == -1:
            return 0.0
        
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id == relevant:
                dcg = 1.0 / np.log2(rank + 1)
                break
        
        # Ideal DCG (relevant doc at position 1)
        idcg = 1.0 / np.log2(2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_method(self, method_name: str, search_func, test_data: List[Dict], 
                       top_k: int = 10) -> Dict:
        """Evaluate a retrieval method on test data"""
        mrr_scores = []
        recall_at_1 = []
        recall_at_3 = []
        recall_at_5 = []
        recall_at_10 = []
        ndcg_scores = []
        
        for example in tqdm(test_data, desc=f"    Evaluating {method_name}"):
            question = example['question']
            answer = example['answer']
            
            # Find relevant document
            relevant_idx = self.find_relevant_doc(question, answer)
            
            # Retrieve documents
            retrieved = search_func(question, top_k=top_k)
            
            # Compute metrics
            mrr_scores.append(self.compute_mrr(retrieved, relevant_idx))
            recall_at_1.append(self.compute_recall_at_k(retrieved, relevant_idx, 1))
            recall_at_3.append(self.compute_recall_at_k(retrieved, relevant_idx, 3))
            recall_at_5.append(self.compute_recall_at_k(retrieved, relevant_idx, 5))
            recall_at_10.append(self.compute_recall_at_k(retrieved, relevant_idx, 10))
            ndcg_scores.append(self.compute_ndcg(retrieved, relevant_idx, top_k))
        
        return {
            'method': method_name,
            'mrr_mean': np.mean(mrr_scores),
            'mrr_std': np.std(mrr_scores),
            'mrr_scores': mrr_scores,
            'recall@1': np.mean(recall_at_1),
            'recall@3': np.mean(recall_at_3),
            'recall@5': np.mean(recall_at_5),
            'recall@10': np.mean(recall_at_10),
            'ndcg@10': np.mean(ndcg_scores),
            'ndcg_std': np.std(ndcg_scores),
            'ndcg_scores': ndcg_scores,
            'n_examples': len(test_data)
        }
    
    def run_ablation(self, test_file: str) -> Dict:
        """
        Run full ablation study
        Args:
            test_file: Path to test dataset JSON file
        Returns: 
            Dict: Results for each retrieval method
            
        """
        print("\n" + "="*60)
        print("RETRIEVAL ABLATION STUDY")
        print("="*60)
        
        # Load test data
        print(f"\n[2] Loading test data: {test_file}")
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        print(f"    {len(test_data)} test examples")
        
        # Evaluate each method
        print("\n[3] Evaluating retrieval methods...")
        
        results = {}
        
        # BM25 only
        results['bm25'] = self.evaluate_method(
            'BM25', self.bm25_search, test_data
        )
        
        # FAISS only
        results['faiss'] = self.evaluate_method(
            'FAISS', self.faiss_search, test_data
        )
        
        # Hybrid
        results['hybrid'] = self.evaluate_method(
            'Hybrid (RRF)', self.hybrid_search, test_data
        )
        
        return results
    
    def statistical_comparison(self, results: Dict) -> Dict:
        """Statistical comparison between methods
        Args:
            results: Dict of results from run_ablation
        Returns:
            Dict: Statistical comparison results"""
        print("\n[4] Statistical comparison...")
        
        comparisons = {}
        
        # Compare Hybrid vs BM25 (MRR)
        hybrid_mrr = results['hybrid']['mrr_scores']
        bm25_mrr = results['bm25']['mrr_scores']
        t_stat, p_value = stats.ttest_rel(hybrid_mrr, bm25_mrr)
        
        comparisons['hybrid_vs_bm25_mrr'] = {
            'hybrid_mean': np.mean(hybrid_mrr),
            'bm25_mean': np.mean(bm25_mrr),
            'improvement': (np.mean(hybrid_mrr) - np.mean(bm25_mrr)) / np.mean(bm25_mrr) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Compare Hybrid vs FAISS (MRR)
        faiss_mrr = results['faiss']['mrr_scores']
        t_stat, p_value = stats.ttest_rel(hybrid_mrr, faiss_mrr)
        
        comparisons['hybrid_vs_faiss_mrr'] = {
            'hybrid_mean': np.mean(hybrid_mrr),
            'faiss_mean': np.mean(faiss_mrr),
            'improvement': (np.mean(hybrid_mrr) - np.mean(faiss_mrr)) / np.mean(faiss_mrr) * 100 if np.mean(faiss_mrr) > 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Compare Hybrid vs BM25 (NDCG)
        hybrid_ndcg = results['hybrid']['ndcg_scores']
        bm25_ndcg = results['bm25']['ndcg_scores']
        t_stat, p_value = stats.ttest_rel(hybrid_ndcg, bm25_ndcg)
        
        comparisons['hybrid_vs_bm25_ndcg'] = {
            'hybrid_mean': np.mean(hybrid_ndcg),
            'bm25_mean': np.mean(bm25_ndcg),
            'improvement': (np.mean(hybrid_ndcg) - np.mean(bm25_ndcg)) / np.mean(bm25_ndcg) * 100 if np.mean(bm25_ndcg) > 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return comparisons


def print_results(results: Dict, comparisons: Dict):
    """Print formatted results"""
    print("\n" + "="*60)
    print("RETRIEVAL ABLATION RESULTS")
    print("="*60)
    
    print("\n[5] Performance Metrics:")
    print("-"*60)
    print(f"{'Method':<15} {'MRR':>10} {'R@1':>10} {'R@5':>10} {'R@10':>10} {'NDCG@10':>10}")
    print("-"*60)
    
    for method in ['bm25', 'faiss', 'hybrid']:
        r = results[method]
        print(f"{r['method']:<15} {r['mrr_mean']:>10.4f} {r['recall@1']:>10.4f} "
              f"{r['recall@5']:>10.4f} {r['recall@10']:>10.4f} {r['ndcg@10']:>10.4f}")
    
    print("\n[6] Statistical Comparisons:")
    print("-"*60)
    
    for name, comp in comparisons.items():
        print(f"\n{name}:")
        print(f"  Improvement: {comp['improvement']:+.2f}%")
        print(f"  t-statistic: {comp['t_statistic']:.3f}")
        print(f"  p-value:     {comp['p_value']:.6f}")
        print(f"  Significant: {'YES' if comp['significant'] else 'NO'} (α=0.05)")


def save_results(results: Dict, comparisons: Dict, output_dir: str = "results"):
    """Save results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as JSON (excluding large score arrays)
    summary = {
        'methods': {},
        'comparisons': comparisons
    }
    
    for method in ['bm25', 'faiss', 'hybrid']:
        r = results[method]
        summary['methods'][method] = {
            'mrr_mean': r['mrr_mean'],
            'mrr_std': r['mrr_std'],
            'recall@1': r['recall@1'],
            'recall@3': r['recall@3'],
            'recall@5': r['recall@5'],
            'recall@10': r['recall@10'],
            'ndcg@10': r['ndcg@10'],
            'ndcg_std': r['ndcg_std'],
            'n_examples': r['n_examples']
        }
    
    # Convert numpy types to native Python types
    def convert(o):
        if isinstance(o, (np.integer, np.int64)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, (np.bool_)): return bool(o)
        return str(o)
    
    with open(output_path / 'retrieval_ablation.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    
    print(f"\n Results saved to: {output_path / 'retrieval_ablation.json'}")
    

def main():
    """Main execution"""
    # Initialize
    ablation = RetrievalAblation()
    ablation.load_indices()
    
    # Run ablation
    results = ablation.run_ablation("data/processed/bioasq_test.json")
    
    # Statistical comparison
    comparisons = ablation.statistical_comparison(results)
    
    # Print results
    print_results(results, comparisons)
    
    # Save results
    save_results(results, comparisons)
    
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    
    # Summary for hypothesis testing
    print("\n HYPOTHESIS H2 ASSESSMENT:")
    print("-"*40)
    hybrid_mrr = results['hybrid']['mrr_mean']
    bm25_mrr = results['bm25']['mrr_mean']
    faiss_mrr = results['faiss']['mrr_mean']
    
    print(f"Hybrid MRR:  {hybrid_mrr:.4f}")
    print(f"BM25 MRR:    {bm25_mrr:.4f}")
    print(f"FAISS MRR:   {faiss_mrr:.4f}")
    
    improvement_bm25 = (hybrid_mrr - bm25_mrr) / bm25_mrr * 100
    improvement_faiss = (hybrid_mrr - faiss_mrr) / faiss_mrr * 100 if faiss_mrr > 0 else 0
    
    print(f"\nImprovement over BM25:  {improvement_bm25:+.2f}%")
    print(f"Improvement over FAISS: {improvement_faiss:+.2f}%")
    
    if comparisons['hybrid_vs_bm25_mrr']['significant']:
        print("\n H2 SUPPORTED: Hybrid significantly outperforms BM25")
    else:
        print("\n H2 NOT SUPPORTED: Difference not statistically significant")


if __name__ == "__main__":
    main()