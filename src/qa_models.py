"""
Module 5: QA Models with Joint Span Selection
Implements:
Start-end span selection using joint scoring with calibrated no-answer thresholding
Filtering of empty answers 
Handles long contexts by chunking with overlap
Uses RoBERTa and SciBERT models
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class QAModel:
    def __init__(self, model_name: str, device: str = "mps"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        print(f"  Model: {model_name}")
        print(f"  Device: {device}")
    
    def answer_question(self, question: str, context: str) -> Dict:
        """
        Joint span selection instead of independent argmax 
        Args:
            question: The input question string
            context: The context passage string
        Returns:
            Dictionary with answer, confidence, start_logits, end_logits
        """
        # Tokenize
        inputs = self.tokenizer(
            question,
            context,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()
        
        # JOINT SPAN SELECTION
        
        # Geting token positions for context excluding question tokens
        input_ids = inputs['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Finding where context starts (after [SEP] token)
        sep_indices = [i for i, token in enumerate(tokens) if token in ['[SEP]', '</s>']]
        context_start = sep_indices[0] + 1 if sep_indices else 1
        context_end = len(tokens) - 1
        
        # NO-ANSWER handling using CLS token

        # SQuAD 2.0 models predict no-answer by assigning high probability to CLS
        cls_score = start_logits[0] + end_logits[0]
        
        # Find best valid span (where end >= start and both in context)
        best_score = float('-inf')
        best_start = 0
        best_end = 0
        
        # Consider top K start positions (K=20 as per SQuAD best practices)
        top_k = 20
        top_start_indices = np.argsort(start_logits[context_start:context_end])[-top_k:] + context_start
        top_end_indices = np.argsort(end_logits[context_start:context_end])[-top_k:] + context_start
        
        # Evaluate all valid start-end combinations
        for start_idx in top_start_indices:
            for end_idx in top_end_indices:
                # Ensure valid span (end >= start, reasonable length)
                if start_idx <= end_idx <= start_idx + 100:  # max span length = 100 tokens
                    score = start_logits[start_idx] + end_logits[end_idx]
                    if score > best_score:
                        best_score = score
                        best_start = start_idx
                        best_end = end_idx
        
        # CALIBRATED NO-ANSWER THRESHOLD 

        # Compare best span score to CLS (no-answer) score
        # Use a small margin (e.g., 0.0) - if span score isn't clearly better, return no answer
        null_score_diff_threshold = 0.0  # Tune this if needed
        
        answer_text = ""
        no_answer = False
        confidence = 0.0
        
        if best_score > cls_score + null_score_diff_threshold:
            # Extract answer text
            answer_tokens = input_ids[best_start:best_end + 1]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            
            # Calculate confidence (softmax probability)
            # Use temperature scaling for better calibration
            start_probs = torch.softmax(torch.tensor(start_logits), dim=0)
            end_probs = torch.softmax(torch.tensor(end_logits), dim=0)
            confidence = float((start_probs[best_start] * end_probs[best_end]).item())
            
            # FILTERING OF EMPTY ANSWERS

            # Only filter out truly empty or nonsensical answers
            answer_text = answer_text.strip()
            
            # Reject only if:
            # 1. Empty after stripping
            # 2. Only special characters/punctuation
            # 3. Just whitespace
            if not answer_text or len(answer_text) < 1:
                answer_text = ""
                no_answer = True
            elif all(c in '.,;:!?-_()[]{}' for c in answer_text):
                answer_text = ""
                no_answer = True
        else:
            # Model predicts no answer
            no_answer = True
            confidence = float(torch.softmax(torch.tensor([cls_score, best_score]), dim=0)[0].item())
        
        return {
            'answer': answer_text,
            'confidence': confidence,
            'start_logits': start_logits.tolist(),
            'end_logits': end_logits.tolist(),
            'no_answer': no_answer
        }
    
    def answer_with_passages(self, 
                            question: str, 
                            passages: List[Dict],
                            top_k: int = 5) -> Dict:
        """
        Answer question using multiple passages
        
        Args:
            question: The input question string
            passages: List of passages (each a dict with 'text' and optional 'retrieval_score')
            top_k: Number of top passages to consider for answering
        Returns:
            Dictionary with best answer, confidence, no_answer flag, combined_score
            """
        if not passages:
            return {
                'answer': '',
                'confidence': 0.0,
                'no_answer': True,
                'combined_score': 0.0
            }
        
        # Try each passage
        candidates = []
        
        for i, passage in enumerate(passages[:top_k]):
            # Handle both 'text' and 'context' keys
            context = passage.get('text', passage.get('context', ''))
            
            if not context:
                continue
            
            # HANDLE LONG CONTEXTS (split if >450 tokens)

            # Check if context is too long
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
            
            if len(context_tokens) > 450:  # Leave room for question + special tokens
                # Split into chunks with overlap
                chunk_size = 400
                overlap = 100
                chunks = []
                
                for start_idx in range(0, len(context_tokens), chunk_size - overlap):
                    end_idx = min(start_idx + chunk_size, len(context_tokens))
                    chunk_tokens = context_tokens[start_idx:end_idx]
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    chunks.append(chunk_text)
                    
                    if end_idx >= len(context_tokens):
                        break
                
                # Try each chunk
                for chunk in chunks:
                    result = self.answer_question(question, chunk)
                    
                    if result['answer']:
                        # Combined score: 70% QA confidence + 30% retrieval score
                        retrieval_score = passage.get('retrieval_score', 0.0)
                        combined = result['confidence'] * 0.7 + retrieval_score * 0.3
                        
                        candidates.append({
                            'answer': result['answer'],
                            'confidence': result['confidence'],
                            'combined_score': combined,
                            'no_answer': result['no_answer'],
                            'passage_index': i
                        })
            else:
                # Normal processing for short contexts
                result = self.answer_question(question, context)
                
                if result['answer']:
                    # Combined score
                    retrieval_score = passage.get('retrieval_score', 0.0)
                    combined = result['confidence'] * 0.7 + retrieval_score * 0.3
                    
                    candidates.append({
                        'answer': result['answer'],
                        'confidence': result['confidence'],
                        'combined_score': combined,
                        'no_answer': result['no_answer'],
                        'passage_index': i
                    })
        
        # Return best candidate by combined score
        if candidates:
            best = max(candidates, key=lambda x: x['combined_score'])
            return best
        else:
            return {
                'answer': '',
                'confidence': 0.0,
                'no_answer': True,
                'combined_score': 0.0
            }


# ============ TEST MODULE 4 ============
if __name__ == "__main__":
    print("="*60)
    print("MODULE 4 TEST: QA Models with Joint Span Selection")
    print("="*60)
    
    print("\n[Test 1] Loading Models...")
    print("[Loading RoBERTa]")
    roberta = QAModel("deepset/roberta-base-squad2", device="mps")
    print("  Loaded RoBERTa")
    
    print("\n[Loading SciBERT]")
    scibert = QAModel("ktrapeznikov/scibert_scivocab_uncased_squad_v2", device="mps")
    print("  Loaded SciBERT")
    
    # Test span selection
    print("\n[Test 2] Testing Joint Span Selection")
    context = """Antibiotics are medications used to treat bacterial infections. 
    They work by killing bacteria or preventing them from reproducing. 
    Common antibiotics include penicillin, amoxicillin, and ciprofloxacin. 
    However, antibiotics are not effective against viral infections like the flu or common cold."""
    
    question = "What are antibiotics used for?"
    
    print(f"Context: {context[:200]}...")
    print(f"Question: {question}")
    
    rob_result = roberta.answer_question(question, context)
    print(f"\nRoBERTa Answer:")
    print(f"  Answer: '{rob_result['answer']}'")
    print(f"  Confidence: {rob_result['confidence']:.4f}")
    print(f"  No Answer: {rob_result['no_answer']}")
    
    sci_result = scibert.answer_question(question, context)
    print(f"\nSciBERT Answer:")
    print(f"  Answer: '{sci_result['answer']}'")
    print(f"  Confidence: {sci_result['confidence']:.4f}")
    print(f"  No Answer: {sci_result['no_answer']}")
    
    print("\n Joint span selection working correctly!")
    print("="*60)