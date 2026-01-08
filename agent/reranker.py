"""Reranker module for selecting top-k documents from retrieved results.

This module provides functionality to merge and rerank documents retrieved from
both query and information retrieval systems, computing missing similarities dynamically.
"""


from typing import List, Dict, Literal, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import torch
from modelscope import AutoModelForSequenceClassification, AutoTokenizer



"""
    Rerank retrieved docs using bge-reranker-v2-m3
"""
class rerank_bge:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-base",
        GPUID: int = 0
        ) -> None:
        
        print(f"Initializing reranker with {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.device = f"cuda:{GPUID}"
        self.model = self.model.to(self.device)
        self.model.eval()

    # Rerank retrieved docs using bge-reranker-v2-m3
    def __call__(
        self,
        pairs: List[List[str]] # [[query, doc], ...]
        ):
        
        with torch.no_grad():
            # Tokenize with proper padding and truncation
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu().detach().numpy()
        
        return scores
    
    




"""
    Select topk_new documents from the retrieved documents of query and info (two groups of docs) using Inner Product (IP) metric
"""
def select_topk_of_query_info(
    doc_ids_query,         # np.array (shape: [batch_size, topn_retrieve]) (query-based documents ids)
    doc_ids_info,          # np.array (shape: [batch_size, topn_retrieve]) (info-based documents ids)
    doc_embeddings_query,  # np.array (shape: [batch_size, topn_retrieve, embed_dim])
    doc_embeddings_info,   # np.array (shape: [batch_size, topn_retrieve, embed_dim])
    batch_D_query,         # np.array (shape: [batch_size, topn_retrieve]) (Inner Product)
    batch_D_info,          # np.array (shape: [batch_size, topn_retrieve]) (Inner Product)
    query_embeddings,      # np.array (shape: [batch_size, embed_dim]) (query embeddings)
    info_embeddings,       # np.array (shape: [batch_size, embed_dim]) (info embeddings)
    topk_new,              # Final number of documents to select per query
    consider_adaptive: bool = False
):
    """
    Selects top-k documents from all retrieved docs (query + info), computing missing similarities on-the-fly.
    For documents not in one retrieval, their similarity to the missing term is computed dynamically.
    """
    batch_size = len(doc_ids_query)
    selected_doc_ids = []
    selected_scores = []

    for i in range(batch_size):
        # Step 1: Merge all unique documents and track their embeddings/scores
        doc_to_data = defaultdict(dict)
        
        # Add query-retrieved docs
        for doc_id, emb, s1 in zip(doc_ids_query[i], doc_embeddings_query[i], batch_D_query[i]):
            doc_to_data[doc_id]["s1"] = s1
            doc_to_data[doc_id]["emb"] = emb  # Store embedding for missing-s2 computation
        
        # Add info-retrieved docs
        for doc_id, emb, s2 in zip(doc_ids_info[i], doc_embeddings_info[i], batch_D_info[i]):
            doc_to_data[doc_id]["s2"] = s2
            if "emb" not in doc_to_data[doc_id]:
                doc_to_data[doc_id]["emb"] = emb  # Store embedding for missing-s1 computation

        # Step 2: Compute missing similarities dynamically
        merged_doc_ids = []
        merged_scores = []
        
        query_emb = query_embeddings[i]  # Shape: [embed_dim]
        info_emb = info_embeddings[i]    # Shape: [embed_dim]

        for doc_id, data in doc_to_data.items():
            emb = data["emb"]
            
            # Compute s1 if missing (doc_id not in query retrieval)
            s1 = data.get("s1")
            if s1 is None:
                s1 = np.dot(emb, query_emb)  # Inner Product = Cosine sim (if normalized)
            
            # Compute s2 if missing (doc_id not in info retrieval)
            s2 = data.get("s2")
            if s2 is None:
                s2 = np.dot(emb, info_emb)
            
            # Clip and calculate score
            s1 = np.clip(s1, -1, 1)
            s2 = np.clip(s2, -1, 1)
            if consider_adaptive:
                s0 = np.dot(query_embeddings[i], info_embeddings[i])
                s0 = np.arccos(s0)
                s1 = np.arccos(s1)
                s2 = np.arccos(s2)
                alpha = 0.05848 * s0 + 0.45520
                score = -(alpha * s1 + (1 - alpha) * s2)
            else:
                # score = s1 * s2 - np.sqrt(1 - s1 ** 2) * np.sqrt(1 - s2 ** 2)
                score = s1 + s2
            
            merged_doc_ids.append(doc_id)
            merged_scores.append(score)

        # Step 3: Select top-k
        topk_indices = np.argsort(-np.array(merged_scores))[:topk_new]
        selected_doc_ids.append([merged_doc_ids[idx] for idx in topk_indices])
        selected_scores.append([merged_scores[idx] for idx in topk_indices])

    return selected_doc_ids, np.array(selected_scores)