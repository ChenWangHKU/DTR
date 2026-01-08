from typing import List, Dict, Literal, Optional, Any, Tuple
import re
import numpy as np
import os
import json
import time
import faiss

from llama_index.core.schema import Node
from sentence_transformers import SentenceTransformer

from .ingestion import Ingestion




"""
    hotpotqa corpos example:
    {'id': '2', 'title': 'Move (1970 film)',
    'sentences': ['Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.',
    'The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber.']}

    2wiki includes "id", "title", and "text" with the Dataframe format.
    a document was split into several "text" (similar length) with the same "title".
"""


EMB_MAP = {
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-base-v2",
}


class DocumentRetriever:
    """Embed and index a corpus; support batch and top-k retrieval without LLM."""
    def __init__(
        self,
        emb_type: Literal["bge", "e5"] = "e5",
        data_root: str = "../../data/hotpotqa",
        topk = 0,
        corpus = None,
        GPUID: int = 0
    ) -> None:
        
        # root directory
        self.root = data_root
        self.topk = topk

        # Embedding model
        self.emb_name = EMB_MAP[emb_type]
        self.embed = SentenceTransformer(self.emb_name)
        
        # Database
        self.corpus = corpus
        
        # GPUID
        self.GPUID = GPUID

        # Index
        self.index = None
        self.retriever = None
        self.embeddings = None



    # Load index with embeddings only
    # This index can be used on GPU and can only search relevant embeddings (we need to fetch documents from corpus manually)
    def load_index_emb(self, path: str = None, load_embeddings: bool = False):
        if path is None:
            path = os.path.join(self.root, "faiss_index_emb")
        print(f"Loading index from {path} onto CUDA:{self.GPUID} ")
        
        start_time = time.time()
        # gpu faiss
        res = faiss.StandardGpuResources()   # Initialize a StandardGpuResources object
        read_index = faiss.read_index(path)  # Read the index from the specified path
        self.index = faiss.index_cpu_to_gpu(res, self.GPUID, read_index)  # Move the index to the specified GPU

        end_time = time.time()
        print('time consuming: {:.1f} seconds'.format(end_time - start_time))
        print("-"*50)

        # We also load the embeddings if specified, which is useful for retrieval embdeddings
        if load_embeddings:
            emb_path = path.split("faiss_index_emb")[0] + "corpus_embeddings.npy"
            self.embeddings = np.load(emb_path)
            print(f"Loaded {len(self.embeddings)} embeddings from {emb_path}")



    # Retrieve top-k documents from the index for each query in a batch
    """
        The key idea is to embed the query in batch, then search top k relevant embeddings.
        Finally, fetch the corresponding documents from the corpus based on the indices.
    """
    def batch_retrieve(self, queries: List[str], k: Optional[int] = None, return_nodes: bool = False) -> List[List[Node]]:
        # Embed the query in batch
        query_embeddings = self.embed.encode(
            queries,
            device=f"cuda:{self.GPUID}",
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
            )
        # Search top k relevant embeddings
        k = k if k is not None else self.topk
        batch_D, batch_I = self.index.search(query_embeddings, k) # [batch_size, k]
        

        if return_nodes:  
            # Fetch embeddings for postprocessing
            embs = []
            for i in range(len(queries)):
                if self.embeddings is not None:
                    embs.append([self.embeddings[idx] for idx in batch_I[i]])
                else:
                    embs.append([self.index.reconstruct(idx) for idx in batch_I[i]])
            return batch_I, np.array(embs), batch_D, query_embeddings
        
        # Only return documents when no embeddings are loaded
        else:
            # Fetch the corresponding documents
            docs = []
            doc_idx = []
            for i in range(len(queries)):
                docs.append([self.corpus[idx] for idx in batch_I[i]])
                doc_idx.append([int(idx) for idx in batch_I[i]])
            return docs, doc_idx



    # Retrieve top-k documents from the index for each hybrid embedding in a batch
    def HyDE_retrieve(self, queries: List[str], infos: List[List[str]], k: Optional[int] = None, return_nodes: bool = False) -> List[List[Node]]:
        # Embed the query in batch
        query_embeddings = self.embed.encode(
            queries,
            device=f"cuda:{self.GPUID}",
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
            )
        for i in range(len(queries)):
            info = infos[i] if isinstance(infos[i], list) else [infos[i]]
            # Embed the info in batch
            info_embeddings = self.embed.encode(
                info,
                device=f"cuda:{self.GPUID}",
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
                )
            # Concatenate the query and info embeddings along axis 0
            hybrid_embeddings = np.concatenate((query_embeddings[i:i+1], info_embeddings))
            # Calculate the average embeddings
            query_embeddings[i] = np.mean(hybrid_embeddings, axis=0)

        # Search top k relevant embeddings
        k = k if k is not None else self.topk
        batch_D, batch_I = self.index.search(query_embeddings, k)
        
    
        # Fetch the corresponding index, embeddings and query embeddings
        if return_nodes:  
            # Fetch embeddings for postprocessing
            embs = []
            for i in range(len(queries)):
                embs.append([self.index.reconstruct(int(idx)) for idx in batch_I[i]])
            embs = np.array(embs)
            return batch_I, embs, batch_D, query_embeddings, info_embeddings

        else:
            # Fetch the corresponding documents
            docs = []
            for i in range(len(queries)):
                docs.append([self.corpus[idx] for idx in batch_I[i]])
            
            return docs
