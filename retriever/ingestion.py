from typing import List, Dict, Literal, Optional, Any, Tuple
import re
import torch
from tqdm import tqdm
import numpy as np
import os
import faiss
from concurrent.futures import ThreadPoolExecutor
import argparse

from sentence_transformers import SentenceTransformer



"""
    hotpotqa and 2wikimultihopqa corpus example:
    {'id': '2', 'title': 'Move (1970 film)',
    'sentences': ['Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.',
    'The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber.']}
    
    21M Wikipedia corpus includes "id", "title", and "text" with the dataframe format.
    a document is split into several "text" (similar length) with the same "title".
"""
# We have unified the corpus format in dataset.load_data.py
# You may revise the paths according to your data storage.





EMB_MAP = {
    "bge": "BAAI/bge-base-en-v1.5",
    "e5": "intfloat/e5-base-v2",
}


class Ingestion:
    """Embed and index a corpus; support batch and top-k retrieval without LLM."""
    def __init__(
        self,
        emb_type: Literal["bge", "e5"] = "bge",
        data_root: str = "./data"
    ) -> None:
        
        # root directory
        self.root = data_root
        os.makedirs(self.root, exist_ok=True)
        # Embedding model
        self.emb_name = EMB_MAP[emb_type]
        
    

    """
        These funsctions are used to embed and index the corpus in advance. The index is built by FAISS and saved to disk.
        When using this retriever, we can load index from disk directly, without needing to embed the corpus again.
    """
    # Embed documents and save to disk.
    # Refernce: https://huggingface.co/BAAI/bge-large-en-v1.5
    def embedding(self, corpus: List[Dict[str, Any]], batch_size: int = 128, GPUID: int = 0) -> None:
        print("Embedding...")
        texts = [doc["text"] for doc in corpus]
        
        model = SentenceTransformer(self.emb_name)
        
        # Embed the corpus in batch
        embeddings_all =[]
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch embeding"):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(
                batch,
                device=f"cuda:{GPUID}",
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings_all.append(embeddings)
        
        embeddings = np.vstack(embeddings_all)
        print("Final shape of the embeddings:", embeddings.shape)

        # Save to disk for reuse
        np.save(os.path.join(self.root, "corpus_embeddings.npy"), embeddings)
        
        print(f'Saved embeddings to {os.path.join(self.root, "corpus_embeddings.npy")}')
    
    

    # Embed documents in parallel ahead of time and save to disk. This can facilitate the embedding process.
    # This function is implemented with the help of ChatGPT and DeepSeek.
    def embedding_parallel(self, corpus: List[Dict[str, Any]], batch_size: int = 128) -> None:
        print("Parallel embedding...")

        texts = [doc["text"] for doc in corpus]

        # Number of GPUs to use
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 0, "No CUDA devices available"
        print(f"Detected and will embed on {num_gpus} GPUs")

        # Split indices across GPUs
        splits = np.array_split(np.arange(len(texts)), num_gpus)

        # ----------------------------------------------------
        # 1. Load one model per GPU (OUTSIDE threads)
        # ----------------------------------------------------
        models = []
        for gpu_id in range(num_gpus):
            device = f"cuda:{gpu_id}"
            print(f"Loading model on {device}")
            model = SentenceTransformer(self.emb_name, device=device)
            models.append(model)

        # ----------------------------------------------------
        # 2. Thread worker (NO model creation)
        # ----------------------------------------------------
        def embed_on_gpu(split_ids, model, device):
            results = []

            for i in range(0, len(split_ids), batch_size):
                batch_ids = split_ids[i:i + batch_size]
                batch = [texts[j] for j in batch_ids]

                emb = model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                results.append(emb)

            return np.vstack(results)

        # ----------------------------------------------------
        # 3. Parallel execution
        # ----------------------------------------------------
        all_embeddings = [None] * num_gpus

        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = {
                executor.submit(embed_on_gpu, splits[i], models[i], f"cuda:{i}"): i
                for i in range(num_gpus)
            }

            for future, idx in futures.items():
                all_embeddings[idx] = future.result()

        # ----------------------------------------------------
        # 4. Concatenate in correct order
        # ----------------------------------------------------
        embeddings = np.vstack(all_embeddings)
        print("Final shape of the embeddings:", embeddings.shape)

        # Save to disk
        save_path = os.path.join(self.root, "corpus_embeddings.npy")
        np.save(save_path, embeddings)
        print(f"Saved embeddings to {save_path}")
        
    


    # Index the embeddings only by FAISS and save to disk ahead of time
    # Reference: https://github.com/facebookresearch/faiss/wiki/Getting-started
    def indexing_emb_faiss(self) -> None:
        print("Building FAISS index...")
        # Load embeddings from disk
        embeddings = np.load(os.path.join(self.root, "corpus_embeddings.npy"))
        print(f"Loaded embeddings shape: {embeddings.shape}")
        
        # Normalize embeddings for Inner Product (IP)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1)  # Avoid division by zero
        
        index = faiss.IndexFlatIP(embeddings.shape[1]) # Brute force with Inner Product (IP)
        # Adding embeddings to the index in batches
        for i in tqdm(range(0, len(embeddings), 10000), desc="Adding embeddings to the index"):
            emb = embeddings[i:i+10000]
            emb = emb.astype('float32')
            if not index.is_trained:
                index.train(emb)
            index.add(emb)

        # Save index to disk
        faiss.write_index(index, os.path.join(self.root, "faiss_index_emb"))
        
        print(f'Saved index to {os.path.join(self.root, "faiss_index_emb")}')



if __name__ == "__main__":
    from dataset.load_data import load_corpus

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_name", type=str, default="hotpotqa", choices=["hotpotqa", "21MWiki"])
    argparser.add_argument("--emb_type", type=str, default="e5", choices=["e5", "bge"])
    args = argparser.parse_args()

    # Example usage
    if args.data_name == "hotpotqa":
        data_name = "hotpotqa"
    elif args.data_name == "21MWiki":
        data_name = "nq"  # or "triviaqa", "webqa", "squad"
    else:
        raise ValueError(f"Unsupported data name: {args.data_name}")

    corpus = load_corpus(data_name=data_name)
    ingestor = Ingestion(emb_type=args.emb_type, data_root=f"data/{data_name}_{args.emb_type}")
    ingestor.embedding_parallel(corpus, batch_size=128)
    ingestor.indexing_emb_faiss()
