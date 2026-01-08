"""
    Evaluate the adaptive RAG model on question answering datasets.
    (1) Generating answers directly
    (2) Generating answers with retrieved documents
"""

import json
import os
from statistics import mean
from tqdm import tqdm
import argparse
import numpy as np

from dataset.load_data import load_test_qa, load_corpus, get_index_path
from agent.adaptive_rag import AdaptiveRAG
from retriever.retriever import DocumentRetriever
from agent.reranker import select_topk_of_query_info



def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path', 
        type=str, 
        default='Qwen/Qwen2.5-7B-Instruct',
        help='The name or path of the pre-trained model.'
    )
    parser.add_argument(
        '--data_name',
        type=str,
        default='hotpotqa',
        help='The dataset name for evaluation.'
    )
    parser.add_argument(
        '--uncertainty_threshold',
        type=float,
        default=0.001,
        help='The uncertainty threshold for retrieval-augmented generation (RAG).'
    )
    parser.add_argument(
        '--emb_type',
        type=str,
        default='bge',
        choices=['bge', 'e5'],
        help='The type of embedding model to use for the retriever.'
    )
    parser.add_argument(
        '--RAG_type',
        type=str,
        default='adaptive',
        choices=['adaptive', 'standard'],
        help='The type of RAG model to use.'
    )
    parser.add_argument(
        '--topn',
        type=int,
        default=5,
        help='Initial number of documents to retrieve for each query. Used when using reranking (automatically using reranking when topn > topk).'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=3,
        help='Number of top documents to retrieve.'
    )
    parser.add_argument(
        '--result_root',
        type=str,
        default='../results',
        help='The root directory for generated results.'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='Number of samples to use in evaluation. -1 means all'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Number of queries per batch'
    )
    parser.add_argument(
        '--index_GPUID',
        type=int,
        default=2,
        help='Load index onto GPU'
    )
    parser.add_argument(
        '--gpu_size',
        type=int,
        default=2,
        help='Number of GPUs to use for tensor parallelism'
    )
    parser.add_argument(
        '--use_vLLM',
        type=bool,
        default=True,
        help='Whether to use vLLM'
    )
    args = parser.parse_args()
    return args



def eval():
    args = setup_args()

    print(f"Initialize Adaptive agent")
    # Initialize FastRAG
    agent = AdaptiveRAG(
        model_name_or_path=args.model_name_or_path,
        max_new_tokens=1024,
        USE_VLLM=args.use_vLLM,
        gpu_size=args.gpu_size
    )
    
    print(f"Load test data...")
    qa_pairs = load_test_qa(args.data_name, args.num_samples)
    print(f"{len(qa_pairs)} test samples loaded")
    print(f"Total number of examples: {len(qa_pairs)}")
    print('-'*50)


    # Initialize document retriever if RAG is used
    print(f"Load corpus...")
    corpus = load_corpus(args.data_name)
    print(f"{len(corpus)} documents loaded")

    print(f"Initialize document retriever")
    # Initialize document retriever
    doc_retriever = DocumentRetriever(
        emb_type=args.emb_type,
        topk=args.topk,
        corpus=corpus,
        GPUID=args.index_GPUID
    )
    
    print("Loading faiss index...")
    # Load indexed embeddings (this may take some time)
    doc_retriever.load_index_emb(path=get_index_path(args.data_name), load_embeddings=True)


    # evaluate
    results = []
    for idx in tqdm(range(0, len(qa_pairs), args.batch_size), desc="Evaluating in batches"):
        # We process a batch of queries using vLLM to accelerate the process
        batch = qa_pairs[idx: idx + args.batch_size]
        queries = [item["question"] for item in batch]
        answers = [item["answers"] for item in batch]

        # Initial generation (no retrieval)
        responses_init, probs_init = agent.generate_response_batch(
            queries, num_responses=1, randomness=False
        )

        # Compute uncertainty for each query (based on initial generation)
        uncertainties = []
        for probs in probs_init:
            # -log(p) averaged over tokens (omit last token assumed EOS)
            token_nll = [(-np.log(p) if p and p > 0 else 0.0) for p in probs]
            u = mean(token_nll[:-1]) if token_nll else 0.0
            uncertainties.append(u)

        # Decide which queries need retrieval
        need_rag_indices = [i for i, u in enumerate(uncertainties) if u >= args.uncertainty_threshold]

        # First, record those that do NOT need RAG (use initial results)
        for i, (q, a, resp, probs, u) in enumerate(zip(queries, answers, responses_init, probs_init, uncertainties)):
            if u < args.uncertainty_threshold:
                results.append({
                    "question": q,
                    "expected_answer": a,
                    "model_answer": resp,
                    "model_answer_token_probs": probs,
                    "retrieved_docs": None,
                    "retrieved_indices": None,
                    "uncertainty": u,
                    "used_retrieval": False,
                })

        # For those needing RAG, retrieve + re-generate, then record final results
        if need_rag_indices:
            q_need_rag = [queries[i] for i in need_rag_indices]
            a_need_rag = [answers[i] for i in need_rag_indices]
            u_init_need_rag = [uncertainties[i] for i in need_rag_indices]
            # Standard RAG
            if args.RAG_type == "standard":
                retrieved_docs_batch, retrieved_indices_batch = doc_retriever.batch_retrieve(
                    q_need_rag, k=args.topk
                )
            else:
                """ Dual-path retrieval """
                # First generate pseudo-context
                infos, _ = agent.generate_response_batch(q_need_rag, pseudo_context=True, need_prob=False)
                infos = [info[0] if isinstance(info, list) else info for info in infos]
                
                # Retrieve top-n documents for each query and its corresponding info (batch_size, topn)
                doc_ids_query, doc_embs_query, batch_D_query, query_embs = doc_retriever.batch_retrieve(q_need_rag, args.topn, return_nodes=True)
                doc_ids_info, doc_embs_info, batch_D_info, info_embs = doc_retriever.batch_retrieve(infos, args.topn, return_nodes=True)
                
                # Select top-k documents among the retrieved documents using query and info
                selected_doc_ids, selected_scores = select_topk_of_query_info(
                    doc_ids_query,
                    doc_ids_info,
                    doc_embs_query,
                    doc_embs_info,
                    batch_D_query,
                    batch_D_info,
                    query_embs,          # Embeddings of the original queries
                    info_embs,           # Embeddings of the original info
                    topk_new=args.topk,  # Number of documents to select
                    consider_adaptive=False
                )

                retrieved_docs_batch = [[corpus[id] for id in ids] for ids in selected_doc_ids]
                retrieved_indices_batch = [[int(id) for id in ids] for ids in selected_doc_ids]

            responses_rag, probs_rag = agent.generate_response_batch(
                q_need_rag,
                docs=retrieved_docs_batch,
                num_responses=1,
                randomness=False,
            )

            # (Optional) you can store the uncertainty of the final answer too.
            for q, a, u_init, docs, idxs, resp, probs in zip(
                q_need_rag, a_need_rag, u_init_need_rag,
                retrieved_docs_batch, retrieved_indices_batch,
                responses_rag, probs_rag
            ):
                token_nll = [(-np.log(p) if p and p > 0 else 0.0) for p in probs]
                u_final = mean(token_nll[:-1]) if token_nll else 0.0

                results.append({
                    "question": q,
                    "expected_answer": a,
                    "model_answer": resp,                  # final generation with retrieval
                    "model_answer_token_probs": probs,
                    "retrieved_docs": docs,
                    "retrieved_indices": idxs,
                    "uncertainty": u_init,                 # uncertainty used for triggering
                    "uncertainty_final": u_final,          # optional, for analysis
                    "used_retrieval": True,
                })


    # Save results
    os.makedirs(args.result_root, exist_ok=True)
    RAG_num = args.topk if args.RAG else 0
    file_name = f"gene_{args.data_name}_num{len(results)}_RAG{RAG_num}_" + args.model_name_or_path.split('/')[-1] + ".json"
    
    with open(os.path.join(args.result_root, file_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Result saved to {os.path.join(args.result_root, file_name)}")
    


    



if __name__ == "__main__":
    eval()
    
