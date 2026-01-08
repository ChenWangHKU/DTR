import math
from typing import List, Dict, Literal, Optional, Any, Tuple
import re
import torch
import time
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams



class AdaptiveRAG:
    """
        Fast RAG that can dynamically retrieve the top-k documents according to the LLM's estimation.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 1024,
        USE_VLLM: bool = True,
        gpu_size: int = 2,
        GPUID: int = 0
        ) -> None:
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.use_vllm = USE_VLLM

        # Initialize the tokenizer and model
        """ We use vLLM to achieve batch inference """
        if self.use_vllm:
            # self.model = LLM(model=model_name_or_path, tensor_parallel_size=1)
            self.model = LLM(
                model=model_name_or_path,
                tensor_parallel_size=gpu_size,
                gpu_memory_utilization=0.7, # Reduced from default 0.9
                max_num_seqs=16,            # Limit concurrent sequences
                max_model_len=7200,         # Start with smaller context
                )
        # Basic case
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map={"": f"cuda:{GPUID}"}).eval()
        
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, device_map={"": f"cuda:{GPUID}"})
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        
        

    # Get prompt for direct question answering (without retrieval)
    def get_prompt_direct(self, q: str) -> str:

        prompt = f"""
            Question: {q}\n
            Answer the question using a single word or phrase.
        """

        message = [
            {"role": "user", "content": prompt}
        ]

        return message

    
    # Get prompt for generation based on retrieved documents
    def get_prompt_with_docs(self, q: str, docs: List[str]) -> str:
        """
            q: question
            docs: list of retrieved documents
        """
        # wiki style
        if type(docs[0]) == dict:
                context = []
                for doc in docs:
                    title = doc["title"]
                    content = doc["text"]
                    context.append(f"Title: {title}. Content: {content}")
                context = "\n".join(context)
        # plain style
        else:
            context = "\n".join(docs)
        
        prompt = f"""
            Question: {q}\n
            Context: {context}\n
            Answer the question based on the above context using a single word or phrase.
        """

        message = [
            {"role": "user", "content": prompt}
        ]
        return message


    # Prompt for generating pseudo-context
    def get_prompt_pseudo_context(self, q: str) -> str:
        prompt = f"""
            Please write a passage to answer the question\n
            Question: {q}\n
            Passage:
        """
        message = [
            {"role": "user", "content": prompt}
        ]
        return message


    # Prompt for judging whether retrieval is needed by LLM itself
    def get_prompt_judge(self, q: str) -> str:
        prompt = f"""
            Question: {q}\n
            Determine whether external information is needed to answer the question accurately. 
            Respond with "Yes" if additional information is required, or "No" if the question can be answered without it.
        """

        message = [
            {"role": "user", "content": prompt}
        ]

        return message



    # Generate the response of a given question using the LLM
    def generate_response(self, q: str):
        """
            q: question
            docs: list of retrieved documents. None means no retrieval.
            prev_eval: whether it is an pre-evaluation of the given question
        """
        
        # Prepare the prompt
        user_msg = self.get_prompt_direct(q)
        
        # apply_chat_template is the officially recommended way to prepare input
        if "Qwen3" in self.model_name_or_path:
            prompt_text = self.tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
        else:
            prompt_text = self.tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True
            )
        # Tokenize
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)
        full_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return full_text
    



    # Generate the response of a batch of questions using the vLLM
    def generate_response_batch(
            self,
            qs: List[str],
            judge: bool = False,
            pseudo_context: bool = False,
            docs = None,
            num_responses: int = 1,
            randomness: bool = False,
            need_prob: bool = True
            ) -> Tuple[List[List[str]], List[List[float]]]:
        """
            q: a batch of questions
            judge: whether to generate a judgment on the need for retrieval
            pseudo_context: whether to generate pseudo-context passages
            docs: list of retrieved documents for each question. None means no retrieval.
            num_responses: the number of responses to generate for each query
            randomness: whether to use randomness in the generation process
        """
        
        # Make sure q is a list
        if not isinstance(qs, list):
            qs = [qs]

        # Prepare the input
        messages = []
        for i, q in enumerate(qs):
            # Prepare the prompt
            if docs is not None:
                user_msg = self.get_prompt_with_docs(q, docs[i])
            elif judge:
                user_msg = self.get_prompt_judge(q)
            elif pseudo_context:
                user_msg = self.get_prompt_pseudo_context(q)
            else:
                user_msg = self.get_prompt_direct(q)
            
            # apply_chat_template is the officially recommended way to prepare input
            msg = self.tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True
            )
            # Build a single-turn ChatML prompt
            messages.append(msg)

        # Considering the randomness
        if randomness:
            tmp, top_p, top_k = 0.7, 0.8, 20
        # Using greedy sampling
        else:
            tmp, top_p, top_k = 0, 1.0, 1
        # parameters
        sampling_params = SamplingParams(
            n=num_responses, # How many responses to generate for each query
            temperature=tmp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            max_tokens=self.max_new_tokens,
            stop=["<|im_end|>"],  # vLLM stops when any token is generated
            logprobs=1  # To get the logprobs of top 20 tokens
        )
        # Generation
        outputs = self.model.generate(messages, sampling_params)
        
        # Extract both responses and probabilities
        responses_all = []
        probs_all = []
        
        for output in outputs:
            # Each output corresponds to one input query
            query_responses = []
            query_probs = []
            
            for gen in output.outputs:
                """
                CompletionOutput(index=0, text='No', token_ids=[2753, 151645], cumulative_logprob=-0.0028615298797376454,
                logprobs=[{2753: Logprob(logprob=-0.0007790867821313441, rank=1, decoded_token='No')}, {151645: Logprob(logprob=-0.0020824430976063013, rank=1, decoded_token='<|im_end|>')}],
                finish_reason=stop, stop_reason=None)
                """
                # Get the generated text
                generated_text = gen.text.strip()
                query_responses.append(generated_text)
                
                # Calculate the probabilities of tokens if needed
                if need_prob:
                    # Check if logprobs are available
                    if gen.logprobs:
                        for token_logprobs in gen.logprobs:
                            if token_logprobs:  # Not empty dict
                                # token_logprobs is {token_id: Logprob(...)}
                                # Get the Logprob object (there's usually just one for greedy/argmax)
                                logprob_obj = next(iter(token_logprobs.values()))
                                
                                # Access the logprob attribute
                                logprob_value = logprob_obj.logprob
                                
                                # Convert to probability
                                probability = math.exp(logprob_value)
                                query_probs.append(probability)
                            else:
                                # Empty dict for this token position
                                query_probs.append(0.0)
                        
                    else:
                        # No logprobs available
                        print("WARNING: No logprobs returned. Make sure 'logprobs' is set in SamplingParams.")
                        query_probs.append([])
            
            responses_all.append(query_responses)
            probs_all.append(query_probs)
        
        return responses_all, probs_all
        