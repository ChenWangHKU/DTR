import logging
import argparse
import numpy as np 

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from dataset.load_data import load_json
from evaluation.metrics import ems, f1_score



logger = logging.getLogger(__file__)

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_file',
        type=str,
        default='results_hotpotqa_num7404_top2_Qwen2.5-7B-Instruct.json',
        help='The name of the input file.'
    )
    parser.add_argument(
        '--result_root',
        type=str,
        default='../../results',
        help='The root directory for generated results.'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='Number of samples to use in evaluation. -1 means all'
    )
    args = parser.parse_args()
    return args

    


def main():
    args = setup_args()
    file_path = args.eval_file
    print(f"loading data: {file_path}")
    data = load_json(file_path)
    
    if args.num_samples > 0:
        data = data[:args.num_samples]

    em_scores, f1_scores = [], []
    used_retrieval = 0
    for example in data:
        gold_answers = example["expected_answer"]
        # Ensure gold_answers is a list
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        
        model_answer = example["model_answer"]

        ems_score = max(ems(model_answer[0], ga) for ga in gold_answers)
        f1 = max(f1_score(model_answer[0], ga)[0] for ga in gold_answers)
        em_scores.append(ems_score)
        f1_scores.append(f1)

        if example["used_retrieval"]:
            used_retrieval += 1


    
    avg_ems = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)
    retrieval_rate = used_retrieval / len(data)


    print("==================== Evaluation Result ====================")
    print(">>>> File: {}".format(args.eval_file))
    print(">>>> EM: {:.5f}".format(avg_ems))
    print(">>>> F1: {:.5f}".format(avg_f1))
    print(">>>> Retrieval Rate: {:.5f}".format(retrieval_rate))
    print("===========================================================")
     

     

if __name__ == "__main__":
    
    main()