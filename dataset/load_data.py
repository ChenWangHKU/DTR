"""
    This module implements loading data accoding to the given name or path.
"""

import json
import pandas as pd



################################################################
# You may revise the paths according to your data storage.
################################################################
data_name_2_path = {
    # Multi-hop QA
    "hotpotqa": {
        "test": "data/hotpotqa/test_qa_pairs.json",
        "corpus": "data/hotpotqa/corpus.json",
        "index": "data/hotpotqa_bge/faiss_index_emb",
        #"index": "data/hotpotqa_e5/faiss_index_emb"
    },
    
    # Open-domain QA (all use the same corpus)
    "triviaqa": {
        "test": "data/TriviaQA/unfiltered-web-dev.json",
        "corpus": "data/21MWiki/psgs_w100.tsv",
        "index": "data/21MWiki_bge/faiss_index_emb"
        # "index": "data/21MWiki_e5/faiss_index_emb"
    },
    "nq":{
        "test": "data/nq/nq-test-contriever.json",
        "corpus": "data/21MWiki/psgs_w100.tsv",
        "index": "data/21MWiki_bge/faiss_index_emb"
        # "index": "data/21MWiki_e5/faiss_index_emb"
    },
    "webqa": {
        "test": "data/webqa/wq-test-contriever.json",
        "corpus": "data/21MWiki/psgs_w100.tsv",
        "index": "data/21MWiki_bge/faiss_index_emb"
        # "index": "data/21MWiki_e5/faiss_index_emb"
    },
    "squad":{
        "test": "data/SQuAD/validation-00000-of-00001.parquet",
        "corpus": "data/21MWiki/psgs_w100.tsv",
        "index": "data/21MWiki_bge/faiss_index_emb"
        # "index": "data/21MWiki_e5/faiss_index_emb"
    }
}


# Load test data according to the given name
# We convert the json file to a list of dicts for convenience.
def load_test_qa(data_name = "hotpotqa", num_samples = -1):
    # Json file
    if data_name in ["hotpotqa", "2wikimultihopqa"]:
        test_qa = load_json(data_name_2_path[data_name]["test"])
        #test_qa = load_json(data_name_2_path[data_name]["train"])
    
    # dict file
    elif data_name == "triviaqa":
        test_qa = load_json(data_name_2_path[data_name]["test"])
        test_qa = test_qa['Data']
        test_qa = [{"question": qa['Question'], 'answers': qa['Answer']["Aliases"]} for qa in test_qa]
    
    # question, answer, and contexts retrieved from 21MWiki (reference: https://github.com/bbuing9/ICLR24_SuRe)
    elif data_name in ["nq", "webqa"]:
        test_qa = load_json(data_name_2_path[data_name]["test"])
        test_qa = [{"question": qa['question'], 'answers': qa['answers']} for qa in test_qa]
    
    # dataframe file
    elif data_name == "squad":
        test_qa = pd.read_parquet(data_name_2_path[data_name]["test"])
        test_qa = test_qa.to_dict("records")
        test_qa = [{"question": qa['question'], 'answers': list(qa['answers']['text']), 'title': qa['title'], 'context': qa['context']} for qa in test_qa]
    
    else:
        raise ValueError(f"Unsupported data name: {data_name}. Please choose one of ['hotpotqa', '2wikimultihopqa', 'triviaqa', 'nq', 'webqa', 'squad'].")

    if num_samples > 0:
        test_qa = test_qa[:num_samples]

    return test_qa



# Load corpus data according to the given name
def load_corpus(data_name = "hotpotqa"):
    # We use their own corpus
    """
    hotpotqa and 2wikimultihopqa corpus example:
        {'id': '2', 'title': 'Move (1970 film)',
        'sentences': ['Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.',
        'The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber.']}
    """
    if data_name in ["hotpotqa", "2wikimultihopqa"]:
        with open(data_name_2_path[data_name]["corpus"], encoding="utf8") as f:
            corpus = json.load(f)
        # We link all sentences together to text
        for i in range(len(corpus)):
            corpus[i]["text"] = " ".join(corpus[i]["sentences"])
            # del corpus[i]["sentences"]

    # We use 21MWiki corpus
    # """
    #     21M Wikipedia corpus includes "id", "title", and "text" with the dataframe format.
    #     a document is split into several "text" (similar length) with the same "title".
    # """
    elif data_name in ["triviaqa", "nq", "webqa", "squad"]:
        df = pd.read_csv(data_name_2_path[data_name]["corpus"], sep='\t')
        corpus = [{"id": str(i), "title": title, "text": text} for i, title, text in zip(df["id"], df["title"], df["text"])]

    else:
        raise ValueError(f"Unsupported data name: {data_name}. Please choose one of ['hotpotqa', '2wikimultihopqa', 'triviaqa', 'nq', 'webqa', 'squad'].")


    return corpus


# get the index path according to the given name
def get_index_path(data_name = "hotpotqa"):
    return data_name_2_path[data_name]["index"]





def load_json(path, type="json"):
    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        outputs = json.loads(open(path, "r", encoding="utf-8").read())
    elif type == "jsonl":
        outputs = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                outputs.append(json.loads(line))
    else:
        outputs = []
        
    return outputs


def save_json(data, path, type="json", use_indent=True):

    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        with open(path, "w", encoding="utf-8") as fout:
            if use_indent:
                fout.write(json.dumps(data, indent=4, ensure_ascii=False))
            else:
                fout.write(json.dumps(data, ensure_ascii=False))

    elif type == "jsonl":
        with open(path, "w", encoding="utf-8") as fout:
            for item in data:
                fout.write("{}\n".format(json.dumps(item, ensure_ascii=False)))

    return path