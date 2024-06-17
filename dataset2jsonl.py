from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("dev7halo/kor_corpus", use_auth_token=True, cache_dir="/workspace/data")

# Iterate over the dataset and convert to JSONL
with open("/workspace/data/output.jsonl", "w") as f:
    for example in dataset["train"]:
        json_string = json.dumps(example)
        f.write(json_string + "\n")

