from datasets import load_dataset, load_from_disk

dataset = load_dataset("dev7halo/kor_corpus", use_auth_token=True, cache_dir="/workspace/data")

