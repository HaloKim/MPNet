import transformers

path = "/workspace/MPNet/dump"
model = transformers.MPNetForMaskedLM.from_pretrained(path)
tokenizer = transformers.MPNetTokenizerFast.from_pretrained("my_tokenizer")

model.push_to_hub("dev7halo/MPNet-still-training",)
tokenizer.push_to_hub("dev7halo/MPNet-still-training",)

