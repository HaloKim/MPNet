import transformers

path = "/workspace/MPNet/dump"
model = transformers.MPNetForMaskedLM.from_pretrained(path)
tokenizer = transformers.MPNetTokenizerFast.from_pretrained("klue/roberta-base")

model.push_to_hub("dev7halo/MPNet",)
tokenizer.push_to_hub("dev7halo/MPNet",)

