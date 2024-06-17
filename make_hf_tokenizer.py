from transformers import BertTokenizer

vocab_path = "vocab.txt"
tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
tokenizer.save_pretrained("my_tokenizer")

