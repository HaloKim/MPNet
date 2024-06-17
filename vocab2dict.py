with open('./my_tokenizer/vocab.txt', 'r', encoding='utf-8') as file:
    content = file.readlines()

with open('dict.txt', 'a', encoding='utf-8') as file:
    for idx, txt in enumerate(content):
        file.write(f"{txt.split()[0]} {idx}\n")

