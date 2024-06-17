import random


def split_text_file(input_file, output_file1, output_file2, output_file3):
    with open(input_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)
    total_lines = len(lines)

    # 전체 라인 수의 80%까지를 output_file1에 저장
    with open(output_file1, 'w', encoding="utf-8") as f1:
        f1.writelines(lines[:int(total_lines*0.8)])

    # 전체 라인 수의 80%부터 90%까지를 output_file2에 저장
    with open(output_file2, 'w', encoding="utf-8") as f2:
        f2.writelines(lines[int(total_lines*0.8):int(total_lines*0.9)])

    # 전체 라인 수의 90% 이후 부분을 output_file3에 저장
    with open(output_file3, 'w', encoding="utf-8") as f3:
        f3.writelines(lines[int(total_lines*0.9):])

# Example usage:
path = "/workspace/MPNet/"
input_file = 'corpus.txt'
output_file1 = path + 'train.txt'
output_file2 = path + 'test.txt'
output_file3 = path + 'valid.txt'
split_text_file(input_file, output_file1, output_file2, output_file3)

