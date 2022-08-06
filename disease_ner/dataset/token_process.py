#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 22:40
# @Author  : XuHaolei
# @File    : token_process.py
import transformers

tokenizer = transformers.BertTokenizer.from_pretrained("../dataset/bert/alvaroalon2biobert_diseases_ner")

file = "dev.txt"
text = []
tag = []
with open(file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if len(line.split(" ")) == 2:
            line = line.strip("\n")
            text.append(line.split(" ")[0])
            tag.append(line.split(" ")[1])
        else:
            new_text = []
            new_tag = []
            for cnt, word in enumerate(text):
                word_piece = tokenizer.tokenize(word)
                new_text += word_piece
                new_tag += [tag[cnt]]
                if 'B-' in tag[cnt] or 'I-' in tag[cnt]:
                    new_tag += ["I-Disease"] * (len(word_piece) - 1)
                else:
                    new_tag += ["O"] * (len(word_piece) - 1)
            with open("new/dev.txt", 'a', encoding='utf-8') as f:
                for i in range(len(new_text)):
                    f.write(new_text[i] + ' ' + new_tag[i] + '\n')
                f.write('\n')
            text = []
            tag = []


# file = "new/test.txt"
# with open(file, 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     cnt = 0
#     length = []
#     for line in lines:
#         if len(line.split(" ")) == 2:
#             cnt += 1
#         else:
#             length.append(cnt)
#             cnt = 0
#     print(max(length))