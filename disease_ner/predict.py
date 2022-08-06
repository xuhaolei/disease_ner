#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 22:40
# @Author  : XuHaolei
# @File    : predict.py
import torch
from model.BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
from scripts.config import Config
from scripts.utils import load_vocab
import transformers
'''用于识别输入的句子（可以换成批量输入）的命名实体
    <pad>   0
    B-Disease   1
    I-Disease   2
    O       3
    <START> 4
    <EOS>   5
'''
tags = [(1, 2)]


def predict(input_seq, max_length=180):
    '''
    :param input_seq: 输入一句话
    :return:
    '''
    config = Config()
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    tagset_size = len(label_dic)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BERT_BiLSTM_CRF(tagset_size,
                            config.bert_embedding,
                            config.rnn_hidden,
                            config.rnn_layer,
                            config.dropout,
                            config.pretrain_model_name,
                            device).to(device)

    checkpoint = torch.load(config.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])

    # 构造输入
    input_list = []
    for i in range(len(input_seq)):
        input_list.append(input_seq[i])

    if len(input_list) > max_length - 2:
        input_list = input_list[0:(max_length - 2)]
    input_list = ['[CLS]'] + input_list + ['[SEP]']

    input_ids = [int(vocab[word]) if word in vocab else int(vocab['[UNK]']) for word in input_list]
    input_mask = [1] * len(input_ids)

    if len(input_ids) < max_length:
        input_ids.extend([0] * (max_length - len(input_ids)))
        input_mask.extend([0] * (max_length - len(input_mask)))
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length

    # 变为tensor并放到GPU上, 二维, 这里mask在CRF中必须为unit8类型或者bool类型
    input_ids = torch.LongTensor([input_ids]).to(device)
    input_mask = torch.ByteTensor([input_mask]).to(device)

    feats = model(input_ids, input_mask)
    # out_path是一条预测路径（数字列表）, [1:-1]表示去掉一头一尾, <START>和<EOS>标志
    out_path = model.predict(feats, input_mask)[0][1:-1]
    res = find_all_tag(out_path)

    disease = []
    # LOC = []
    # ORG = []
    for name in res:
        if name == 1:
            for i in res[name]:
                disease.append(input_seq[i[0]:(i[0]+i[1])])
        # if name == 2:
        #     for j in res[name]:
        #         LOC.append(input_seq[j[0]:(j[0]+j[1])])
        # if name == 3:
        #     for k in res[name]:
        #         ORG.append(input_seq[k[0]:(k[0]+k[1])])

    # 输出结果
    print('预测结果:', '\n', 'disease:', disease)


def find_tag(out_path, B_label_id=1, I_label_id=2):
    '''
    找到指定的label
    :param out_path: 模型预测输出的路径 shape = [1, rel_seq_len]
    :param B_label_id:
    :param I_label_id:
    :return:
    '''
    global start_pos
    sentence_tag = []
    for num in range(len(out_path)):
        if out_path[num] == B_label_id:
            start_pos = num
        if out_path[num] == I_label_id and out_path[num-1] == B_label_id:
            length = 2
            for num2 in range(num, len(out_path)):
                if out_path[num2] == I_label_id and out_path[num2-1] == I_label_id:
                    length += 1
                    if num2 == len(out_path)-1:  # 如果已经到达了句子末尾
                        sentence_tag.append((start_pos, length))
                        return sentence_tag
                if out_path[num2] == 3:
                    sentence_tag.append((start_pos, length))
                    break
    return sentence_tag


def find_all_tag(out_path):
    num = 1  # 1: PER、 2: LOC、3: ORG
    result = {}
    for tag in tags:
        res = find_tag(out_path, B_label_id=tag[0], I_label_id=tag[1])
        result[num] = res
        num += 1
    return result


if __name__ == "__main__":
    tokenizer = transformers.BertTokenizer.from_pretrained("../dataset/bert/alvaroalon2biobert_diseases_ner")
    while True:
        input_seq = input("Example: Parkinson's syndrome is a common disease in the elderly.\nPlease input a sentence: ")
        text = input_seq.split(' ')
        # Germline mutations of the APC gene in Korean familial adenomatous polyposis patients.
        print(text)
        new_text = []
        for cnt, word in enumerate(text):
            word_piece = tokenizer.tokenize(word)
            new_text += word_piece
        print(new_text)
        predict(new_text)


