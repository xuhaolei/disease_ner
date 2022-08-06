#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/5 22:40
# @Author  : XuHaolei
# @File    : config.py
class Config(object):
    '''配置类'''

    def __init__(self):
        self.label_file = '../dataset/tag.txt'
        self.train_file = '../dataset/train.txt'
        self.dev_file = '../dataset/dev.txt'
        self.test_file = '../dataset/test.txt'
        self.vocab = '../dataset/bert/biobert/vocab.txt'
        self.max_length = 180
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 32
        # 神经元个数
        self.rnn_hidden = 180
        # 词向量维数
        self.bert_embedding = 768
        self.dropout = 0.5
        self.rnn_layer = 1
        self.lr = 0.0001
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = '../result/checkpoints/disease_result/disease_best.pth.tar'
        self.epochs = 32
        self.max_grad_norm = 10
        self.target_dir = '../result/checkpoints/disease_result'
        self.patience = 10
        # bert预训练模型 https://huggingface.co/
        self.pretrain_model_name = '../dataset/bert/alvaroalon2biobert_diseases_ner'
        # 对抗训练,给word_embedding增加扰动 参数为1时选用FGM,2为PGD,0为正常训练
        # PGD有bug,保存不了图，不知道为什么
        self.adversarial = 0
        # 洪泛法
        # loss = (loss - b).abs() + b
        self.b = 0.2

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)