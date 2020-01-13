# coding=utf-8
# created by msg on 2019/12/4 2:48 下午

import tensorflow_datasets as tfds

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

print(train_examples, val_examples)

# 从数据集中创建自定义子词分词器:encode将字符串转为索引序列，decode将索引序列变为字符串
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'
print(tokenizer_en.encode(sample_string))       # [7915, 1248, 7946, 7194, 13, 2799, 7877]




