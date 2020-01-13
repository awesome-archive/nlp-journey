# coding=utf-8
# created by msgi on 2019/10/26 10:38 上午

import codecs
import os

import data_utils


def load_sentences(path):
    """
    加载数据集，每一行至少包含一个汉字和一个标记，句子和句子之间是以空格作为分割
    :param path: 数据路径
    :return: 句子集合 [[['word','freq'],['word','freq']], [['word','freq'],['word','freq']]]
    """
    # 存放数据集
    sentences = []
    # 临时存放每一个句子
    sentence = []
    for line in codecs.open(path, 'r', encoding='utf8'):
        # 去除两边的空格
        line = line.strip()
        # 首先判断是不是空，如果是则表示句子和句子之间的分割点
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
                # 清空sentence表示一句话完结
                sentence = []
        else:
            if line[0] == " ":
                continue
            else:
                word = line.split()
                assert len(word) >= 2
                sentence.append(word)
    # 循环走完要判断一下，防止最后一个句子没有进入到句子集合中
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def word_mapping(sentences):
    """
    构建字典
    :param sentences: 语料数据
    :return: word统计频数字典、word_id映射、id_word映射
    """
    word_list = [[x[0] for x in s] for s in sentences]
    word_count = data_utils.create_item_count(word_list)
    # 一个小trick, 将填充词和未登录词的词频设置的特别大，排序字典可以排到前边
    word_count["<PAD>"] = 10000001
    word_count["<UNK>"] = 10000000
    word_to_id, id_to_word = data_utils.create_mapping(word_count)
    return word_count, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    构建标签字典
    :param sentences: 语料数据
    :return: tag统计频数字典、tag_id映射、id_tag映射
    """
    tag_list = [[x[1] for x in s] for s in sentences]
    tag_count = data_utils.create_item_count(tag_list)
    tag_to_id, id_to_tag = data_utils.create_mapping(tag_count)
    return tag_count, tag_to_id, id_to_tag


def prepare_dataset(sentences, word_to_id, tag_to_id, train=True):
    """
    数据预处理，将数据变成id数字形式
    :param sentences: 原来数据
    :param word_to_id: word_id映射表
    :param tag_to_id: tag_id映射表
    :param train: 是否训练模式
    :return: [word_list, word_id_list, seg_id_list, tag_id_list]
    """
    none_index = tag_to_id['O']
    data = []
    for s in sentences:
        # 获取所有的字，替换为ID，未登录词替换为<UNK>的ID
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id[w if w in word_to_id else '<UNK>'] for w in word_list]
        # 把所有的字合并成句子
        seg_id_list = data_utils.get_seg_id_list("".join(word_list))
        if train:
            tag_id_list = [tag_to_id[w[-1]] for w in s]
        else:
            # 非训练模式下，tag全部初始化为 'O'
            tag_id_list = [none_index for _ in s]
        data.append([word_list, word_id_list, seg_id_list, tag_id_list])
    return data


def augment_with_pre_trained(word_dict, emb_path, more_words):
    """
    增加字典里有预训练的词向量的词的数目
    :param word_dict:
    :param emb_path:
    :param more_words:
    :return:
    """
    print('从词向量文件%s中加载...' % emb_path)
    assert os.path.isfile(emb_path)
    pre_trained_words = [line.rstrip().split()[0].strip() for line in codecs.open(emb_path, 'r', 'utf-8')]
    if more_words is None:
        for word in pre_trained_words:
            if word not in word_dict:
                word_dict[word] = 0
    else:
        for word in more_words:
            if any(x in pre_trained_words for x in [word, word.lower()]) and word not in word_dict:
                word_dict[word] = 0
    word_to_id, id_to_word = data_utils.create_mapping(word_dict)
    return word_dict, word_to_id, id_to_word


def update_tag_scheme(sentences, tag_scheme):
    """
    更新为指定编码
    :param sentences: 数据
    :param tag_scheme: 新的tag编码模式
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        if not data_utils.check_bio(tags):
            s_str = '\n'.join(" ".join(w) for w in s)
            raise Exception("输入的句子应为BIO编码，请检查输入句子%i:\n%s" % (i, s_str))
        if tag_scheme == 'BIOES':
            new_tags = data_utils.bio_to_bioes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('非法目标编码')


if __name__ == '__main__':
    sentences = load_sentences('data/ner.dev')
    update_tag_scheme(sentences, 'BIOES')
    _, word_to_id, id_to_word = word_mapping(sentences)
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
    data = prepare_dataset(sentences, word_to_id, tag_to_id)
    print(data)
