# coding=utf-8
# created by msgi on 2019/10/26 10:39 上午

import jieba
import math
import random
import codecs
import numpy as np


def check_bio(tags):
    """
    检测输入的tags是否是bio编码
    如果不是bio编码，那么错误的类型如下：
        1.编码不在bio中
        2.第一个编码是I
        3.当前编码不是B，前一个编码是O
    :param tags: 编码
    :return: True or False
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue

        tag_list = tag.split('-')
        if len(tag_list) != 2 or tag_list[0] not in {'B', 'I'}:
            # 表示非法编码
            return False

        if tag_list[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == "O":
            # 如果第一个位置不是B，或者当前编码不是B并且前一个编码是O，则转化为B
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            # 如果当前编码的后面类型编码与tags中的前一个编码中后面类型编码相同则跳过
            continue
        else:
            # 如果类型不一致，则重新从B开始编码
            tags[i] = "B" + tag[1:]
    return True


def bio_to_bioes(tags):
    """
    把bio编码转化为bioes编码，返回新的tags
    :param tags: 之前的编码
    :return: 新的编码
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            # 如果tag是以B开头，首先，如果当前tag不是最后一个，并且紧跟着的后一个是I
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                # tag是一个元素，或着紧跟着后一个不是I，那么表示单字，需要把B换成S表示单字
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            # 如果tag是以I开头，那么部门需要进行下面的判断：
            # 首先，如果当前的tag不是最后一个，并且紧跟着的一个是I
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                # 如果是最后一个，或者后一个不是I开头的，那么就表示一个词的结尾，就把I换成E
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('非法编码')
    return new_tags


def bioes_to_bio(tags):
    """
    把bioes编码转回bio编码
    :param tags: bioes编码的tag
    :return: 新的bio编码的tag
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('非法编码')
    return new_tags


def create_item_count(item_list):
    """
    对于item_list中每一个items，统计items中的item在item_list中的次数
    item:出现的次数
    :param item_list:
    :return:
    """
    assert type(item_list) is list
    word_count = {}
    for items in item_list:
        for item in items:
            if item not in word_count:
                word_count[item] = 1
            else:
                word_count[item] += 1
    return word_count


def create_mapping(item_count):
    """
    创建item_to_id , id_to_item, item的排序按词典中出现的次数
    :param item_count:
    :return:
    """
    sorted_items = sorted(item_count.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {k: v for v, k in id_to_item.items()}
    return item_to_id, id_to_item


def get_seg_id_list(words):
    """
    利用jieba分词，采用类似bio的编码，0表示单字成词，1表示一个词的开始，2表示一个词的中间，3表示一个词的结尾
    :param words: 合并起来的句子
    :return: 分词特征
    """
    seg_id_list = []
    word_list = list(jieba.cut(words))
    for word in word_list:
        if len(word) == 1:
            seg_id_list.append(0)
        else:
            temp = [2] * len(word)
            temp[0] = 1
            temp[-1] = 3
            seg_id_list.extend(temp)
    return seg_id_list


def load_word2vec(emb_file, id_to_word, word_dim, old_weights):
    """
    加载词向量
    :param emb_file: 词向量文件
    :param id_to_word: 索引到词的映射
    :param word_dim: 词向量维度
    :param old_weights:
    :return:
    """
    new_weights = old_weights
    pre_trained = {}
    emb_invalid = []
    for i, line in enumerate(codecs.open(emb_file, 'r', encoding='utf8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid.append(line)
    if len(emb_invalid) > 0:
        print('warning: %i invalid lines! lines are: \n(%s)' % (len(emb_invalid), emb_invalid))
    num_words = len(id_to_word)
    for i in range(num_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
    print('加载了 %i 个词向量, 可用 %i 个词向量' % (len(pre_trained), len(new_weights)))
    return new_weights


def full_to_half(s):
    """
    Convert full-width character to half-width one
    """
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def replace_html(s):
    s = s.replace('&quot;', '"')
    s = s.replace('&amp;', '&')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&nbsp;', ' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;", "")
    s = s.replace("\xa0", " ")
    return s


def input_from_line(line, char_to_id):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    line = full_to_half(line)
    line = replace_html(line)
    line = line.decode("utf-8")
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([
        [char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
         for char in line]
    ])
    inputs.append([get_seg_id_list(line)])
    inputs.append([[]])
    print(inputs)
    return inputs


class BatchManager(object):

    def __init__(self, data, batch_size):
        """
        批次处理类
        :param data: 数据
        :param batch_size: 一个批次大小
        """
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = []
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        words_list = []
        word_id_list = []
        seg_id_list = []
        tag_id_list = []
        # 怀疑代码有问题，这里的最大长度每次迭代都是变化的？
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            words, word_ids, seg_ids, tag_ids = line
            padding = [0] * (max_length - len(words))
            words_list.append(words + padding)
            word_id_list.append(word_ids + padding)
            seg_id_list.append(seg_ids + padding)
            tag_id_list.append(tag_ids + padding)
        return [words_list, word_id_list, seg_id_list, tag_id_list]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
