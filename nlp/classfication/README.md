## 文本分类算法

> 部分代码来自大牛的github: [textClassifier](https://github.com/jiangxinyang227/textClassifier),[TextClassification-Keras](https://github.com/ShawnyXiao/TextClassification-Keras), 感谢!

### 1) fasttext简单分类（速度快，可作为基准）

```python
from nlp.classfication.fasttext_classifier import FastTextClassifier
train_path = 'data/imdb'
model = FastTextClassifier('./model/fasttext/classifier', train=True, file_path=train_path)
```

> fasttext语料格式：文本和标签之间加一个空格， 比如： I like it __label__pos \
> 在imdb数据集上，能够达到88%的准确率（epoch=100,dim=256, lr=0.01）

### 2) SVM算法

* 运用线性分类模型进行分类, 在文件svm_sklearn.py里.

```python
from nlp.classfication.svm_classifier import SVMClassifier
svm_model = SVMClassifier('model/svm/model.pkl', 'data/imdb/aclImdb.txt', train=True)
```

> 在imdb数据集上取得了90%的准确率

### 3) BiLSTM

* 双向lstm获取句子的表示，采用简单的全连接层分类，简单的二分类

* 很容易过拟合

### 4) CNN

来自论文[Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181)

> 训练时没有使用训练好的词向量，一直过拟合，需要进一步微调

### 5) BiLSTM+CNN

> 待训练测试

### 6) BiLSTM+Attention

* 双向lstm获取句子的表示，然后用attention机制，最后采用简单的全连接层分类，简单的二分类

```python
model = BiLSTMAttentionClassifier('GoogleNews-vectors-negative300.bin.gz', 'model/att','model/att/config.pkl', train=True)
print(model.predict_result('this is very good movie, i want to watch it again!'))
print(model.predict_result('this is very bad movie, i hate it'))
```

> 第一次训练的时候, 需要训练好的词向量，如google的， 后边再次训练的时候，train设置为True, 只用训练好的模型做推理的时候，train设置为False

### 7) HAN

### 8) RCNN

### 9) bert



* 实验结果

    