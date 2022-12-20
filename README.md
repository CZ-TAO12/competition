### Requirements
```
python==3.9.7
pytorch==1.13+cu117
numpy==1.23.5
scikit-learn==1.1.3
sentence-transformers==2.2.2
```

### 运行顺序
```shell
# 处理训练数据，将微博id按照post.csv的顺序映射为idx，并应用到各个文件
python gen_train.py

# 生成负样本，共8000000条
python gen_neg.py 

# 使用sentence-transformer生成微博内容embedding
python content_emb.py

# 运行模型
python main.py 

# 得到输出结果
python main.py --mode export --save_path ./checkpoint/UserPredictionV3.pt --batch_size 875111

# 格式化结果
python answer_format.py
```

### Model说明
```
1. 数据分析：微博转发行为主要跟微博内容、微博发布人、微博转发人相关，主要利用微博内容、微博发布人、微博转发人的相关特征

2. 模型框架思路：
(1)我们借鉴了NLP领域中的one-hot编码，对每一个user进行编码，通过数据标签来更新user编码，是user特征更适合于当前数据集

(2)采用了预训练的sentence-transformer模型对微博内容进行编码，预训练参数为‘cyclone/simcse-chinese-roberta-wwm-ext’

(3)为了更好地训练模型，我们构建的随机生成的负样本。我们使用repost.data调整模型参数，由于其中数据均构成转发关系，为正样本，故还需要随机生成一部分负样本：在数据范围内，随机生成weiboid、userid对。将正负样本混合，使模型可以辨别正负样本。

(4)微博内容的embedding，发布user的embedding、转发user的embedding拼接输入MLPs进行训练。

(5)为了选取最佳模型，我们按照9：1比例将数据分为训练集，测试集，利用测试集的结果选取模型，然后进行测试

```


### 模型
```
SentenceTransformers 是一个可以用于句子、文本和图像嵌入的Python库。 
可以为 100 多种语言计算文本的嵌入并且可以轻松地将它们用于语义文本相似性、语义搜索和同义词挖掘等常见任务。
在论文Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks提出。
该框架基于 PyTorch 和 Transformers，并提供了大量针对各种任务的预训练模型，还可以很容易根据自己的模型进行微调。
借助SentenceTransformer，我们使用Cyclone SIMCSE RoBERTa WWM Ext Chinese模型直接生成对微博内容数据的嵌入。
该模型基于Whole Word Masking(WWM)的中文预训练BERT模型Chinese RoBERTa WWM Ext，提供了基于简单对比学习（SimCSE:Simple Contrastive Learning of Sentence Embeddings）的简体中文句子嵌入。

```
### 训练好的模型参数
Baidu Cloud：https://pan.baidu.com/s/17nlQ-WMy4q7L19hzLM63jw 
password：zbci
