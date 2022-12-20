'''
采用预训练的SentenceTransformer对微博中文内容进行预处理，生成对应的token编码，然后将token编码存下来用于下游任务
SentenceTransformer的预训练参数来自于https://huggingface.co/cyclone/simcse-chinese-roberta-wwm-ext
'''
import pickle
from sentence_transformers import SentenceTransformer


weiboid2contentemb = {}
content_list = []
lines = -2
#加载预训练好的模型，将微博内容输入模型，得到内容的embedding后存到二进制文件中
model = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext').cuda()
with open('post.data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        lines += 1
        if lines == -1:
            continue
        parts = line.strip().split('\t')
        weiboid = parts[0]
        content = parts[2]
        emb = model.encode(content)
        weiboid2contentemb[weiboid] = emb

with open('content_emb.pickle', 'wb') as fwrite:
    pickle.dump(weiboid2contentemb, fwrite)

