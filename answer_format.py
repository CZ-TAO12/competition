#最终结果格式化，将转化为idx的weiboid转化回原始的id，并写入文件

weibo2idx = {}
lines = 0
idx = 0
with open('post.data.csv', 'r', encoding='utf-8') as p:
    for line in p:
        if lines == 0:
            lines += 1
            continue
        parts = line.strip().split('\t')
        weiboid = parts[0]
        if weiboid not in weibo2idx:
            weibo2idx[weiboid] = idx
            idx += 1

idx2weibo = {}
lines = 0
for key,value in weibo2idx.items():
    idx2weibo[value] = key
print(idx2weibo)
with open('infer.data_content_ver3.txt', 'r') as f:
    with open('submission.csv', 'w') as fwrite:
        fwrite.write('rootweiboid\trootuserid\tweiboid\tuserid\tlabel\n')
        for line in f:
            if lines == 0:
                lines += 1
                continue
            parts = line.strip().split('\t', 1)
            rootweiboid = idx2weibo[int(parts[0])]
            fwrite.write(rootweiboid+'\t'+'\t'.join(parts[1:])+'\n')
