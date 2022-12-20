#处理训练数据，将微博id按照post.csv的顺序映射为idx，并对应修改各个文件中的weiboid

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
lines = 0
with open('post.data.csv', 'r', encoding='utf-8') as p:
    with open('post.data.txt', 'w', encoding='utf-8') as fwrite:
        for line in p:
            if lines == 0:
                lines += 1
                fwrite.write('rootweiboid\trootuserid\tcontent\tpubtime\n')
                continue
            parts = line.strip().split('\t',1)
            weiboid = parts[0]
            ohter = parts[1]
            new_parts = str(weibo2idx[weiboid]) + '\t'+ohter + '\n'
            fwrite.write(new_parts)
lines = 0
with open('repost.data.csv', 'r', encoding='utf-8') as p:
    with open('repost.data.txt', 'w', encoding='utf-8') as fwrite:
        for line in p:
            if lines == 0:
                lines += 1
                fwrite.write('rootweiboid\trootuserid\tweiboid\tuserid\tcontent\tpubtime\n')
                continue
            parts = line.strip().split('\t',1)
            weiboid = parts[0]
            ohter = parts[1]
            new_parts = str(weibo2idx[weiboid]) + '\t'+ohter + '\n'
            fwrite.write(new_parts)
lines = 0
with open('infer.data.csv', 'r', encoding='utf-8') as p:
    with open('infer.data.txt', 'w', encoding='utf-8') as fwrite:
        for line in p:
            if lines == 0:
                lines += 1
                fwrite.write('rootweiboid\trootuserid\tweiboid\tuserid\tobservetime\n')
                continue
            parts = line.strip().split('\t', 1)
            weiboid = parts[0]
            ohter = parts[1]
            new_parts = str(weibo2idx[weiboid]) + '\t'+ohter + '\n'
            fwrite.write(new_parts)



























