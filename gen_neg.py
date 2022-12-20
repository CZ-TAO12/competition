import random
#生成负样本，在post的所有微博中随机选择微博，再从所有的user中随机选择一名user，表示这名用户没有转发该条微博，生成时要对这条微博是否真的未被这名用户转发进行判断
repost_dict = {}
weiboid_rootid = {}
infer_dict = {}
with open('infer.data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        weiboid = parts[0]
        rootuserid = parts[1]
        userid = parts[3]
        if weiboid in infer_dict.keys():
            infer_dict[weiboid].append(userid)
        else:
            infer_dict[weiboid] = [userid]
# weibo_list = []
# user_list = []
#weibo 1326666
#user 2181734
#repost >8000000
neg_num = 8000000
with open('repost.data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        weiboid = parts[0]
        rootuserid = parts[1]
        userid = parts[3]
        #weiboid_rootid[weiboid] = rootuserid
        #weibo_list.append(weiboid)
        if weiboid in repost_dict.keys():
            repost_dict[weiboid].append(userid)
        else:
            repost_dict[weiboid] = [userid]


with open('post.data.txt', 'r', encoding='utf-8')as p:
    for line in p:
        parts = line.strip().split('\t')
        weiboid = parts[0]
        rootuserid = parts[1]
        weiboid_rootid[weiboid] = rootuserid
neg_count = 0
with open('repost.data_neg.txt', 'w', encoding='utf-8') as fwrite:
    fwrite.write('rootweiboid\trootuserid\tuserid\n')
    while neg_count <= neg_num:
        userid = str(random.randint(0, 2181734))
        weiboid = str(random.randint(0, 1326666))
        if weiboid in infer_dict.keys() and userid in infer_dict[weiboid]:
            continue

        if weiboid not in repost_dict.keys():
            fwrite.write(weiboid+'\t'+weiboid_rootid[weiboid]+'\t'+userid+'\n')
            neg_count += 1
        elif userid not in repost_dict[weiboid]:
            fwrite.write(weiboid+'\t'+weiboid_rootid[weiboid]+'\t'+userid+'\n')
            neg_count += 1


