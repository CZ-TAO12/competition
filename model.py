import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class PostPredict(torch.nn.Module):
    def __init__(self, batch_size):
        super(PostPredict, self).__init__()
        #self.weibo_emb = nn.Embedding(1326667, 5).cuda()
        self.user_emb = nn.Embedding(2181735, 15).cuda()
        self.batch_size = batch_size
        #self.st = SentenceTransformer('cyclone/simcse-chinese-roberta-wwm-ext').cuda()
        #self.gru = nn.GRU(num_layers=3, input_size=20, hidden_size=20, batch_first=True).cuda()
        self.l1 = nn.Linear(50, 20).cuda()
        self.l2 = nn.Linear(20, 5).cuda()
        self.l3 = nn.Linear(5, 2).cuda()
        self.ls1 = nn.Linear(768, 200).cuda()
        self.ls2 = nn.Linear(200, 20).cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, weiboid, userid, content):
        # print(weiboid.size())
        # print(userid.size())
        #content = self.st.encode(content, convert_to_tensor=True)
        content = F.relu(self.ls1(content))
        content = F.relu(self.ls2(content))
        #w_emb = self.weibo_emb(weiboid)
        u_emb = self.user_emb(userid)
        #w_emb = w_emb.reshape(self.batch_size, -1)
        u_emb = u_emb.reshape(self.batch_size, -1)
        # print(w_emb.size())
        # print(u_emb.size())
        #emb = torch.cat([w_emb, u_emb], 1)
        emb = torch.cat([content, u_emb], 1)
        pre = F.relu(self.l1(emb))
        pre = F.relu(self.l2(pre))
        pre = self.l3(pre)
        return pre


#得到当前batch的数据
def get_slice(data, i):
    weiboid = np.array(data)[:, 0:1][i]
    users = np.array(data)[:, 1:3][i]
    lable = np.array(data)[:, -1][i]
    #content = np.array(data)[:, 3][i]
    return weiboid, users, lable


#获取微博内容数据
def get_weibo_content(weiboids, content_dict):
    contents = []
    weiboids = np.squeeze(weiboids)
    for weiboid in weiboids:
        contents.append(content_dict[str(weiboid)])
    return contents


#将内容embedding，userid输入模型
def forward(model, train_data, i, content_dict):
    #contents = get_weibo_content(train_data, content_dict)
    weiboid, users, lable = get_slice(train_data, i)
    #print(weiboid)
    content_emb = np.array((get_weibo_content(weiboid, content_dict)))
    content_emb = torch.Tensor(content_emb).cuda()
    #weiboid.astype(int)
    #users.astype(int)
    #weiboid = torch.LongTensor(weiboid).cuda()
    users = torch.LongTensor(users).cuda()
    lable = torch.LongTensor(lable).cuda()
    # print(lable.size())
    # print(lable)
    pre = model(weiboid, users, content_emb)
    return lable, pre


#预测结果
def predict(model, infer_data, i, content_dict):
    weiboid = np.array(infer_data)[:, 0:1][i]
    users = np.array(infer_data)[:, 1:4:2][i]
    content_emb = np.array((get_weibo_content(weiboid, content_dict)))
    content_emb = torch.Tensor(content_emb).cuda()
    #weiboid.astype(int)
    #users.astype(int)
    #weiboid = torch.LongTensor(weiboid).cuda()
    users = torch.LongTensor(users).cuda()
    #lable = torch.LongTensor(lable).cuda()
    # print(lable.size())
    # print(lable)
    pre = model(weiboid, users, content_emb)
    return pre
