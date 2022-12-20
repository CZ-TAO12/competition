import argparse
import pickle

from model import *
import random
from tqdm import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=int, default=1e-3, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=512000, help='input batch size')
parser.add_argument('--emb_size', type=int, default=100, help='embedding size')
parser.add_argument('--epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--save_path', default="./checkpoint/UserPredictionV3.pt")
parser.add_argument('--mode', default='train')

opt = parser.parse_args()
#weibo 1326666 user 2181734
#repost 746827
#repost times 8147543 neg num 8000000


#获取每个batch的index
def generate_batch(length, batch_size):
    n_batch = int(length / batch_size)
    if length % batch_size != 0:
        n_batch += 1
    slices = np.split(np.arange(n_batch * batch_size), n_batch)
    slices[-1] = np.arange(length - batch_size, length)
    return slices


#获取训练和测试数据
def get_data():
    repost_list = []
    neg_list = []
    lines = 0
    with open('repost.data.txt', 'r', encoding='utf-8') as p:
        for line in p:
            if lines == 0:
                lines += 1
                continue
            parts = line.strip().split('\t')
            repost_list.append([int(parts[0]), int(parts[1]), int(parts[3]), 1])
    #print(len(repost_list))
    lines = 0
    with open('repost.data_neg.txt', 'r', encoding='utf-8') as p:
        for line in p:
            if lines == 0:
                lines += 1
                continue
            parts = line.strip().split('\t')
            neg_list.append([int(parts[0]), int(parts[1]), int(parts[2]), 0])
    #print(len(neg_list))
    # repost_list = repost_list[1:]
    # neg_list = neg_list[1:]
    return repost_list+neg_list


#获取生成的内容embedding
def get_content():
    with open('content_emb.pickle', 'rb') as f:
        return pickle.load(f)

#输出预测结果并写入文件
def predict_repost(content_dict):
    model = PostPredict(opt.batch_size)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()
    infer_data = []
    infer_data_str = []
    count = 0
    with open('infer.data.txt') as f:
        for line in f:
            if count == 0:
                count += 1
                continue
            parts = line.strip().split('\t')
            # weiboid = parts[0]
            # rootuserid = parts[1]
            # userid = parts[3]
            infer_data.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
            infer_data_str.append([parts[0], parts[1], parts[2], parts[3]])

    slices = generate_batch(len(infer_data), opt.batch_size)
    pre_list = []
    #y_pred = []
    with open('infer.data_content_ver3.txt', 'w') as fwrite:
        fwrite.write('rootweiboid\trootuserid\tweiboid\tuserid\tlabel\n')
        for i in tqdm(slices):
            pre = predict(model, infer_data, i, content_dict)
            pre = pre.cpu().detach().numpy()
            # lable = lable.cpu().detach().numpy()
            # print(pre)
            #y_true.extend(lable)
            for pred in pre:
                if pred[0] > pred[1]:
                    pre_list.append(0)
                else:
                    pre_list.append(1)
            # pre_list = np.array(pre_list)
            # pre_list = pre_list[np.newaxis, :]
            answer = np.array(infer_data_str)[i].tolist()
            #print(answer)
            # answer = np.concatenate([answer, pre])
            #answer = np.append(answer,pre_list, axis=1)
            for index in range(len(pre_list)):
                answer[index].append(str(pre_list[index]))
                # answer = np.append()
            #print(answer)
            for line in answer:
                fwrite.write('\t'.join(line)+'\n')


if __name__ == '__main__':
    content_dict = get_content()
    if opt.mode == 'export':
        predict_repost(content_dict)
    else:
        #temp = np.array()
        #print(content_dict)
        repost_list = get_data()
        random.shuffle(repost_list)
        #print(len(repost_list))
        train_data = repost_list[1600000:]
        test_data = repost_list[:1600000]

        #repost_his = get_repost_his()

        #print(repost_list)
        model = PostPredict(opt.batch_size)
        best_f1 = 0.0
        for epoch in range(opt.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            torch.autograd.set_detect_anomaly(True)
            model.train()
            total_loss = 0.0
            slices = generate_batch(len(train_data), opt.batch_size)
            # print(slices)
            # print(slices[-1])
            model_loss = torch.nn.CrossEntropyLoss()
            for i in tqdm(slices):
                model.zero_grad()
                lable, pre = forward(model, train_data, i, content_dict)
                # pre = torch.squeeze(pre)
                # print(lable.size())
                # print(pre.size())
                # print(lable)
                # print(pre)
                loss = model_loss(pre, lable)
                loss.backward()
                model.optimizer.step()
                total_loss += loss
            print('\tLoss:\t%.3f' % total_loss)

            print('start predicting: ')
            model.eval()
            slices = generate_batch(len(test_data), opt.batch_size)
            pre_list = []
            y_true = []
            for i in tqdm(slices):
                lable, pre = forward(model, test_data, i, content_dict)
                pre = pre.cpu().detach().numpy()
                lable = lable.cpu().detach().numpy()
                #print(pre)
                y_true.extend(lable)
                for pred in pre:
                    if pred[0]>pred[1]:
                        pre_list.append(0)
                    else:
                        pre_list.append(1)
            pre_list = np.array(pre_list)
            y_true = np.array(y_true)
            f1 = f1_score(y_true=y_true, y_pred=pre_list, average='binary')
            acc = accuracy_score(y_true=y_true, y_pred=pre_list)
            recall = recall_score(y_true=y_true, y_pred=pre_list, average='macro')
            precision = precision_score(y_true=y_true, y_pred=pre_list)

            print('f1:'+str(f1))
            print('acc:'+str(acc))
            print('recall:'+str(recall))
            print('precision:'+str(precision))
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), opt.save_path)
                print('Save best model!!!')




































