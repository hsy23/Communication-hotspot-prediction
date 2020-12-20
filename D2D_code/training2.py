import argparse
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

from D2D_code.Lstm_model import LSTMs
from D2D_code.Gcn_model import GCN
from D2D_code.units import get_all_input, last_value, collate_fn, MyData
from torch.utils.data import DataLoader


"""Multi-channel temporal end-to-end framework"""

"""Input datasets
    dgls: dataframe with userid as index, contains labels, and activity sequence if needed
    graphs: dictionary with format {user_id: list of networkx graphs}}

    ** Modify acitivity and macro flags to run different versions of model to include features
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gcn_in', type=int, default=32)
    parser.add_argument('--gcn_hid', type=int, default=48)
    parser.add_argument('--gcn_out', type=int, default=48)
    parser.add_argument('--lstm_hid', type=int, default=64)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_drop', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--period', type=int, default=9)  # 8 periods for train,Time Window is about 1 week

    parser.add_argument('--label_path', type=str, default='../spatio-temporal net/gowalla/d75h10/abel_538.npy')
    parser.add_argument('--graphs_path', type=str,
                        default='../spatio-temporal net/gowalla/d75h10/sub_net_538.npy')
    parser.add_argument('--input_path', type=str, default='../spatio-temporal net/gowalla/d75h10/feature_538.npy')

    args = parser.parse_args()
    k = args.period

    # load data
    # 若遇到数据读取错误，须在units的错误反馈下修改pickle类文件库的源码，某些库版本会因为pickle属性为false禁止读取，只要注释掉就好了
    dgls = get_all_input(args.graphs_path, True)
    inputs = get_all_input(args.input_path, False)
    labels = get_all_input(args.label_path, False)
    print('size of labels:{}, size of inputs feature mat:{}'.format(len(labels), len(inputs)))
    print('num of not hot:{}, hotspots:{}'.format(sum(sum(labels == 0)), sum(sum(labels == 1))))
    edges = 0
    nodes = 0
    gs = 0
    for G in dgls:
        gs += 1
        for G_x in G:
            edges += G_x.number_of_edges()
            nodes += G_x.number_of_nodes()
    print('total edges:', edges, "total nodes:", nodes, "average nodes:", nodes / gs)

    # train, test split
    n = len(dgls)
    print('num of dgls = {}'.format(n))
    split = int(n * .8)  # 取80%的训练集
    index = np.arange(n)
    np.random.seed(32)
    np.random.shuffle(index)
    train_index, test_index = index[:split], index[split:]
    # prep labels
    train_labels = labels[train_index]
    test_labels = Variable(torch.FloatTensor(labels[test_index]))

    test_labels = test_labels.cuda()

    # prep input data
    trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]
    testGs = [dgl.batch([dgl.DGLGraph(u[i]) for u in testGs]) for i in range(k)]

    train_inputs, test_inputs = [inputs[i] for i in train_index], [inputs[i] for i in test_index]
    test_inputs = [torch.FloatTensor(np.concatenate([inp[i] for inp in test_inputs])) for i in range(k)]

    data = MyData(trainGs, train_inputs, train_labels)
    data_loader = DataLoader(data, batch_size=1000, shuffle=False, collate_fn=collate_fn)
    warnings.filterwarnings("ignore")
    '''*************************************************************************************************************'''
    # define models
    model = LSTMs(args.gcn_out, args.lstm_hid, args.num_classes, args.lstm_layers, args.lstm_drop)
    net = GCN(args.gcn_in, args.gcn_hid, args.gcn_out)
    mode = model.cuda()
    net = net.cuda()

    # model1 = LSTMs(args.a_in, args.lstm_hid, args.num_classes, args.lstm_layers, args.lstm_drop)
    linear_in_dim = args.num_classes
    linear = nn.Linear(linear_in_dim, args.num_classes)
    linear = linear.cuda()

    parameters = list(net.parameters()) + list(model.parameters()) + list(linear.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    dropout = nn.Dropout(args.drop)

    epoch_loss_all = []
    train_score = []
    test_score = []
    pre_score = []
    re_score = []
    for epoch in tqdm(range(args.epoch)):
        model.train()
        net.train()
        for step, (g, f, batch_labels) in enumerate(data_loader):
            batch_labels = batch_labels.cuda()
            # Run through GCN
            sequence = torch.stack([net(g[i], f[i]) for i in range(k)], 1)
            # Temporal graph embeddings through lstm
            last, out = model(sequence)

            cat = out
            cat = dropout(cat)  # dropout层

            mapped = linear(cat)
            sigmoid = nn.Sigmoid()
            sigp = sigmoid(mapped).cuda()

            Bce_loss = nn.BCELoss()
            BCE_loss = Bce_loss(sigp.reshape(1, -1).squeeze(0), batch_labels.view(1, -1).squeeze(0))

            temp_label = batch_labels.cpu().view(1, -1).squeeze(0)
            temp_pre = np.int64(sigp.cpu().detach().numpy() > 0.5).reshape(1, -1).squeeze(0)
            f1 = f1_score(temp_label, temp_pre)

            #  back propagation
            optimizer.zero_grad()
            BCE_loss.backward()
            optimizer.step()

        # eval
        model.eval()
        net.eval()

        test_sequence = torch.stack([net(testGs[i], test_inputs[i]) for i in range(k)], 1)  # 生成序列
        last, out = model(test_sequence)

        cat = out
        mapped = linear(cat)
        test_sigp = sigmoid(mapped).cuda()

        temp_label = test_labels.cpu().view(1, -1).squeeze(0)
        temp_pre = np.int64(test_sigp.cpu().detach().numpy() > 0.5).reshape(1, -1).squeeze(0)
        test_f1 = f1_score(temp_label, temp_pre)
        test_pre = precision_score(temp_label, temp_pre)
        test_recall = recall_score(temp_label, temp_pre)

        if epoch % 50 == 0:
            print('Epoch %d | Train Loss: %.4f | Train F1: %.4f | Test Rate: %.4f '
                  % (epoch, BCE_loss.item(), f1, test_f1))
        epoch_loss_all.append(BCE_loss.cpu())
        train_score.append(f1)
        test_score.append(test_f1)
        pre_score.append(test_pre)
        re_score.append(test_recall)

    # # save the params of net
    # torch.save(net.state_dict(), 'gcn_params.pkl')
    # torch.save(model.state_dict(), 'lstm_params.pkl')
    # torch.save(linear.state_dict(), 'linear_params.pkl')

    print('lr:{}, max(pre_score):{}, max(re_score):{}, max(f1_score):{},'
          .format(args.lr, max(pre_score), max(re_score), max(test_score)))

    print('last_pre:{}\n, last_re:{}\n, last_f1:{}\n'.format(last_value(pre_score[-10:]),
                                                             last_value(re_score[-10:]), last_value(test_score[-10:])))

    save_cate = 538
    plt.title("Loss and F1 over epoch")
    plt.subplot(121)
    plt.plot(epoch_loss_all, label='BCE_Loss', color="#0000FF")
    plt.legend()
    plt.subplot(122)
    plt.plot(train_score, label='f1_socre', color="#0000FF")
    plt.legend()
    plt.savefig('./cate' + str(save_cate) + 'train_gow.png', bbox_inches='tight')
    plt.clf()  # 防止图片叠加

    plt.title('test f1_socre')
    plt.plot(test_score, label='f1_score', color="#DB7093")
    plt.legend()
    plt.savefig('./cate' + str(save_cate) + 'test_gow.png', bbox_inches='tight')
    plt.clf()  # 防止图片叠加

    plt.title('test pre and recall')
    plt.subplot(121)
    plt.plot(pre_score, label='pre_score', color="#0000FF")
    plt.legend()
    plt.subplot(122)
    plt.plot(re_score, label='re_score', color="#0000FF")
    plt.legend()
    plt.savefig('./cate' + str(save_cate) + 'test_gow2.png', bbox_inches='tight')
    plt.clf()  # 防止图片叠加'''
