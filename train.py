import dgl
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import warnings
import pickle

from T_EEGCN_Framework.EEGCN import EEGCN
from T_EEGCN_Framework.Lstm_model import LSTMs
from units import get_all_input, last_value, collate_fn_old, MyData
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gcn_in', type=int, default=32)
    parser.add_argument('--gcn_hid', type=int, default=48)
    parser.add_argument('--gcn_out', type=int, default=48)
    parser.add_argument('--lstm_hid', type=int, default=64)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_drop', type=int, default=0.5)
    parser.add_argument('--num_areas', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=4000)
    parser.add_argument('--drop', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--period', type=int, default=9)  # 9 periods for train,Time Window is about 1 week

    parser.add_argument('--label_path', type=str, default='data_for_train_example/label_tools.pkl')
    parser.add_argument('--graphs_path', type=str, default='data_for_train_example/sub_net_tools.pkl')
    parser.add_argument('--n_feat_path', type=str, default='data_for_train_example/n_feat_tools.pkl')
    parser.add_argument('--e_feat_path', type=str, default='data_for_train_example/e_feat_tools.pkl')

    args = parser.parse_args()
    k = args.period

    save_cate = 'TOOLS_OPPST'
    print(save_cate)

    # load data
    # If you encounter a data read error, you must modify the source code of the pickle class file library with the
    # feedback of the Units error. Some library versions disable reading because the pickle attribute is false,
    # and simply comment it out can solve the problem
    dgls = get_all_input(args.graphs_path, True)
    n_feat = np.array(get_all_input(args.n_feat_path, False))
    e_feat = np.array(get_all_input(args.e_feat_path, False))
    labels = np.array(get_all_input(args.label_path, False))

    hot_rate = sum(sum(labels == 0))/sum(sum(labels == 1))

    # train, test split
    n = len(dgls)
    print('num of st-nets = {}'.format(n))
    split = int(n * .8)
    index = np.arange(n)
    np.random.seed(50)
    np.random.shuffle(index)
    train_index, test_index = index[:split], index[split:]

    # prep labels
    train_labels = labels[train_index]
    test_labels = torch.FloatTensor(labels[test_index])
    test_labels = test_labels.cuda()

    # prep input data
    trainGs, testGs = [dgls[i] for i in train_index], [dgls[i] for i in test_index]
    testGs = [dgl.batch([dgl.DGLGraph(u[i]) for u in testGs]) for i in range(k)]

    train_n_feat, test_n_feat = [n_feat[i] for i in train_index], [n_feat[i] for i in test_index]
    test_n_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in test_n_feat])) for i in range(k)]

    train_e_feat, test_e_feat = [e_feat[i] for i in train_index], [e_feat[i] for i in test_index]
    test_e_feat = [torch.FloatTensor(np.concatenate([inp[i] for inp in test_e_feat])) for i in range(k)]

    data = MyData(trainGs, train_n_feat, train_e_feat, train_labels)
    data_loader = DataLoader(data, batch_size=1000, shuffle=False, collate_fn=collate_fn_old)
    warnings.filterwarnings("ignore")
    '''*************************************************************************************************************'''
    # define models
    model_egcn = EEGCN(args.gcn_in, args.gcn_hid, args.gcn_out)
    model_l = LSTMs(args.gcn_out, args.lstm_hid, args.num_areas, args.lstm_layers, args.lstm_drop)
    model_egcn.cuda()
    model_l.cuda()
    print(model_egcn, model_l)
    print('lr:', args.lr)

    parameters = list(model_egcn.parameters()) + list(model_l.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    dropout = nn.Dropout(args.drop)

    MSE_loss = nn.MSELoss().cuda()
    # BCE_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hot_rate])).cuda()

    epoch_loss_all = []
    train_score = []
    test_score = []
    pre_score = []
    auc_score = []
    re_score = []
    for epoch in tqdm(range(args.epoch)):
        model_egcn.train()
        model_l.train()
        for step, (g, nf, ef, batch_labels) in enumerate(data_loader):
            batch_labels = batch_labels.cuda()
            # Run through GCN
            sequence = torch.stack([model_egcn(g[i], [nf[i].cuda(), ef[i].cuda()]) for i in range(k)], 1)
            # Temporal graph embeddings through lstm
            last, out = model_l(sequence)

            loss = MSE_loss(out, batch_labels)

            #  back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_all.append(loss.item())

        # eval
        model_egcn.eval()
        model_l.eval()

        test_sequence = torch.stack([model_egcn(testGs[i], [test_n_feat[i].cuda(), test_e_feat[i].cuda()])
                                     for i in range(k)], 1)  # 生成序列
        last, out = model_l(test_sequence)

        temp_label = test_labels.cpu().view(1, -1).squeeze(0)
        temp_pre = np.int64(out.cpu().detach().numpy() > 0.5).reshape(1, -1).squeeze(0)

        test_f1 = f1_score(temp_label, temp_pre)
        test_pre = precision_score(temp_label, temp_pre)
        test_auc = roc_auc_score(temp_label, temp_pre)
        test_recall = recall_score(temp_label, temp_pre)

        test_score.append(test_f1)
        pre_score.append(test_pre)
        auc_score.append(test_auc)
        re_score.append(test_recall)

        if epoch % 10 == 0:
            print('Epoch %d | Train Loss: %.4f | Test F1: %.4f | Test AUC: %.4f '
                  % (epoch, loss.item(), test_f1, test_auc))

            metrics_res = [epoch_loss_all, pre_score, re_score, test_score, auc_score]
            with open('value_result/cate_' + save_cate + '_result', 'wb') as f:  # save result,if no path,then create
                pickle.dump(metrics_res, f)

            if epoch % 1000 == 0:
                # save the params of net
                torch.save(model_egcn.state_dict(), 'egcn_params_' + save_cate + '.pkl')
                torch.save(model_l.state_dict(), 'lstm_params_' + save_cate + '.pkl')

    print('lr:{}, max(pre_score):{}, max(re_score):{}, max(f1_score):{}, max(auc):{}'
          .format(args.lr, max(pre_score), max(re_score), max(test_score), max(auc_score)))

    print('last_pre:{}\nlast_re:{}\nlast_f1:{}\nlast_auc:{}'.format(last_value(pre_score[-10:]),
                                                                    last_value(re_score[-10:]),
                                                                    last_value(test_score[-10:]),
                                                                    last_value(auc_score[-10:])))
