"""Borrowed from https://github.com/PetarV-/DGI"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random

class Infomax(nn.Module):

    def __init__(self, n_h1, n_h2):
        super(Infomax, self).__init__()
        self.n_h1 = n_h1
        self.n_h2 = n_h2
        self.trans1 = nn.Linear(2 * n_h2, n_h2)
        self.trans2 = nn.Linear(2, n_h2)

        self.f_k = nn.Bilinear(n_h1, n_h2, 1)
        self.node_indices = None
        self.bce = nn.BCEWithLogitsLoss()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)
        c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x))
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))#.view(-1,1)

        return logits

    def get_loss(self, x, f_x, edge_set, **kwargs):
        self.node_indices = None
        if self.node_indices is None:
            self.node_indices = np.arange(x.shape[0])
            labels_pos = torch.ones(x.shape[0])
            labels_neg = torch.zeros(x.shape[0])
            self.labels = torch.cat((labels_pos, labels_neg))
            if torch.cuda.is_available():
                self.labels = self.labels.cuda()
        pos = (x)

        err_rate = 1.1
        bst_idx = None
        early_stopping = 0
        while True:
            idx = np.random.permutation(self.node_indices)
            flag = 0
            for i in range(idx.shape[0]):
                u, v = i, idx[i]
                if u == v:
                    flag += 1
                    continue
                if u in edge_set and v in edge_set[u]:
                    flag += 1
                    continue
                if v in edge_set and u in edge_set[v]:
                    flag += 1
                    continue
            if flag / x.shape[0] < err_rate:
                err_rate = flag / x.shape[0]
                bst_idx = idx
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping > 5:
                    break

        neg = (x[bst_idx])
        logits = self.forward(f_x, pos, neg)
        loss = self.bce(logits, self.labels)
        return loss
    
    def forward_1(self, c, h_pl, c_neg, h_mi, s_bias1=None, s_bias2=None):
        # c_x = torch.unsqueeze(c, 1)
        # c_x = c_x.expand_as(h_pl)
        c_x = c
        c_x_neg = c_neg
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x))
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x_neg))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))#.view(-1,1)

        return logits

    def get_loss_1(self, x, f_x, edge_set, **kwargs):
        if f_x.size(-1) != self.n_h2:
            # print(f_x.size())
            if f_x.size(-1) == 2:
                f_x = self.trans2(f_x)
            else:
                f_x = self.trans1(f_x)
            # print('==>', f_x.size())

        self.node_indices = None
        if self.node_indices is None:
            self.node_indices = np.arange(x.shape[0])
        
        pos = (x)

        select_idx = []
        idx = np.random.permutation(self.node_indices)
        for i in range(idx.shape[0]):
            flag = True
            u, v = i, idx[i]
            if u == v:
                flag = False
            # elif u in edge_set and v in edge_set[u]:
            #     flag = False
            # elif v in edge_set and u in edge_set[v]:
            #     flag = False
            if flag:
                select_idx.append(i)

        neg = (x[idx])[select_idx]
        f_x_neg = f_x[select_idx]

        labels_pos = torch.ones(x.shape[0])
        labels_neg = torch.zeros(neg.shape[0])
        self.labels = torch.cat((labels_pos, labels_neg))
        if torch.cuda.is_available():
            self.labels = self.labels.cuda()

        logits = self.forward_1(f_x, pos, f_x_neg, neg)
        loss = self.bce(logits, self.labels)
        return loss
    
    

