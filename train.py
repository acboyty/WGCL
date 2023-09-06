import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification

from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset

from GCN_utils import GCNConv_wo_weight
from eval import log_regression, MulticlassEvaluator



def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()

    data_x = x
    data_edge_index = edge_index
    if args.dataset in ['Coauthor-Phy']:
        # print(data_x.size(), data_edge_index.size())
        idx = torch.randperm(data_x.size(0))[:int(data_x.size(0) * .5)].numpy()
        idx_set = {}
        for i, id in enumerate(idx):
            idx_set[id] = i
        x = data_x[idx].cuda()

        data_edge_index = data_edge_index.cpu().numpy()
        row = []
        col = []
        for i in range(data_edge_index.shape[1]):
            u, v = data_edge_index[0, i], data_edge_index[1, i]
            if u in idx_set and v in idx_set:
                row.append(idx_set[u])
                col.append(idx_set[v])
        edge_index = torch.stack([torch.Tensor(row), torch.Tensor(col)], dim=0).long().cuda()
        # print(x.size(), edge_index.size())

    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    z1, zs1 = model(x_1, edge_index_1)
    z2, zs2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=1024 if args.dataset in ['Coauthor-Phy', 'ogbn-arxiv'] else None)

    with torch.no_grad():
        GCNConv_wo_w = GCNConv_wo_weight()
        GCNConv_wo_w.eval()
        x_neighbor_1 = GCNConv_wo_w(GCNConv_wo_w(x_1, edge_index_1), edge_index_1)
        x_neighbor_2 = GCNConv_wo_w(GCNConv_wo_w(x_2, edge_index_2), edge_index_2)

    loss_corr = None
    for zi1, zi2 in zip(zs1, zs2):
        temp_loss = model.infomax.get_loss_1(x_neighbor_1, zi1, None) + model.infomax.get_loss_1(x_neighbor_1, zi2, None) + \
                    model.infomax.get_loss_1(x_neighbor_2, zi1, None) + model.infomax.get_loss_1(x_neighbor_2, zi2, None)
        if loss_corr is None:
            loss_corr = temp_loss / 2
        else:
            loss_corr += temp_loss / 2
    
    loss += (config['_lambda'] * loss_corr)
    loss.backward()
    optimizer.step()

    return loss.item()


def test():
    model.eval()
    z, _ = model(data.x, data.edge_index)
    
    evaluator = MulticlassEvaluator()
    if args.dataset == 'ogbn-arxiv':
        acc = log_regression(z, dataset, evaluator, split='ogb', num_epochs=5000)['acc']
    elif args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        accs = []
        for i in range(5):
            acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--seed', type=int, default=39788)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    })[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                        'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
        name = 'dblp' if name == 'DBLP' else name
        root_path = osp.expanduser('~/datasets')

        if name == 'Coauthor-CS':
            return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

        if name == 'Coauthor-Phy':
            return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

        if name == 'WikiCS':
            # return WikiCS(root=path, transform=T.NormalizeFeatures())
            return WikiCS(root=path)

        if name == 'Amazon-Computers':
            return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

        if name == 'Amazon-Photo':
            return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

        if name.startswith('ogbn'):
            # return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())
            return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name)

        return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())
        # return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name)


    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index)

        if epoch % 50 == 0:
            acc = test()

            print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')

    acc = test()

    f = open(f'results/{args.dataset}.txt','a')
    print(f'{acc}', file=f)
    f.close()
