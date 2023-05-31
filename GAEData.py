import copy
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse


class GAEData(InMemoryDataset):

    def __init__(self, root, split='train', fold=1, transform=None, pre_transform=None,
                 pre_filter=None, th=0.5):
        self.th = th
        assert split in ['train', 'val', 'test']
        assert fold in [1, 2, 3, 4, 5]
        super().__init__(root, transform, pre_transform, pre_filter)

        if split == 'train' and fold == 1:
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'train' and fold == 2:
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'train' and fold == 3:
            self.data, self.slices = torch.load(self.processed_paths[2])
        elif split == 'train' and fold == 4:
            self.data, self.slices = torch.load(self.processed_paths[3])
        elif split == 'train' and fold == 5:
            self.data, self.slices = torch.load(self.processed_paths[4])

        # elif split == 'test':
        #     self.data, self.slices = torch.load(self.processed_paths[10])

    @property
    def raw_file_names(self):
        return ['x_train1.npy', 'y_train1.npy', 'train_disease1.npy',
                'x_train2.npy', 'y_train2.npy', 'train_disease2.npy',
                'x_train3.npy', 'y_train3.npy', 'train_disease3.npy',
                'x_train4.npy', 'y_train4.npy', 'train_disease4.npy',
                'x_train5.npy', 'y_train5.npy', 'train_disease5.npy', ]

    @property
    def processed_file_names(self):
        return ['train1.pt', 'train2.pt', 'train3.pt', 'train4.pt', 'train5.pt']

    def download(self):
        # path = download_url(self.url, self.root)
        # extract_zip(path, self.raw_dir)
        # os.unlink(path)
        pass

    def process(self):
        # for i in range(1, 6):
        #     x_path = osp.join(self.raw_dir, f'x_train{i}.npy')
        #     disease_path = osp.join(self.raw_dir, f'train_disease{i}.npy')
        #     x = pd.DataFrame(np.load(x_path, allow_pickle=True))
        #     disease = pd.DataFrame(np.load(disease_path, allow_pickle=True))
        #     data = edge_process(x, disease, self.th)
        #     torch.save(self.collate(data), self.processed_paths[(i-1)])
        #     del data
        for i in range(1, 6):
            x_path = osp.join(self.root, 'raw', f'x_train{i}.npy')
            y_path = osp.join(self.root, 'raw', f'y_train{i}.npy')
            disease_path = osp.join(self.root, 'raw', f'train_disease{i}.npy')
            # x_path = osp.join(self.raw_dir, f'x_test_all.npy')
            # y_path = osp.join(self.raw_dir, f'y_test_all.npy')
            # disease_path = osp.join(self.raw_dir, f'test_disease_all.npy')
            x = pd.DataFrame(np.load(x_path, allow_pickle=True))
            y = pd.DataFrame(np.load(y_path, allow_pickle=True))
            disease = pd.DataFrame(np.load(disease_path, allow_pickle=True))
            data,_,_,_,_ = edge_process(x, y, disease, self.th)
            torch.save(self.collate(data), self.processed_paths[(i - 1)])
            del data


def pearson_adj(x, th):
    sh = x.shape[0]
    x = np.array(x)
    my_rho = np.corrcoef(x, x)
    pearson = my_rho[:sh, :sh]
    pos_pearson = copy.deepcopy(pearson)
    row, col = np.diag_indices_from(pos_pearson)

    pos_pearson[row, col] = 0
    pos_pearson[pos_pearson >= th] = 1
    pos_pearson[pos_pearson <= -th] = 1
    pos_pearson[pos_pearson != 1] = 0

    neg_pearson = copy.deepcopy(pearson)
    neg_pearson[neg_pearson >= th] = 0
    neg_pearson[neg_pearson <= -th] = 0
    neg_pearson[neg_pearson != 0] = 1
    return torch.tensor(pos_pearson), torch.tensor(neg_pearson)


def cancer_type_adj(strat):
    strat.replace("READ", "COAD", inplace=True)
    sh = len(strat)
    pos_adj = np.zeros((sh, sh))
    neg_adj = np.ones((sh, sh))
    uns = np.unique(strat)
    for i in uns:
        k = (strat == i)
        index = np.where(k)[0]
        for j in index:
            pos_adj[j, index] = 1
            neg_adj[j, index] = 0
    row, col = np.diag_indices_from(pos_adj)
    pos_adj[row, col] = 0
    return torch.tensor(pos_adj), torch.tensor(neg_adj)


def edge_process(x_train, y_train, train_disease, th):
    data = []
    train_pearson_pos_adj,train_pearson_neg_adj= pearson_adj(x_train, th)
    train_pearson_pos_edge_index = dense_to_sparse(train_pearson_pos_adj)[0]
    train_pearson_neg_edge_index = dense_to_sparse(train_pearson_neg_adj)[0]
    del train_pearson_pos_adj,train_pearson_neg_adj

    train_cancer_pos_type_adj, train_cancer_neg_type_adj = cancer_type_adj(train_disease)
    train_cancer_pos_edge_index = dense_to_sparse(train_cancer_pos_type_adj)[0]
    train_cancer_neg_edge_index = dense_to_sparse(train_cancer_neg_type_adj)[0]
    del train_cancer_pos_type_adj, train_cancer_neg_type_adj

    x_train = torch.tensor(data=x_train.values)
    y_train = torch.tensor(data=y_train.values)

    train_data = Data(x_train, pearson_pos_edge_index=train_pearson_pos_edge_index,
                      pearson_neg_edge_index=train_pearson_neg_edge_index,
                      cancer_pos_edge_index=train_cancer_pos_edge_index,
                      cancer_neg_edge_index=train_cancer_neg_edge_index,
                      y_train=y_train)
    # k_hop_subgraph(node_idx=[0], num_hops=1, edge_index=train_data.pearson_pos_edge_index)
    data.append(train_data)
    del x_train,y_train,train_data
    return data, train_pearson_pos_edge_index,train_pearson_neg_edge_index, train_cancer_pos_edge_index,train_cancer_neg_edge_index

def cos_sim(train_x,test_x,train_disease,test_disease):
    disease_index = []
    for index, row in train_disease.iterrows():
        if row.values[0] == test_disease.values[0][0]:
          disease_index.append(index)
    train_x_disease = train_x.iloc[disease_index,:]
    x1 = torch.tensor(train_x_disease.values)
    x2 = torch.tensor(test_x.values)
    cos_sim1 = F.cosine_similarity(x1, x2, dim=1).detach().numpy()
    replace_index = np.where(cos_sim1 == cos_sim1.max())[0][0]
    return disease_index[replace_index]

def test_edge_process(path, fold, item, th,mode='val'):  # TODO: renew test 1by1

    assert mode in ['val', 'test']
    assert fold in [1, 2, 3, 4, 5]
    x_path = osp.join(path, 'raw', f'x_train{fold}.npy')
    y_path = osp.join(path, 'raw', f'y_train{fold}.npy')
    disease_path = osp.join(path, 'raw', f'train_disease{fold}.npy')
    x = pd.DataFrame(np.load(x_path, allow_pickle=True))
    y = pd.DataFrame(np.load(y_path, allow_pickle=True))
    disease = pd.DataFrame(np.load(disease_path, allow_pickle=True))

    if mode == 'val':
        add_x_path = osp.join(path, 'raw', f'x_val{fold}.npy')
        add_y_path = osp.join(path, 'raw', f'y_val{fold}.npy')
        add_disease_path = osp.join(path, 'raw', f'val_disease{fold}.npy')
        add_x = pd.DataFrame(np.load(add_x_path, allow_pickle=True))
        add_y = pd.DataFrame(np.load(add_y_path, allow_pickle=True))
        add_disease = pd.DataFrame(np.load(add_disease_path, allow_pickle=True))
    elif mode == 'test':
        add_x_path = osp.join(path, 'raw', f'x_test_all.npy')
        add_y_path = osp.join(path, 'raw', f'y_test_all.npy')
        add_disease_path = osp.join(path, 'raw', f'test_disease_all.npy')
        add_x = pd.DataFrame(np.load(add_x_path, allow_pickle=True))
        add_y = pd.DataFrame(np.load(add_y_path, allow_pickle=True))
        add_disease = pd.DataFrame(np.load(add_disease_path, allow_pickle=True))
    item_x = add_x[item:item + 1]
    item_y = add_y[item:item + 1]
    item_disease = add_disease[item:item + 1]
    del add_x, add_y, add_disease

    replace_index = cos_sim(x, item_x, disease, item_disease)
    x.iloc[replace_index, :] = item_x
    y.iloc[replace_index, :] = item_y
    del item_x, item_y

    train_data, train_pearson_pos_edge_index,train_pearson_neg_edge_index, train_cancer_pos_edge_index,train_cancer_neg_edge_index = edge_process(x, y, disease, th)



    x_train = torch.tensor(data=x.values)
    y_train = torch.tensor(data=y.values)


    test_data = Data(x_train, pearson_pos_edge_index=train_pearson_pos_edge_index,
                     pearson_neg_edge_index = train_pearson_neg_edge_index,
                     cancer_pos_edge_index=train_cancer_pos_edge_index,
                     cancer_neg_edge_index=train_cancer_neg_edge_index,
                     y_train=y_train,
                     replace_index = replace_index)
    data = []
    data.append(test_data)
    del test_data,x_train,y_train, train_data, train_pearson_pos_edge_index,train_pearson_neg_edge_index, train_cancer_pos_edge_index,train_cancer_neg_edge_index
    return data
