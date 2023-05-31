import os.path
import os.path as osp
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import datetime
import random
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.nn import LayerNorm
from GAEData import GAEData, edge_process, test_edge_process
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import matplotlib
from util_args import get_args
device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
matplotlib.use('Agg')

random.seed(9)
np.random.seed(47)
torch.manual_seed(47)
torch.cuda.manual_seed(47)
torch.cuda.manual_seed_all(47)
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

args = get_args()
OC = args.GAE_out_channels
LR = args.GAE_lr
EPOCH = args.GAE_epochs
PATIENCE = args.GAE_patience
BATCH_SIZE = args.GAE_batch_size
TH = args.GAE_th
NF = args.GAE_num_features
KEYWORD =args.keyword
GN = args.LOSS_gamma_neg
GP = args.LOSS_gamma_pos
CLIP = args.LOSS_clip

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.norm = LayerNorm(out_channels)

        self.layer3 = nn.Sequential(nn.Linear(64, 6), nn.Sigmoid())

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm(x).relu()
        return self.conv2(x, edge_index),self.layer3(x)


def GAEtrain(model, device, loader, optimizer,BCEloss):
    model.train()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z1,z2 = model.encode(batch.x, batch.pearson_pos_edge_index)
        loss1 = model.recon_loss(z1, pos_edge_index=batch.cancer_pos_edge_index,
                                neg_edge_index=batch.cancer_neg_edge_index)
        # loss1 = model.recon_loss(z1, pos_edge_index=batch.pearson_pos_edge_index,
        #                         neg_edge_index=batch.pearson_neg_edge_index)
        # loss = model.recon_loss(z, pos_edge_index=batch.pearson_pos_edge_index)
        loss2 = BCEloss(z2, batch.y_train[:, :-1])
        print('-------------------------')
        print(loss1)
        print(loss2)
        print('-------------------------')

        loss = loss1+(loss2*100)
        # loss=loss2
        loss.backward()
        optimizer.step()
    return loss.item(), z1


def GAEtest(model, loader):
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        z = model.encode(batch.x, batch.pearson_pos_edge_index)
    return model.test(z, batch.pearson_pos_edge_index, batch.cancer_neg_edge_index)


def GAEuse(model, loader):  # TODO：shuffle?
    # res = torch.tensor(np.array([]))
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        z1,z2 = model.encode(batch.x, batch.pearson_pos_edge_index)
        # res = torch.cat([res, z], 0)
        # return model.test(z, batch.pearson_edge_index, batch.cancer_edge_index)
    return z1


def train():
    out_channels = OC
    num_features = NF
    epochs = EPOCH
    batch_size = BATCH_SIZE
    th = TH  # TODO: OTHER TH
    lr =LR

    keyword = KEYWORD
    patience = PATIENCE
    BCEloss = AsymmetricLoss(gamma_neg=GN, gamma_pos=GP, clip=CLIP, disable_torch_grad_focal_loss=True)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'GAE')
    model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'model', 'GAE', f'{keyword}')
    if os.path.exists(model_path) == True:
        pass
    else:
        os.mkdir(model_path)
    loss_list = []
    epochs_list = []
    for i in range(1, 6):
        wait_txt = osp.join(osp.dirname(osp.realpath(__file__)), f'train_loss_{keyword}_{i}.txt')
        start = datetime.datetime.now()
        with open(wait_txt, 'a') as file:
            file.write('\n' + str(start) + 'start train' + str(i))

        train_dataset = GAEData(path, split='train', fold=i, th=th)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print('start train')
        model = GAE(GCNEncoder(num_features, out_channels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        checkpoint = osp.join(model_path, f'GAE_model_{keyword}_{i}.pt')
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint)
        for epoch in range(epochs):
            loss, train_embedding = GAEtrain(model, device, train_loader, optimizer,BCEloss)
            print(f'Epoch: {epoch:03d}, LOSS: {loss:.4f}')

            with open(wait_txt, 'a') as file:
                file.write('\n' + f'Epoch: {epoch:03d}, LOSS: {loss:.4f}')

            early_stopping(loss, model)
            if i == 1:
                loss_list.append([loss])
            else:
                try:
                    loss_list[epoch] += [loss]
                except:
                    loss_list.append([loss])
            if early_stopping.early_stop:
                print("Early stopping")
                break
        del train_dataset

        # 算一个train跑的时间
        end = datetime.datetime.now()
        print("程序运行时间：" + str((end - start).seconds) + "秒")
        with open(wait_txt, 'a') as file:
            file.write('\n' + str(end) + 'start train' + str(i))
            file.write('\n' + "程序运行时间：" + str((end - start).seconds) + "秒")

        epochs_list.append(epoch)
        model = GAE(GCNEncoder(num_features, out_channels)).to(device)
        model.load_state_dict(torch.load(osp.join(model_path, f'GAE_model_{keyword}_{i}.pt')))

        out = GAEuse(model, train_loader)

        for step, (_, _, _, _,_, y_trian,_,_) in enumerate(train_loader):
            y = y_trian[1]
        data = torch.utils.data.TensorDataset(out, y)

        torch.save(data,
                   osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'RAS', 'processed') + f'/train_{keyword}_{i}.pt')
        print('done')

    min_epochs = min(epochs_list)
    epochs_list = [i for i in range(min_epochs)]
    loss_list = loss_list[:min_epochs]

    plot_learning_curve(loss_list, epochs_list, th,out_channels)

def test(mode):
    out_channels = OC
    num_features = NF
    th = TH  # TODO: OTHER TH
    keyword = KEYWORD
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'GAE')
    model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'model', 'GAE', f'{keyword}')
    if os.path.exists(model_path) == True:
        pass
    else:
        os.mkdir(model_path)


    for i in range(1, 6):
        # for val & test
        if mode == 'val':
            x_path = osp.join(path, 'raw', f'x_val{i}.npy')
        else:
            x_path = osp.join(path, 'raw', f'x_test_all.npy')
        x = pd.DataFrame(np.load(x_path, allow_pickle=True))
        num = len(x)
        del x_path, x

        wait_txt = osp.join(osp.dirname(osp.realpath(__file__)), f'{mode}_loss{i}.txt')
        for item in range(num):
            #  算一个val跑的时间
            start = datetime.datetime.now()
            with open(wait_txt, 'a') as file:
                file.write('\n'+str(start)+f'start {mode}'+ str(item))
            print(f'start {mode} {item}')

            test_dataset = test_edge_process(path, fold=i, item=item, th=th, mode=mode)
            test_data = test_dataset[0].to(device)

            model = GAE(GCNEncoder(num_features, out_channels)).to(device)
            model.load_state_dict(torch.load(osp.join(model_path, f'GAE_model_{keyword}_{i}.pt'),map_location='cpu'))

            model.eval()
            out,z = model.encode(test_data.x, test_data.pearson_pos_edge_index)
            print('a')
            if item == 0:
                emb = torch.unsqueeze(out[test_data.replace_index], 0)
                y = torch.unsqueeze(test_data.y_train[test_data.replace_index], 0)
                # y = single_dataset.y_trian
            else:
                emb_new = torch.unsqueeze(out[test_data.replace_index], 0)
                y_new = torch.unsqueeze(test_dataset[0].y_train[test_data.replace_index], 0)

                emb = torch.cat((emb, emb_new), 0)
                y = torch.cat((y, y_new), 0)

            del test_data,test_dataset,out
            #算一个val跑的时间
            end = datetime.datetime.now()
            print("程序运行时间：" + str((end - start).seconds) + "秒")
            with open(wait_txt, 'a') as file:
                file.write('\n'+str(end)+f'start {mode}'+ str(item))
                file.write('\n' + "程序运行时间：" + str((end - start).seconds) + "秒")
        data = torch.utils.data.TensorDataset(emb, y)
        torch.save(data,
                   osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'RAS', 'processed') + f'/{mode}_{keyword}_{i}.pt')
        del data
        print(f'Fold {i} processed!')

def plot_learning_curve(loss_list, epoch_list, th,out_channels):
    plt.figure()
    plt.title('learning curve')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    train_scores_mean = np.mean(loss_list, axis=1)
    train_scores_std = np.std(loss_list, axis=1)
    plt.grid()

    plt.fill_between(epoch_list, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.plot(epoch_list, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.legend(loc="best")
    plt.savefig(f'loss{str(out_channels)}.png')
    return plt


def main():
    train()
    test('val')
    test('test')

if __name__ == "__main__":
    main()