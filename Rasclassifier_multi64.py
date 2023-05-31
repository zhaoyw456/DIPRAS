import torch.nn as nn
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random

device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
from util_args import get_args
import sys
import math
import os.path
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os.path as osp
from torch_geometric.loader import DataLoader

import random
random.seed(9)
np.random.seed(47)
torch.manual_seed(47)
torch.cuda.manual_seed(47)
torch.cuda.manual_seed_all(47)
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

args = get_args()
LR = args.RAS_lr
EPOCH = args.RAS_epoch
BATCH_SIZE = args.RAS_batch_size
PATIENCE = args.RAS_patience
GN = args.LOSS_gamma_neg
GP = args.LOSS_gamma_pos
CLIP = args.LOSS_clip
KEYWORD =args.keyword

class RASmodel(nn.Module):
    def __init__(self):
        super(RASmodel, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32),nn.Dropout(p=0.5))

        self.layer2 = nn.Sequential(nn.Linear(32, 8), nn.BatchNorm1d(8),nn.Dropout(p=0.5))

        self.layer3 = nn.Sequential(nn.Linear(8, 6),nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
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
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.79,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

        self.celoss = torch.nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, label):
        '''
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        '''

        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        ce_loss=(-(label * torch.log(logits)) - (
                    (1 - label) * torch.log(1 - logits)))
        # ce_loss=(-(label * torch.log(torch.softmax(logits, dim=1))) - (
        #             (1 - label) * torch.log(1 - torch.softmax(logits, dim=1))))
        pt = torch.where(label == 1, logits, 1 - logits)
        # ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

def plot_learning_curve(loss_list, epoch_list,i,mode):
    plt.figure()
    plt.title('learning curve')

    plt.xlabel("Epoch")
    plt.ylabel("loss")

    train_scores_mean = np.mean(loss_list, axis=1)
    train_scores_std = np.std(loss_list, axis=1)
    plt.grid()

    plt.fill_between(epoch_list, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.plot(epoch_list, train_scores_mean, 'o-', color="r",
             label=f"{mode} score")
    plt.legend(loc="best")
    plt.savefig(f'loss{str(mode)}_{i}.png')
    return plt
def get_threshold_metrics(y_true, y_pred, drop_intermediate=False,
                          disease='all'):
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score

    roc_columns = ['fpr', 'tpr', 'threshold']
    pr_columns = ['precision', 'recall', 'threshold']

    if drop_intermediate:
        roc_items = zip(roc_columns,
                        roc_curve(y_true, y_pred, drop_intermediate=False))
    else:
        roc_items = zip(roc_columns, roc_curve(y_true, y_pred))

    roc_df = pd.DataFrame.from_dict(dict(roc_items))

    prec, rec, thresh = precision_recall_curve(y_true, y_pred)
    pr_df = pd.DataFrame.from_records([prec, rec]).T
    pr_df = pd.concat([pr_df, pd.Series(thresh)], ignore_index=True, axis=1)
    pr_df.columns = pr_columns

    auroc = roc_auc_score(y_true, y_pred, average='weighted')
    aupr = average_precision_score(y_true, y_pred, average='weighted')

    return {'auroc': auroc, 'aupr': aupr, 'roc_df': roc_df,
            'pr_df': pr_df, 'disease': disease}

def calculate_performance(labels, predict_score,epoch,th = 0.5):
    labels = np.array(labels)
    predict_score = np.array(predict_score)
    labels_count = labels.shape[0]
    labels_len = labels.shape[1]
    labels_d = {}
    tp = [0 for i in range(labels_len)]
    fp = [0 for i in range(labels_len)]
    tn = [0 for i in range(labels_len)]
    fn = [0 for i in range(labels_len)]

    for labels_n in range(labels_len):
        label = labels[:,labels_n]
        predict_s = predict_score[:,labels_n]
        for index in range(labels_count):
            if label[index] == 1 and predict_s[index] >= th:
                tp[labels_n] += 1
            elif label[index] == 1 and predict_s[index] < th:
                fn[labels_n] += 1
            elif label[index] == 0 and predict_s[index] >= th:
                fp[labels_n] += 1
            else:
                tn[labels_n] += 1

    pre_y = np.where(predict_score > 0.5, 1, 0)
    for labels_i in range(labels_count):
        lsum = np.sum(labels[labels_i])
        if lsum > 1 and (labels[labels_i] == pre_y[labels_i]).all():
            if 'two_right' not in labels_d:
                labels_d['two_right'] = 1
            else:
                labels_d['two_right'] += 1
        if lsum not in labels_d:
            labels_d[lsum] = 1
        else:
            labels_d[lsum] += 1

    # for index, value in enumerate(tp):
    #     if value==0:
    #         tp[index]=1
    # for index, value in enumerate(fp):
    #     if value==0:
    #         fp[index]=1
    # for index, value in enumerate(tn):
    #     if value==0:
    #         tn[index]=1
    # for index, value in enumerate(fn):
    #     if value==0:
    #         fn[index]=1

    print(tp)
    print(fp)
    print(tn)
    print(fn)
    acc_all = []
    for index in range(labels_len):
        acc = float(tp[index] + tn[index]) / (tp[index] +fp[index] + tn[index] + fn[index] + sys.float_info.epsilon)
        acc_all.append(acc)
    # print(acc_all)

    precision_all = []
    for index in range(labels_len):
        precision = float(tp[index]) / (tp[index] +fp[index] + sys.float_info.epsilon)
        precision_all.append(precision)
    # print(precision_all)

    sensitivity_all = [] #recall
    for index in range(labels_len):
        sensitivity = float(tp[index]) / (tp[index] + fn[index] + sys.float_info.epsilon)
        sensitivity_all.append(sensitivity)
    # print(sensitivity_all)

    specificity_all = []
    for index in range(labels_len):
        specificity = float(tn[index]) / (tn[index] + fp[index] + sys.float_info.epsilon)
        specificity_all.append(specificity)
    # print(specificity_all)

    f1_all = []
    for index in range(labels_len):
        f1 = 2 * precision_all[index] * sensitivity_all[index] / (
                    precision_all[index] + sensitivity_all[index] + sys.float_info.epsilon)
        f1_all.append(f1)
    # print(f1_all)

    mcc_all = []
    for index in range(labels_len):
        mcc = float(tp[index] * tn[index] - fp[index] * fn[index]) / (
                math.sqrt((tp[index] + fp[index]) * (tp[index] + fn[index]) * (tn[index] + fp[index]) * (tn[index] + fn[index]))
                + sys.float_info.epsilon)
        mcc_all.append(mcc)
    # print(mcc_all)

    strResults = f'epoch:{epoch},tp:{tp},fn:{fn},tn:{tn},fp:{fp},' \
                 f'acc:{acc_all},precision:{precision_all},' \
                 f'sensitivity:{sensitivity_all},specificity:{specificity_all},' \
                 f'f1:{f1_all},mcc:{mcc_all}'
    print(strResults)
    print(labels_d)
    # aps = average_precision_score(labels, predict_score)
    # fpr, tpr, _ = roc_curve(labels, predict_score)
    # aucResults = auc(fpr, tpr)
    #
    return acc_all,precision_all,sensitivity_all,specificity_all, f1_all,mcc_all

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
def multi_calculate_performance(y_true, y_pred,epoch,th = 0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred > th, 1, 0)
    print(y_true.shape)
    print(y_pred.shape)

    #acc
    tempacc = 0
    for i in range(y_true.shape[0]):
        tempacc += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    acc = tempacc / y_true.shape[0]

    # Hamming Loss
    temphl = 0
    for i in range(y_true.shape[0]):
        temphl += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    hl = temphl / (y_true.shape[0] * y_true.shape[1])

    # Recall
    temprecall = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temprecall += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    recall = temprecall / y_true.shape[0]

    # Precision
    tempprecision = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        tempprecision += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    precision = tempprecision / y_true.shape[0]

    # F1-Measure
    tempf1 = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        tempf1 += (2 * sum(np.logical_and(y_true[i], y_pred[i]))) / (sum(y_true[i]) + sum(y_pred[i]))
    f1 = tempf1 / y_true.shape[0]

    strResults = f'epoch:{epoch}', \
                 f'acc:{acc},Hamming Loss:{hl},' \
                 f'recall:{recall},precision:{precision},' \
                 f'f1:{f1}'
    print(strResults)
    return acc,hl,recall,precision,f1
def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bce_loss

    weighted_loss = alpha * loss
    loss = torch.sum(weighted_loss)
    loss /= torch.sum(labels)
    return loss
def RAStrain(train_loader,val_loader,path,model_path,i):

    dnn = RASmodel()

    # BCEloss = nn.BCELoss()
    # BCEloss = FocalLoss()
    # BCEloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]))
    BCEloss = AsymmetricLoss(gamma_neg=GN, gamma_pos=GP, clip=CLIP, disable_torch_grad_focal_loss=True)
    optimizer = torch.optim.Adam(dnn.parameters(), lr=LR)

    checkpoint = osp.join(model_path, f'RAS_model_{i}.pt')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(PATIENCE, verbose=True, path=checkpoint)

    epochs_list = []
    train_list = []
    val_list = []
    for epoch in range(EPOCH):
        train_losses = []
        dnn.train()
        for step, (b_x, b_y) in enumerate(train_loader):
            train_index = torch.where(b_y[:, -1] == 1)
            output = dnn(b_x[train_index].to(device))
            # output = dnn(b_x.to(device))
            # loss = binary_cross_entropy_with_logits(new_output, b_y.to(device))
            # t_loss = BCEloss(output, b_y[:,:-1].double().to(device))
            t_loss = focal_loss(output, b_y[train_index][:, :-1].double().to(device), 0.75, 2)
            train_losses.append(t_loss.item())
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
        train_loss = np.mean(train_losses)
        del b_x, b_y

        all_pred = []
        all_val = []
        val_losses = []
        dnn.eval()
        for val_step, (val_x, val_y) in enumerate(val_loader):
            val_index = torch.where(val_y[:, -1] == 1)
            val_output = dnn(val_x[val_index].to(device))
            # val_output = dnn(val_x.to(device))
            # v_loss = BCEloss(val_output, val_y[:,:-1].double().to(device))
            v_loss = focal_loss(val_output, val_y[val_index][:, :-1].double().to(device), 0.75, 2)
            val_losses.append(v_loss.item())
            all_pred.extend(val_output.cpu().detach().numpy().tolist())
            all_val.extend(val_y[val_index][:,:-1].cpu().detach().numpy().tolist())
        del val_x, val_y
        # print(all_pred[0])
        # print(all_val[0])

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        all_val = np.array(all_val)
        all_pred = np.array(all_pred)
        calculate_performance(all_val, all_pred,epoch)
        # metrics = get_threshold_metrics(all_val,all_pred)
        # print(metrics)


        early_stopping(val_loss, dnn)

        if i == 1:
            train_list.append([train_loss])
            val_list.append([val_loss])
        else:
            try:
                train_list[epoch] += [train_loss]
                val_list[epoch] += [val_loss]
            except:
                train_list.append([train_loss])
                val_list.append([val_loss])


        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('Epoch: ', epoch, '| train loss: %.8f' % train_loss)
        print('Epoch: ', epoch, '| val loss: %.8f' % val_loss)

    epochs_list = [i for i in range(epoch)]
    train_list = train_list[:epoch]
    val_list = val_list[:epoch]
    plot_learning_curve(train_list, epochs_list,i,'train')
    plot_learning_curve(val_list, epochs_list,i,'val')

    return -early_stopping.best_score,dnn

def main():
    name = 'balance'
    keyword = KEYWORD #multilhead，imbmultilhead
    batch_size = BATCH_SIZE
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'RAS','processed')
    balpath = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'GAE','raw')
    model_path = osp.join(osp.dirname(osp.realpath(__file__)), 'model', 'RAS', f'{keyword}')
    result = osp.join(osp.dirname(osp.realpath(__file__)), 'result', 'RAS', 'multi', f'{keyword}')
    if os.path.exists(model_path) == True:
        pass
    else:
        os.mkdir(model_path)
    if not os.path.exists(result):
        os.makedirs(result)
    #求出每一个标签的占比权重
    balance_y = np.load(balpath + f'/y_test_all.npy')
    balance_sum = np.sum(balance_y[:, :-1], axis=0)
    balance_sum_all = np.sum(balance_sum)
    balance_sum = balance_sum / balance_sum_all

    #每一个标签的指标，求平均
    roc_all = []
    pr_all = []
    acc_all = []
    precision_all = []
    hl_all = []
    recall_all = []
    f1_all = []

    for i in range(1,6):
        train_data = torch.load(osp.join(path, f'train_{keyword}_{i}.pt'),map_location='cpu')
        val_data = torch.load(osp.join(path, f'val_{keyword}_{i}.pt'),map_location='cpu')
        test_data = torch.load(osp.join(path, f'test_{keyword}_{i}.pt'),map_location='cpu')
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        s,dnn = RAStrain(train_loader, val_loader,path,model_path,i)

        test_pred = []
        test_binary = []
        dnn.eval()
        for test_step, (test_x, test_y) in enumerate(test_loader):
            # test_output = dnn(test_x.to(device))
            # test_pred.extend(test_output.cpu().detach().numpy().tolist())
            # test_binary.extend(test_y[:,:-1].cpu().detach().numpy().tolist())
            test_index = torch.where(test_y[:, -1] == 1)
            test_output = dnn(test_x[test_index].to(device))
            test_pred.extend(test_output.cpu().detach().numpy().tolist())
            test_binary.extend(test_y[test_index][:, :-1].cpu().detach().numpy().tolist())
            del test_x, test_y
        test_pred = np.array(test_pred)
        test_binary = np.array(test_binary)

        acc,hl,recall,precision,f1 = multi_calculate_performance(test_binary, test_pred, 'test')

        acc_all.append(acc)
        hl_all.append(hl)
        recall_all.append(recall)
        precision_all.append(precision)
        f1_all.append(f1)

        fold_roc = []
        fold_pr = []
        for j in range(6):
            metrics = get_threshold_metrics(test_binary[:, j], test_pred[:, j])
            print(metrics)
            fold_roc.append(metrics['auroc'])
            fold_pr.append(metrics['aupr'])
        roc_all.append(fold_roc)
        pr_all.append(fold_pr)


    roc_multilabel_mean = np.mean(np.array(roc_all), axis=0)
    pr_multilabel_mean = np.mean(np.array(pr_all), axis=0)
    roc_weight = np.dot(balance_sum, roc_multilabel_mean)
    pr_weight = np.dot(balance_sum, pr_multilabel_mean)

    acc_mean = np.mean(np.array(acc_all))
    hl_mean = np.mean(np.array(hl_all))
    recall_mean = np.mean(np.array(recall_all))
    precision_mean = np.mean(np.array(precision_all))
    f1_mean = np.mean(np.array(f1_all))

    results = f'roc_multilabel_mean:{roc_multilabel_mean},pr_multilabel_mean:{pr_multilabel_mean},' \
              f'acc_mean:{acc_mean},hl_mean:{hl_mean},' \
              f'recall_mean:{recall_mean},precision_mean:{precision_mean},' \
              f'f1_mean:{f1_mean},' \
              f'roc_weight:{roc_weight},pr_weight:{pr_weight},'
    print(results)
    print('------------------------------')
    print(roc_all)
    print(pr_all)
    print(acc_all)
    print(hl_all)
    print(recall_all)
    print(precision_all)

if __name__ == "__main__":
    main()
