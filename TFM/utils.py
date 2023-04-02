import torch, random, os, math 
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch_geometric.data import DataLoader
from TFM.Dataset import MolNet
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
remover = SaltRemover()
meta = ['He', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sn', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Rb', 'Sr', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Gd', 'Tb', 'Ho', 'W', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac']


def load_data(dataset, batch_size, valid_size, test_size, cpus_per_gpu, task, scaffold=True, seed=426):
    data = MolNet(root='./dataset', dataset=dataset)

    #trainset, validset, testset = randomscaffold_split(data, valid_size, test_size, scaffold=scaffold, seed=seed)
    cont = True
    while cont:
        trainset, validset, testset = randomscaffold_split(data, valid_size, test_size, scaffold=scaffold, seed=seed)
        if task == 'clas':
            vy = [d.y for d in validset]; ty = [d.y for d in testset]
            vy = torch.cat(vy, 0); ty = torch.cat(ty, 0)
            if torch.any(torch.mean(vy, 0) == 1) or torch.any(torch.mean(vy, 0) == 0) or torch.any(torch.mean(ty, 0) == 1) or torch.any(torch.mean(ty, 0) == 0):
                cont = True
                if seed is not None:
                    seed += 10
            else:
                cont = False
        else:
            cont = False
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=cpus_per_gpu, drop_last=False)
    return train_loader, valid_loader, test_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class metrics_c(nn.Module):
    def __init__(self, acc_f, pre_f, rec_f, f1_f, auc_f):
        super(metrics_c, self).__init__()
        self.acc_f = acc_f
        self.pre_f = pre_f
        self.rec_f = rec_f
        self.f1_f = f1_f
        self.auc_f = auc_f

    def forward(self, out, prob, tar):
        if len(out.shape) > 1:
            acc, f1, pre, rec, auc = [], [], [], [], []
            
            for i in range(out.shape[-1]):
                acc_, f1_, pre_, rec_, auc_ = 0, 0, 0, 0, 0
                acc_ = self.acc_f(tar[:, i], out[:, i])
                f1_ = self.f1_f(tar[:, i], out[:, i])
                pre_ = self.pre_f(tar[:, i], out[:, i])
                rec_ = self.rec_f(tar[:, i], out[:, i])
                auc_ = self.auc_f(tar[:, i], prob[:, i])
                
                acc.append(acc_); f1.append(f1_); pre.append(pre_); rec.append(rec_); auc.append(auc_)
            return np.mean(acc), np.mean(f1), np.mean(pre), np.mean(rec), np.nanmean(auc)
        else:
            acc = self.acc_f(tar, out)
            f1 = self.f1_f(tar, out)
            pre = self.pre_f(tar, out)
            rec = self.rec_f(tar, out)
            auc = self.auc_f(tar, prob)
        return acc, f1, pre, rec, auc


class metrics_r(nn.Module):
    def __init__(self, mae_f, rmse_f, r2_f):
        super(metrics_r, self).__init__()
        self.mae_f = mae_f
        self.rmse_f = rmse_f
        self.r2_f = r2_f

    def forward(self, out, tar):
        mae, rmse, r2 = 0, 0, 0
        if self.mae_f is not None:
            mae = self.mae_f(tar, out)

        if self.rmse_f is not None:
            rmse = self.rmse_f(tar, out, squared=False)

        if self.r2_f is not None:
            r2 = self.r2_f(tar, out)

        return mae, rmse, r2, None, None

def log_cosh_loss(y_pred, y_true):
    def _log_cosh(x):
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return log_cosh_loss(y_pred, y_true)
    

def create_ffn(task, tasks, output_dim, dropout):
    if task == 'clas':
        act = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim*2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, tasks),
            nn.Sigmoid())
    elif task == 'reg':
        act = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim*2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, tasks))
    else:
        raise NameError('task must be reg or clas!')
    return act


def get_attn_pad_mask(mask): 
    batch_size, len_q = mask.size(0), mask.size(1)
    a = mask.unsqueeze(1).expand(batch_size, len_q, len_q)
    pad_attn_mask = a * a.transpose(-1, -2) 
    return pad_attn_mask.data.eq(0) 


def randomscaffold_split(data, validrate, testrate, scaffold=True, seed=426):
    trainrate = 1 - validrate - testrate
    assert trainrate > 0.4
    lenth = len(data)
    g1 = int(lenth*trainrate)
    g2 = int(lenth*(trainrate+validrate))
    
    if not scaffold:
        rng = np.random.RandomState(seed)
        random_num = rng.permutation(range(0, lenth))
        data = data[random_num]
        return data[:g1], data[g1:g2], data[g2:]

    else:
        train_inds, valid_inds, test_inds = [], [], []
        scaffolds = {}
        for ind, dat in enumerate(data):
            mol = Chem.MolFromSmiles(dat.smi)
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=True)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)
        
        rng = np.random.RandomState(seed)
        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_valid = round(validrate * len(data))
        n_total_test = round(testrate * len(data))
        for scaffold_set in scaffold_sets:
            if len(valid_inds) + len(scaffold_set) <= n_total_valid:
                valid_inds.extend(scaffold_set)
            elif len(test_inds) + len(scaffold_set) <= n_total_test:
                test_inds.extend(scaffold_set)
            else:
                train_inds.extend(scaffold_set)

        return data[train_inds], data[valid_inds], data[test_inds]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
    formatter = logging.Formatter("[%(asctime)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger