import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
from model_utils import MetricHelper
from env_config import *
from torch_geometric.nn.dense import *
from torch_geometric.nn.glob.glob import global_max_pool
from torch_geometric.nn.norm.layer_norm import LayerNorm
import os
import numpy as np

import wandb
import torch.nn.functional as F
from torch.autograd import Variable

class GraphDenseSequential(nn.Sequential):
    def __init__(self, *args):
        super(GraphDenseSequential, self).__init__(*args)

    def forward(self, X, adj, mask):
        for module in self._modules.values():
            try:
                X = module(X, adj, mask)
            except BaseException:
                X = module(X)

        return X


class MaskedLayerNorm(nn.Module):
    def __init__(self, out_fea, affine=True):
        super(MaskedLayerNorm, self).__init__()
        self.norm_gamma = nn.Parameter(
            torch.ones(out_fea), requires_grad=affine)
        self.norm_beta = nn.Parameter(
            torch.zeros(out_fea), requires_grad=affine)
        self.epsilon = 1e-5

    def forward(self, x, adj, masks):
        E_x = (x.sum(1) / masks.sum(1).view(-1, 1))
        E_x2 = (x.pow(2).sum(1) / masks.sum(1).view(-1, 1))
        Var_x = E_x2 - E_x.pow(2)
        E_x, Var_x = E_x.unsqueeze(1), Var_x.unsqueeze(1)
        x = (x - E_x / (Var_x + self.epsilon).pow(0.5))
        x = (x * self.norm_gamma.view(1, 1, -1) + self.norm_beta.view(1, 1, -1))

        return x * masks.unsqueeze(2)


class MaskedGlobalPooling(nn.Module):
    def __init__(self, pooling='max'):
        super(MaskedGlobalPooling, self).__init__()
        self.pooling = pooling

    def forward(self, x, adj, masks):
        masks = masks.unsqueeze(2).repeat(1, 1, x.size(2))
        if self.pooling == 'max':
            x[masks == 0] = -99999.99999
            x = x.max(1)[0]

        return x


class BindingAffinityLayer(nn.Module):
    def __init__(self):
        super(BindingAffinityLayer, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2))
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2))

        self.affinity = nn.Linear(512, 1)

    def forward(self, com, pro):
        x = torch.cat([com, pro], 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return self.affinity(x)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.device_type = args.device
        self.dist_option = args.distributed
        self.num_epochs = args.hp_num_epochs

        """numpy helper"""
        self.metric_helper = MetricHelper(args)

        """model structure"""
        self.f_prot = 54
        self.f_mol = 78
        self.layer = nn.ModuleDict()

        self.layer['comp_conv@shared'] = GraphDenseSequential(
            DenseGCNConv(self.f_mol, self.f_mol, True, True),
            nn.ReLU(),
            DenseGCNConv(self.f_mol, self.f_mol * 2, True, True),
            nn.ReLU(),
            DenseGCNConv(self.f_mol * 2, self.f_mol * 4, True, True),
            nn.ReLU(),
            nn.Linear(self.f_mol * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            MaskedGlobalPooling())

        self.layer['prot_conv@shared'] = GraphDenseSequential(
            DenseGCNConv(self.f_prot, self.f_prot, True, True),
            nn.ReLU(),
            DenseGCNConv(self.f_prot, self.f_prot * 2, True, True),
            nn.ReLU(),
            DenseGCNConv(self.f_prot * 2, self.f_prot * 4, True, True),
            nn.ReLU(),
            nn.Linear(self.f_prot * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            MaskedGlobalPooling())

        self.layer['affinity'] = BindingAffinityLayer()

    def get_objective_loss(self, batch):
        ba_pred, ba_true, _, _, _ = batch
        criterion = nn.MSELoss()

        loss_aff = criterion(ba_pred, ba_true)
        # loss_pwi = torch.zeros(1)

        return loss_aff

    def forward(self, batch):
        af, aa, am, rf, rr, rm, _, _, bav = batch
        af1 = af
        af = self.layer['comp_conv@shared'](af, aa, am)
        rf = self.layer['prot_conv@shared'](rf, rr, rm)
        affinity_args = (af, rf)
        bap = self.layer['affinity'](*affinity_args)
        # if torch.any(af.isnan()):
        #     import pdb
        #     pdb.set_trace()
        batch = [bap.view(-1), bav.view(-1), None, None, None]

        return batch

    def fit(self, train, valid):
        model = self.to(self.device_type)
        if self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)

        for epoch in range(self.num_epochs):
            model.train()
            for idx, batch in enumerate(train):
                # import pickle; pickle.dump(batch, open('./saved/d.pkl', 'wb'))
                # if None in batch:
                optimizer.zero_grad()
                batch = model(batch)
                loss = self.get_objective_loss(batch)
                loss.backward()
                tloss = loss.item()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                self.metric_helper.store_batchwise(batch, tloss, 'train')
                del batch, loss; torch.cuda.empty_cache()
                self.metric_helper.store_epochwise('train')

            if valid:
                model.eval()
                with torch.no_grad():
                    batch = next(iter(valid))
                    batch = model(batch)
                    loss = self.get_objective_loss(batch)
                    vloss = loss.item()
                    self.metric_helper.store_batchwise(batch, loss.item(), 'valid')
                    self.metric_helper.store_epochwise('valid')
                    print(f"Epoch: {epoch}, Training Loss: {tloss}, Validation Loss: {vloss}")
                    del batch, loss; torch.cuda.empty_cache()

            self.metric_helper.wandb_epochwise('train')
            self.metric_helper.wandb_epochwise('valid')
        
        if self.dist_option: return model.module
        else: return model
        
    def predict_eval(self, data, label):
        self.metric_helper.add_label(label)
        model = self.to(self.device_type)
        if self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(data):
                batch = model(batch)
                loss = self.get_objective_loss(batch)
                self.metric_helper.store_batchwise(batch, loss, label)
                del batch, loss 
                torch.cuda.empty_cache()
        self.metric_helper.store_epochwise(label)
        self.metric_helper.wandb_epochwise(label)

    def predict(self, data, label):
        predictions = []
        self.metric_helper.add_label(label)
        model = self.to(self.device_type)
        if  self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(data):
                batch = model(batch)
                predictions.append(batch[0].detach().cpu().numpy().reshape(-1))

        return np.concatenate(predictions)

