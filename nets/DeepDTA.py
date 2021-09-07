import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm
from model_utils import MetricHelper
from env_config import *
import numpy as np

import wandb
import torch.nn.functional as F
from torch.autograd import Variable


class BindingAffinityLayer(nn.Module):
    def __init__(self):
        super(BindingAffinityLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1))

    def forward(self, x):
        return self.fc(x)

class Net(nn.Module):
    def __init__(self, args=0):
        super(Net, self).__init__()
        self.device_type = args.device
        self.dist_option = args.distributed
        self.num_epochs = args.hp_num_epochs

        self.metric_helper = MetricHelper(args)
        self.layer = nn.ModuleDict()
        self.layer['atom_conv'] = nn.Linear(62, 128)
        self.layer['comp_conv'] = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=8, stride=1),
            nn.ReLU())
        self.layer['comp_pool'] = nn.MaxPool1d(100-18+3)

        self.layer['resi_conv'] = nn.Linear(21, 128)
        self.layer['prot_conv'] = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=96, kernel_size=8, stride=1),
            nn.ReLU())
        self.layer['prot_pool'] = nn.MaxPool1d(1000-18+3)

        self.layer['affinity'] = BindingAffinityLayer()

    def get_objective_loss(self, batch):
        ba_pred, ba_true, _, _, _ = batch
        criterion = nn.MSELoss()
        loss_aff = criterion(ba_pred, ba_true)
        # loss_pwi = torch.zeros(1)

        return loss_aff

    def forward(self, batch):
        x1 = self.layer['atom_conv'](batch[0])
        x1 = self.layer['comp_conv'](x1.transpose(1,2))
        x1 = self.layer['comp_pool'](x1).squeeze()
        
        x2 = self.layer['resi_conv'](batch[1])
        x2 = self.layer['prot_conv'](x2.transpose(1,2))
        x2 = self.layer['prot_pool'](x2).squeeze()

        x1, x2 = x1.view(-1,96), x2.view(-1,96)
        bap = self.layer['affinity'](torch.cat([x1,x2],1))
        batch = [bap.view(-1), batch[2].view(-1), None, None, None]
        return batch

    def fit(self, train, valid=None):
        model = self.to(self.device_type)
        model = nn.DataParallel(model)
        if self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0.001)

        for epoch in range(self.num_epochs):
            model.train()
            for idx, batch in enumerate(train):
                optimizer.zero_grad()
                batch = model(batch)  # must retrieve intermediary tensors?
                loss = self.get_objective_loss(batch)
                tloss = loss.item()
                loss.backward()
                optimizer.step()
                self.metric_helper.store_batchwise(batch, tloss, 'train')
            self.metric_helper.store_epochwise('train')

            if valid:
                model.eval()
                with torch.no_grad():
                    batch = next(iter(valid))
                    batch = model(batch)
                    loss = self.get_objective_loss(batch)
                    vloss = loss.item()
                    print(f"Epoch: {epoch}, Training Loss: {tloss}, Validation Loss: {vloss}")
                    self.metric_helper.store_batchwise(batch, vloss, 'valid')
                    self.metric_helper.store_epochwise('valid')

            self.metric_helper.wandb_epochwise('train')
            self.metric_helper.wandb_epochwise('valid')

        if self.dist_option: return model.module
        else: return model

    def predict_eval(self, data, label):
        self.metric_helper.add_label(label)
        model = self.to(self.device_type)
        model = nn.DataParallel(model)
        if self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(data):
                batch = model(batch)
                loss = self.get_objective_loss(batch)
                self.metric_helper.store_batchwise(batch, loss.item(), label)
        self.metric_helper.store_epochwise(label)
        self.metric_helper.wandb_epochwise(label)
        del batch, loss 
        torch.cuda.empty_cache()

    def predict(self, data, label):
        predictions = []
        self.metric_helper.add_label(label)
        model = self.to(self.device_type)
        model = nn.DataParallel(model)
        if  self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(data):
                batch = model(batch)
                predictions.append(batch[0].detach().cpu().numpy().reshape(-1))

        return np.concatenate(predictions)
