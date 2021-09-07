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

import wandb
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

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
        
class MaskedGlobalPooling(nn.Module):
    def __init__(self, pooling='max'):
        super(MaskedGlobalPooling, self).__init__()
        self.pooling = pooling

    def forward(self, x, adj, masks):
        masks = masks.unsqueeze(2).repeat(1, 1, x.size(2))
        if self.pooling == 'max':
            x[masks == 0] = -99999.99999
            x = x.max(1)[0]
        elif self.pooling == 'add':
            x = x.sum(1)
        else:
            print('Not Implemented')

        return x

class BindingAffinityLayer(nn.Module):
    def __init__(self):
        super(BindingAffinityLayer, self).__init__()

        self.fc_gcn = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Dropout(0.2))
            
        self.fc_gin = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(0.2))

        self.affinity_gcn = nn.Linear(512, 1)
        self.affinity_gin = nn.Linear(256, 1)

    def forward(self, com, pro, model='GCN'):
        if model == 'GCN':
            x = torch.cat([com, pro], 1)
            x = self.fc_gcn(x)
            return self.affinity_gcn(x)
        elif model == 'GIN':
            x = torch.cat([com, pro], 1)
            x = self.fc_gin(x)
            return self.affinity_gin(x)
        else:
            print('Not Implemented')

        
        
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.device_type = args.device
        self.dist_option = args.distributed
        self.num_epochs = args.hp_num_epochs

        """numpy helper"""
        self.metric_helper = MetricHelper(args)

        """model structure"""
        self.num_features_xd = 78
        self.num_features_xt = 25
        self.layer = nn.ModuleDict()
        
        self.dim = 32
        
        # GCNConv
        self.layer['GCNConv'] = GraphDenseSequential(
            DenseGCNConv(self.num_features_xd, self.num_features_xd, True, True),
            nn.ReLU(),
            DenseGCNConv(self.num_features_xd, self.num_features_xd * 2, True, True),
            nn.ReLU(),
            DenseGCNConv(self.num_features_xd * 2, self.num_features_xd * 4, True, True),
            nn.ReLU(),
            MaskedGlobalPooling(),
            nn.Linear(self.num_features_xd * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2))
            
        # GINConv    
        nn1 = nn.Sequential(nn.Linear(self.num_features_xd, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        nn2 = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        nn3 = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        nn4 = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        nn5 = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim))
        
        self.GIN1 = DenseGINConv(nn1)
        self.GIN2 = DenseGINConv(nn2)
        self.GIN3 = DenseGINConv(nn3)
        self.GIN4 = DenseGINConv(nn4)
        self.GIN5 = DenseGINConv(nn5)
        
        self.bn1 = torch.nn.BatchNorm1d(self.dim)
        self.bn2 = torch.nn.BatchNorm1d(self.dim)
        self.bn3 = torch.nn.BatchNorm1d(self.dim)
        self.bn4 = torch.nn.BatchNorm1d(self.dim)
        self.bn5 = torch.nn.BatchNorm1d(self.dim)
        
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.GIN = DenseGINConv(nn1)
        self.relu = nn.ReLU()
        self.pool = MaskedGlobalPooling(pooling='add')
        self.fc1_xd = nn.Linear(32, 128)
        
        # Prot_conv        
        self.embedding = nn.Embedding(self.num_features_xt+1, 128)
        self.layer['prot_conv'] = nn.Sequential(
            nn.Conv1d(in_channels=1000, out_channels=32, kernel_size=8))
            
        self.fc1_xt = nn.Linear(32*121, 128)

        self.layer['affinity'] = BindingAffinityLayer()

    def get_objective_loss(self, batch):
        ba_pred, ba_true, _, _, _ = batch
        criterion = nn.MSELoss()

        loss_aff = criterion(ba_pred, ba_true)
        # loss_pwi = torch.zeros(1)

        return loss_aff

    def forward(self, batch, model='GIN'):
        af, aa, am, rf, bav = batch
        if model == 'GIN':
            af = self.relu(self.GIN1(af,aa,am))
            af = self.bn1(af.transpose(1,2))
            af = self.relu(self.GIN2(af.transpose(1,2),aa,am))
            af = self.bn2(af.transpose(1,2))
            af = self.relu(self.GIN3(af.transpose(1,2),aa,am))
            af = self.bn3(af.transpose(1,2))
            af = self.relu(self.GIN4(af.transpose(1,2),aa,am))
            af = self.bn4(af.transpose(1,2))
            af = self.relu(self.GIN5(af.transpose(1,2),aa,am))
            af = self.bn5(af.transpose(1,2))
            af = self.pool(af.transpose(1,2),aa,am)
            af = self.dropout(self.relu(self.fc1_xd(af)))

        elif model == 'GCN':
            af = self.layer['GCNConv'](af, aa, am)
            
        else:
            print('Not Implemented')

        rf = self.embedding(rf.squeeze(-1))
        rf = self.layer['prot_conv'](rf)
        rf = rf.view(-1, 32*121)
        rf = self.fc1_xt(rf)
        affinity_args = (af, rf)
        bap = self.layer['affinity'](*affinity_args)
        # if torch.any(af.isnan()):
        #     import pdb
        #     pdb.set_trace()
        batch = [bap.view(-1), bav.view(-1), None, None, None]

        return batch

    def fit(self, train, valid):
        model = self.to(self.device_type)
        model = nn.DataParallel(model)
        if self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.4)

        for epoch in range(self.num_epochs):
            model.train()
            for idx, batch in enumerate(train):
                optimizer.zero_grad()
                batch = model(batch)
                loss = self.get_objective_loss(batch)
                tloss = loss.item()
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), 5)
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
                    print(f"Epoch: {epoch}, Training Loss: {tloss}, Validation Loss: {vloss}")
                    self.metric_helper.store_batchwise(batch, vloss, 'valid')
                    self.metric_helper.store_epochwise('valid')
                    del batch, loss; torch.cuda.empty_cache()
            
            self.metric_helper.wandb_epochwise('train')
            self.metric_helper.wandb_epochwise('valid')
        
        if  self.dist_option: return model.module
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
