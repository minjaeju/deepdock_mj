import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim

# import metrics

# from dataset import PT_FEATURE_SIZE
from model_utils import MetricHelper

CHAR_SMI_SET_LEN = 64
PT_FEATURE_SIZE = 33

class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DilatedParllelResidualBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
#             print(f'{nIn}-{nOut}: add=False')
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device_type = args.device
        self.dist_option = args.distributed
        self.num_epochs = args.hp_num_epochs

        """numpy helper"""
        self.metric_helper = MetricHelper(args)

        smi_embed_size = 128
        seq_embed_size = 128
        
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128
        
        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)  # (N, *, H_{in}) -> (N, *, H_{out})

        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))  # (N, oc)
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)
        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3))  # (N,C,L)
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)  # (N,oc)

        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)  # (N,128)
        
        self.cat_dropout = nn.Dropout(0.2)
        
        self.classifier = nn.Sequential(
            nn.Linear(seq_oc+pkt_oc+smi_oc, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64,1),
            nn.PReLU())
        
    def get_objective_loss(self, batch):
        ba_pred, ba_true, _, _, _ = batch
        criterion = nn.MSELoss()

        loss_aff = criterion(ba_pred, ba_true)
        # loss_pwi = torch.zeros(1)

        return loss_aff

    def forward(self, batch):
        af, rf, pf, bav = batch
        # assert seq.shape == (N,L,43)
        seq_embed = self.seq_embed(rf)  # (N,L,32)
        seq_embed = torch.transpose(seq_embed, 1, 2)  # (N,32,L)
        seq_conv = self.conv_seq(seq_embed)  # (N,128)

        # assert pkt.shape == (N,L,43)
        pkt_embed = self.seq_embed(pf)  # (N,L,32)
        pkt_embed = torch.transpose(pkt_embed, 1, 2)
        pkt_conv = self.conv_pkt(pkt_embed)  # (N,128)

        # assert smi.shape == (N, L)
        smi_embed = self.smi_embed(af)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)
        smi_conv = self.conv_smi(smi_embed)  # (N,128)

        try:
            cat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=1)  # (N,128*3)
        except:
            cat = torch.cat([seq_conv, pkt_conv, smi_conv], dim=0)  # (N,128*3)
        cat = self.cat_dropout(cat)
        
        bap = self.classifier(cat)
        batch = [bap.view(-1), bav.view(-1), None, None, None]
        return batch
    
    def fit(self, train, valid):
        model = self.to(self.device_type)
        model = nn.DataParallel(model)
        if self.dist_option:
            model = nn.parallel.DistributedDataParallel(model)

        optimizer = optim.AdamW(model.parameters())
        scheduler = optim.lr_scheduler.OneCycleLR(
                        optimizer, max_lr=5e-3, epochs=20,
                        steps_per_epoch=len(train))

        for epoch in range(self.num_epochs):
            model.train()
            for idx, batch in enumerate(tqdm(train)):
                optimizer.zero_grad()
                batch = model(batch)
                loss = self.get_objective_loss(batch)
                loss.backward()
                optimizer.step()
                self.metric_helper.store_batchwise(batch, loss.item(), 'train')
                del batch, loss
                torch.cuda.empty_cache()
            self.metric_helper.store_epochwise('train')

            if valid:
                model.eval()
                with torch.no_grad():
                    batch = next(iter(valid))
                    batch = model(batch)
                    loss = self.get_objective_loss(batch)
                    self.metric_helper.store_batchwise(batch, loss.item(), 'valid')
                    self.metric_helper.store_epochwise('valid')
            
            self.metric_helper.wandb_epochwise('train')
            self.metric_helper.wandb_epochwise('valid')
        
        if self.dist_option: return model.module
        else: return model
        
    # def predict(self, data, label):
    #     self.metric_helper.add_label(label)
    #     model = self.to(self.args.device)
    #     if self.args.distributed:
    #         model = nn.parallel.DistributedDataParallel(model)
    #     model.eval()
    #     for idx, batch in enumerate(tqdm(data)):
    #         batch = model(batch)
    #         loss = self.get_objective_loss(batch)
    #         self.metric_helper.store_batchwise(batch, loss, label)
    #     self.metric_helper.store_epochwise(label)
    #     self.metric_helper.wandb_epochwise(label)

    def predict_eval(self, data, label):
        self.metric_helper.add_label(label)
        model = self.to(self.device_type)
        model = nn.DataParallel(model)
        if self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(data)):
                batch = model(batch)
                loss = self.get_objective_loss(batch)
                self.metric_helper.store_batchwise(batch, loss.item(), label)
        self.metric_helper.store_epochwise(label)
        self.metric_helper.wandb_epochwise(label)
        del batch, loss 
        torch.cuda.empty_cache()

    def predict(self, data, label):
        import pdb
        predictions = []
        self.metric_helper.add_label(label)
        model = self.to(self.device_type)
        model = nn.DataParallel(model)
        if  self.dist_option:
            model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
            # model = apex.parallel.DistributedDataParallel(model)
        model.eval()
        pdb.set_trace()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(data)):
                batch = model(batch)
                predictions.append(batch[0].detach().cpu().numpy().reshape(-1))
        return np.concatenate(predictions)


def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat = model(*x)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }

    return evaluation
