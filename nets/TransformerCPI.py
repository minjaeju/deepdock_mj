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
import math

class SelfAttention(nn.Module):
    def __init__(self, hid_dim=64, n_heads=8, dropout=0.1):
        super(SelfAttention,self).__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to('cuda')

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]
        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        return x
        
class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim=100, hid_dim=64, n_layers=3, kernel_size=5, dropout=0.1):
        super(Encoder,self).__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to('cuda')
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(100,64)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]
            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]
            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]
            #set conv_input to conved for next loop iteration
            conv_input = conved
        conved = conved.permute(0,2,1)
        # conved = [batch size,protein len,hid dim]
        return conved

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim=64, pf_dim=256, dropout=0.1):
        super(PositionwiseFeedforward,self).__init__()
        
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]
        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        x = self.do(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x
        
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim=64, n_heads=8, pf_dim=256, dropout=0.1):
        super(DecoderLayer,self).__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg
        
class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim=34, hid_dim=64, n_layers=3, n_heads=8, pf_dim=256, dropout=0.1):
        super(Decoder,self).__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = DecoderLayer()
        self.self_attention = SelfAttention()
        self.positionwise_feedforward = PositionwiseFeedforward()
        self.dropout = dropout
        self.sa = SelfAttention()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 1)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)
        # trg = [batch size, compound len, hid dim]
        for layer in self.layers:
            trg = layer(trg, src)
        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg,dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm,dim=1)
        # norm = [batch size,compound len]
        trg = torch.squeeze(trg,dim=0)
        norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((trg.shape[0],self.hid_dim)).to('cuda')
        for i in range(norm.shape[0]):
            v = trg[i,]
            v = torch.matmul(norm[i],v)
            sum[i] += v
        #sum = sum.unsqueeze(dim=0)
        # trg = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label
        
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.device_type = args.device
        self.dist_option = args.distributed
        self.num_epochs = args.hp_num_epochs

        """numpy helper"""
        self.metric_helper = MetricHelper(args)

        """model structure"""
        self.layer = nn.ModuleDict()
        
        self.weight = nn.Parameter(torch.FloatTensor(34, 34))
        self.init_weight()
        
        self.Encoder = Encoder()
        self.Decoder = Decoder()
    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input =[num_node, atom_dim]
        # adj = [num_node, num_node]
        support = torch.matmul(input, self.weight)
        # support =[num_node,atom_dim]
        output = torch.matmul(adj, support)
        # output = [num_node,atom_dim]
        return output
        
    def get_objective_loss(self, batch):
        ba_pred, ba_true, _, _, _ = batch
        criterion = nn.MSELoss()

        loss_aff = criterion(ba_pred, ba_true)
        # loss_pwi = torch.zeros(1)

        return loss_aff

    def forward(self, batch):
        af, aa, am, rf, bav = batch
        
        af = self.gcn(af,aa)
        rf = self.Encoder(rf)
        out = self.Decoder(af, rf) 
        batch = [out.view(-1), bav.view(-1), None, None, None]

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
                    self.metric_helper.store_batchwise(batch, vloss, 'valid')
                    self.metric_helper.store_epochwise('valid')
                    print(f"Epoch: {epoch}, Training Loss: {tloss}, Validation Loss: {vloss}")
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
