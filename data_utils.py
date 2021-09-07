import pdb
import traceback
from timeit import default_timer as timer
from collections import namedtuple
import pickle
from tqdm import tqdm
from scipy.linalg import block_diag
from sys import getsizeof
import os
import h5py
import pandas as pd
import numpy as np
from env_config import *
import torch
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import Data as GData 
# from torch_geometric.data import Dataset as GDataset
# from torch_geometric.data import DataLoader as GDataLoader
from scipy.sparse.csgraph import laplacian as lap
from scipy.sparse import coo_matrix
from scipy.sparse import hstack as sparse_hstack
import itertools
# from torch_geometric import dense_to_sparse

from word2vec import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec

from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
    'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def onek_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(
            'input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def stack_and_pad(arr_list, max_length=None):
    M = max([x.shape[0] for x in arr_list]) if not max_length else max_length
    N = max([x.shape[1] for x in arr_list])
    T = np.zeros((len(arr_list), M, N))
    t = np.zeros((len(arr_list), M))
    s = np.zeros((len(arr_list), M, N))
    for i, arr in enumerate(arr_list):
        # sum of 16 interaction type, one is enough
        if len(arr.shape) > 2:
            arr = (arr.sum(axis=2) > 0.0).astype(float)
        T[i, 0:arr.shape[0], 0:arr.shape[1]] = arr
        t[i, 0:arr.shape[0]] = 1 if arr.sum() != 0.0 else 0
        s[i, 0:arr.shape[0], 0:arr.shape[1]] = 1 if arr.sum() != 0.0 else 0
    return T, t, s

def stack_and_pad_2d(arr_list, block='lower_left'):
    max0 = max([a.shape[0] for a in arr_list])
    max1 = max([a.shape[1] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            # sum of 16 interaction type, one is enough
            if len(arr_list[i].shape) > 2:
                arr_list[i] = (arr_list[i].sum(axis=2) == True).astype(float)
            final_result[i, :shape[0], :shape[1]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d


def stack_and_pad_3d(arr_list, block='lower_left'):
    max0 = max([a.shape[0] for a in arr_list])
    max1 = max([a.shape[1] for a in arr_list])
    max2 = max([a.shape[2] for a in arr_list])
    list_shapes = [a.shape for a in arr_list]

    final_result = np.zeros((len(arr_list), max0, max1, max2))
    final_masks_2d = np.zeros((len(arr_list), max0))
    final_masks_3d = np.zeros((len(arr_list), max0, max1))
    final_masks_4d = np.zeros((len(arr_list), max0, max1, max2))

    if block == 'upper_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], :shape[1], :shape[2]] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], :shape[1]] = 1
            final_masks_4d[i, :shape[0], :shape[1], :] = 1
    elif block == 'lower_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, max1-shape[1]:] = 1
            final_masks_4d[i, max0-shape[0]:, max1-shape[1]:, :] = 1
    elif block == 'lower_left':
        for i, shape in enumerate(list_shapes):
            final_result[i, max0-shape[0]:, :shape[1]] = arr_list[i]
            final_masks_2d[i, max0-shape[0]:] = 1
            final_masks_3d[i, max0-shape[0]:, :shape[1]] = 1
            final_masks_4d[i, max0-shape[0]:, :shape[1], :] = 1
    elif block == 'upper_right':
        for i, shape in enumerate(list_shapes):
            final_result[i, :shape[0], max1-shape[1]:] = arr_list[i]
            final_masks_2d[i, :shape[0]] = 1
            final_masks_3d[i, :shape[0], max1-shape[1]:] = 1
            final_masks_4d[i, :shape[0], max1-shape[1]:, :] = 1
    else:
        raise

    return final_result, final_masks_2d, final_masks_3d, final_masks_4d


def ds_normalize(input_array):
    # Doubly Stochastic Normalization of Edges from CVPR 2019 Paper
    assert len(input_array.shape) == 3
    input_array = input_array / np.expand_dims(input_array.sum(1)+1e-8, axis=1)
    output_array = np.einsum('ijb,jkb->ikb', input_array,
                             input_array.transpose(1, 0, 2))
    output_array = output_array / (output_array.sum(0)+1e-8)

    return output_array

def add_index(input_array, ebd_size):
    # batch_size, n_vertex, n_nbs = np.shape(input_array)
    # temp = range(0, (ebd_size)*batch_size, ebd_size)
    add_idx, temp_arrays = 0, []
    for i in range(input_array.shape[0]):
        temp_array = input_array[i,:,:]
        masking_indices = temp_array.sum(1).nonzero()
        temp_array += add_idx
        temp_arrays.append(temp_array)
        add_idx = masking_indices[0].max()+1
    new_array = np.concatenate(temp_arrays, 0)

    return new_array.reshape(-1)

    # add_idx = np.array(temp)
    # add_idx = np.tile(add_idx, n_nbs*n_vertex)
    # add_idx = np.transpose(add_idx.reshape(-1, batch_size))
    # add_idx = add_idx.reshape(-1)
    # new_array = input_array.reshape(-1)+add_idx

    # return new_array

def combine_edge_indices(list_graphs, list_masks1, list_masks2):
    add_idx, list_graphs_new = 0, []
    for graph, mask in zip(list_graphs, list_masks1):
        graph = graph + add_idx
        add_idx += mask.shape[0]
        list_graphs_new.append(graph)

    return np.hstack(list_graphs_new), np.vstack(list_masks1), np.vstack(list_masks2)

def collate_deepdta(batch):
    tensor_list = []
    list_atomwise_features = [x[0] for x in batch]
    list_resiwise_features = [x[1] for x in batch]
    list_ba_values = [x[2] for x in batch]

    x, _, _ = stack_and_pad(list_atomwise_features, 100)
    tensor_list.append(torch.cuda.FloatTensor(x))
    x, _, _ = stack_and_pad(list_resiwise_features, 1000)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(list_ba_values).view(-1, 1))
    return tensor_list
    
def collate_graphdta(batch):
    tensor_list = []
    list_atomwise_features = [x[0] for x in batch]
    list_atom_adjs = [x[1] for x in batch]
    list_resiwise_features = [x[2] for x in batch]
    list_ba_values = [x[3] for x in batch]

    x, _, _ = stack_and_pad(list_atomwise_features)
    tensor_list.append(torch.cuda.FloatTensor(x))
    x, m, _ = stack_and_pad(list_atom_adjs)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.LongTensor(m))
    x, _, _ = stack_and_pad(list_resiwise_features, 1000)
    tensor_list.append(torch.cuda.LongTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(list_ba_values).view(-1, 1))
    return tensor_list

def collate_deepdtaf(batch):
    tensor_list = []
    list_atomwise_features = [x[0] for x in batch]
    list_resiwise_features = [x[1] for x in batch]
    list_pocketwise_features = [x[2] for x in batch]
    list_ba_values = [x[3] for x in batch]

    tensor_list.append(torch.cuda.LongTensor(list_atomwise_features))
    tensor_list.append(torch.cuda.FloatTensor(list_resiwise_features))
    tensor_list.append(torch.cuda.FloatTensor(list_pocketwise_features))
    tensor_list.append(torch.cuda.FloatTensor(list_ba_values).view(-1, 1))

    return tensor_list
    
def collate_transformercpi(batch):
    tensor_list = []
    list_atomwise_features = [x[0] for x in batch]
    list_atom_adjs = [x[1] for x in batch]
    list_resiwise_features = [x[2] for x in batch]
    list_ba_values = [x[3] for x in batch]

    x, _, _ = stack_and_pad(list_atomwise_features)
    tensor_list.append(torch.cuda.FloatTensor(x))
    x, m, _ = stack_and_pad(list_atom_adjs)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.LongTensor(m))
    x, _, _ = stack_and_pad(list_resiwise_features, 1000)
    tensor_list.append(torch.cuda.FloatTensor(x))
    tensor_list.append(torch.cuda.FloatTensor(list_ba_values).view(-1, 1))
    return tensor_list

def collate_dgraph(batch):
    tensor_list = []
    if batch != None:
        list_atomwise_features = [x[0] for x in batch]
        list_atom_adjs = [x[1] for x in batch]
        list_resiwise_features = [x[2] for x in batch]
        list_resi_adjs = [(x[3].sum(2) > 0.).astype(np.int_) for x in batch]
        # list_atomresi_adjs = [(x[4].sum(2)>0.).astype(np.int_) for x in batch]
        list_ba_values = [x[5] for x in batch]

        x, _, _ = stack_and_pad(list_atomwise_features)
        tensor_list.append(torch.cuda.FloatTensor(x))
        x, m, _ = stack_and_pad(list_atom_adjs)
        tensor_list.append(torch.cuda.FloatTensor(x))
        tensor_list.append(torch.cuda.LongTensor(m))
        x, _, _ = stack_and_pad(list_resiwise_features)
        tensor_list.append(torch.cuda.FloatTensor(x))
        x, m, _ = stack_and_pad(list_resi_adjs)
        tensor_list.append(torch.cuda.FloatTensor(x))
        tensor_list.append(torch.cuda.LongTensor(m))
        # x, _, m = stack_and_pad(list_atomresi_adjs)
        tensor_list.append(torch.cuda.FloatTensor(x))
        tensor_list.append(torch.cuda.LongTensor(m))
        tensor_list.append(torch.cuda.FloatTensor(list_ba_values).view(-1, 1))
        return tensor_list
    else: return [None, None, None, None, None, None, None, None, None]

def load_collate_fn(args):
    if args.pred_model == 'dgraphdta': return collate_dgraph
    elif args.pred_model == 'deepdtaf': return collate_deepdtaf
    elif args.pred_model == 'deepdta': return collate_deepdta
    elif args.pred_model == 'graphdta': return collate_graphdta
    elif args.pred_model == 'transformercpi': return collate_transformercpi
    else: raise

def load_dataset(args):
    if args.pred_model == 'dgraphdta': return DgraphDataset(args)
    elif args.pred_model == 'deepdtaf': return DeepDTAFDataset(args)
    elif args.pred_model == 'deepdta': return DeepDTADataset(args)
    elif args.pred_model == 'graphdta' : return GraphDTADataset(args)
    elif args.pred_model == 'transformercpi': return TransformerCPIDataset(args)
    else: raise

def check_exists(path):    
    return True if os.path.isfile(path) and os.path.getsize(path) > 0 else False

class DtiDataset(Dataset):
    def __init__(self, args):
        # self.data_path = os.path.join(f'{ROOT_PATH}', f'dti_dataset_{args.dataset_version}')
        self.data_path = f'{ROOT_PATH}dataset_{args.dataset_version}/'
        cdfs, ldfs, pdfs = [], [], []
        for dataset in args.dataset_subsets.split('+'):
            cdfs.append(pd.read_csv(f'{self.data_path}complex_metadata_{dataset}.csv',index_col='complex_id'))
            ldfs.append(pd.read_csv(f'{self.data_path}ligand_metadata_{dataset}.csv',index_col='ligand_id'))
            pdfs.append(pd.read_csv(f'{self.data_path}protein_metadata_{dataset}.csv',index_col='protein_id'))
        self.cdf, self.ldf, self.pdf = pd.concat(cdfs), pd.concat(ldfs), pd.concat(pdfs)
        self.dataset, self.idtuples, self.clu_indices, self.failed_list = [], [], [], []
        self.clu = None
        self.cache_path = f'./saved/dataset_cache/'
        # self.cache_path = f'{SAVE_PATH}dataset_cache/'
        if not os.path.exists( self.cache_path): os.makedirs(self.cache_path)
        if args.load_n_test is None:
            self.cdf = self.cdf[self.cdf['ba_measure'] == args.ba_measure]
            self.cdf.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.cdf.dropna(subset=['ba_value'], axis=0, inplace=True)

    def __len__(self):
        return len(self.dataset)
        # print(len(failed_list))

    def __getitem__(self, idx):
        return self.get(idx)

class DgraphDataset(DtiDataset):
    def __init__(self, args):
        super().__init__(args)

        def atom_features(atom):
            return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST_Dgraph)
                          + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                          + onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                          + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                          + [atom.GetIsAromatic()], dtype=np.float32)

        def resi_features(resi):
            res_property1=[1 if resi in pro_res_aliphatic_table else 0,
                           1 if resi in pro_res_aromatic_table else 0,
                           1 if resi in pro_res_polar_neutral_table else 0,
                           1 if resi in pro_res_acidic_charged_table else 0,
                           1 if resi in pro_res_basic_charged_table else 0]

            res_property2=[res_weight_table[resi], res_pka_table[resi], res_pkb_table[resi], res_pkx_table[resi], res_pl_table[resi], res_hydrophobic_ph2_table[resi], res_hydrophobic_ph7_table[resi]]
            return np.array(onek_encoding_unk(resi, RESI_LIST_Dgraph)+res_property1+res_property2)

        index = self.cdf.index
        if args.prelim_mode: index=self.cdf.index[:args.prelim_index]
        DATA_CACHE = self.cache_path+f'dgraphdta_{args.dataset_subsets}.dat'
        if check_exists(DATA_CACHE):
            self.dataset, self.idtuples, self.failed_list = pickle.load(open(DATA_CACHE,'rb'))
            print('Loaded Dataset: ', DATA_CACHE)
            # will include cluster indices later...
        else:
            for idx, complex_idx in enumerate(tqdm(index)):
                try:
                    # Ligand Featurization
                    atomwise_features=[]
                    ligand_idx=self.cdf.loc[complex_idx, 'ligand_id']
                    mol = Chem.MolFromSmiles(self.ldf.loc[ligand_idx, 'smiles'])
                    for atom in mol.GetAtoms():
                        try: atomwise_features.append(atom_features(atom).reshape(1, -1))
                        except: atomwise_features.append(np.zeros(78).reshape(1, -1))
                    atomwise_features=np.vstack(atomwise_features)

                    # Protein Featurization
                    resiwise_features = []
                    protein_idx = self.cdf.loc[complex_idx, 'protein_id']
                    if self.pdf.loc[protein_idx, 'fasta_length'] > 1000:
                        raise FastaLengthException(self.pdf.loc[protein_idx, 'fasta_length'])
                    if not check_exists(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.pssm.npy'):
                        raise NoProteinFeaturesException(protein_idx)
                    fasta = self.pdf.loc[protein_idx, 'fasta']
                    for resi in fasta:
                        resiwise_features.append(resi_features(resi).reshape(1, -1))
                    resiwise_features=np.vstack(resiwise_features)

                    # Graph-based Data Arrays
                    atomatom_graph = Chem.rdmolops.GetAdjacencyMatrix(mol)
                    if not check_exists(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.arpeggio.npy'):
                        if not check_exists(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.pc4cmap.npy'):
                            raise NoProteinGraphException(protein_idx)

                    # etc.
                    ba_value = self.cdf.loc[complex_idx, 'ba_value']

                    # Sanity Checking
                    assert atomatom_graph.shape[0] == atomwise_features.shape[0]
                    self.dataset.append([atomwise_features, atomatom_graph, resiwise_features, ba_value])
                    self.idtuples.append([ligand_idx, protein_idx, complex_idx])
                    if self.clu: self.clu_indices.append(int(self.cdf.loc[complex_idx, self.clu]))

                except Exception as e:
                    self.failed_list.append((ligand_idx, protein_idx, complex_idx, e))

            if not args.prelim_mode: pickle.dump((self.dataset, self.idtuples, self.failed_list), open(DATA_CACHE,'wb'))

        self.indices=[i for i in range(len(self.dataset))]

    def get(self, idx):
        atomwise_features, atomatom_graph, resiwise_features, ba_value = self.dataset[idx]
        ligand_idx, protein_idx, complex_idx = self.idtuples[idx]

        pssm = np.load(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.pssm.npy')
        resiwise_features = np.hstack([resiwise_features, pssm])
        if check_exists(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.arpeggio.npy'):
            resiresi_graph = np.load(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.arpeggio.npy')
        else:
            resiresi_graph = np.load(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.pc4cmap.npy')
            resiresi_graph = np.where(resiresi_graph > 0.25, 1, 0)[:,:,np.newaxis]

        #assert resiresi_graph.shape[0] == resiwise_features.shape[0]
        #return [atomwise_features, atomatom_graph, resiwise_features, resiresi_graph, None, ba_value]
        if resiresi_graph.shape[0] == resiwise_features.shape[0]:
            return [atomwise_features, atomatom_graph, resiwise_features, resiresi_graph, None, ba_value]
        else: None

class DeepDTAFDataset(DtiDataset):
    def __init__(self, args):
        super().__init__(args)

        self.max_seq_len = 1000
        self.max_smi_len = 150
        self.max_pkt_len = 63
        self.pkt_window = None
        self.pkt_stride = None

        if self.pkt_window is None or self.pkt_stride is None:
            print(f'Dataset will not fold pkt')

        # physiochem_onehot = ['non_polar', 'polar', 'acidic', 'basic', 'c2_1', 'c2_2', 'c2_3', 'c2_4', 'c2_5', 'c2_6', 'c2_7'] # unknown c2
        self.physiochem_onehot = ['non_polar', 'polar', 'acidic', 'basic']
        self.secondary_onehot = ['s2_B', 's2_C', 's2_E', 's2_G', 's2_H', 's2_I', 's2_S', 's2_T']
        self.aminoacid_onehot = ['a_G', 'a_A', 'a_V', 'a_L', 'a_I', 'a_M', 'a_F', 'a_P', 'a_W', 'a_S', 'a_T', 'a_Y', 'a_C', 'a_Q', 'a_N', 'a_D', 'a_E', 'a_K', 'a_R', 'a_H', 'a_X']
        
        protein_onehot = []
        protein_onehot.extend(self.physiochem_onehot)
        protein_onehot.extend(self.secondary_onehot)
        protein_onehot.extend(self.aminoacid_onehot)
        self.protein_onehot = protein_onehot
        self.all_features = pd.DataFrame(columns=self.protein_onehot)

        def label_smile(smile):
            X = np.zeros(self.max_smi_len, dtype=np.int)
            for i, ch in enumerate(smile[:self.max_smi_len]):
                X[i] = CHAR_SMI_SET[ch] - 1
            return X

        def label_fasta(fasta):
            fasta_df = pd.DataFrame({'a': list(fasta)[:self.max_seq_len]})
            fasta_onehot = pd.get_dummies(fasta_df)

            for aa in self.aminoacid_onehot:
                if aa not in fasta_onehot.columns:
                    fasta_onehot[aa] = 0
            fasta_onehot = fasta_onehot[self.aminoacid_onehot]
            fasta_onehot = pd.merge(self.all_features, fasta_onehot, how='outer').fillna(0)

            assert len(fasta_onehot) == fasta_onehot.sum().sum()
            return fasta_onehot

        def label_pocket(fasta, atomresi_graph):
            pocket_idx = np.transpose((atomresi_graph.sum(axis=2) > 0).nonzero())
            pocket_idx = np.unique(pocket_idx[:,1]) # get fasta interaction index, drop drug
            pocket_idx.sort()

            pocket_fasta = np.array(list(fasta))[pocket_idx].tolist()
            pocket_df = pd.DataFrame({'a': pocket_fasta[:self.max_pkt_len]})
            pocket_onehot = pd.get_dummies(pocket_df)
                
            for aa in self.aminoacid_onehot:
                if aa not in pocket_onehot.columns:
                    pocket_onehot[aa] = 0

            aminoacid_features = pd.DataFrame(columns=self.aminoacid_onehot)
            pocket_onehot = pd.merge(aminoacid_features, pocket_onehot, how='outer')
            pocket_onehot = pocket_onehot[self.aminoacid_onehot]
            pocket_onehot = pd.merge(self.all_features, pocket_onehot, how='outer').fillna(0)

            assert len(pocket_onehot) == pocket_onehot.sum().sum()
            return pocket_fasta, pocket_idx, pocket_onehot

        def label_physiochemical(fasta):
            pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
            pro_res_aromatic_table = ['F', 'W', 'Y']
            pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
            pro_res_acidic_charged_table = ['D', 'E']
            pro_res_basic_charged_table = ['H', 'K', 'R']

            pc_feat = []
            for res in fasta:
                if res in pro_res_acidic_charged_table:
                    pc_feat.append('acidic')
                elif res in pro_res_basic_charged_table:
                    pc_feat.append('basic')
                elif res in pro_res_polar_neutral_table:
                    pc_feat.append('polar')
                else:
                    pc_feat.append('non_polar')

            pc_features = pd.get_dummies(pd.DataFrame({'': pc_feat}))
            pc_features.columns = ['_'.join(col.split('_')[1:]) for col in pc_features.columns]

            for feat in ['acidic', 'basic', 'non_polar', 'polar']:
                if feat not in pc_features.columns:
                    pc_features[feat] = 0
            pc_features = pc_features[self.physiochem_onehot]
            pc_features = pd.merge(pc_features, self.all_features, how='outer').fillna(0)

            assert len(pc_features) == pc_features.sum().sum()
            return pc_features

        def label_secondary(protein_idx, pocket_idx=None):
            # DeepConCNF_SS8: eight-class secondary structure prediction results
            # resi_feat = pickle.load(open(f'{self.data_path}/pkl_proteins/RESF/{protein_idx}.resf', 'rb'))
            scratch1d = pickle.load(open(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.scratch1d', 'rb'))

            # H G I E B T S L(loops), the 8 secondary structure types used in DSSP
            ss8 = scratch1d['ss8']
            ss8_col = ['s2_H', 's2_G', 's2_I', 's2_E', 's2_B', 's2_T', 's2_S', 's2_C']
            ss8 = pd.DataFrame(ss8, columns=ss8_col)
            ss8 = ss8[self.secondary_onehot].astype(int)
            
            if pocket_idx is not None:
                ss8 = ss8.loc[pocket_idx]
            assert ss8.shape[0] == ss8.sum().sum()
            return pd.merge(self.all_features, ss8, how='outer').fillna(0)[self.protein_onehot]

        # self.length = len(self.smi)
        # ba_value = self.cdf.ba_value.to_dict()
        # ligands = self.ldf.smiles.to_dict()
        # proteins = self.pdf.fasta.to_dict()

        index = self.cdf.index
        if args.prelim_mode: index=self.cdf.index[:args.prelim_index]
        DATA_CACHE = self.cache_path+f'deepdtaf_{args.dataset_subsets}.dat'
        if check_exists(DATA_CACHE):
            self.dataset, self.idtuples, self.failed_list = pickle.load(open(DATA_CACHE,'rb'))
            print('Loaded Dataset: ', DATA_CACHE)
            # will include cluster indices later...
        else:
            for idx, complex_idx in enumerate(tqdm(index)):
                try:
                    # Ligand Featurization
                    ligand_idx = self.cdf.loc[complex_idx, 'ligand_id']
                    ligand = self.ldf.loc[ligand_idx, 'smiles']
                    atomwise_features = label_smile(ligand)

                    # Protein Featurization
                    protein_idx = self.cdf.loc[complex_idx, 'protein_id']
                    if self.pdf.loc[protein_idx, 'fasta_length'] > 1000:
                        raise FastaLengthException(self.pdf.loc[protein_idx, 'fasta_length'])
                    fasta = self.pdf.loc[protein_idx, 'fasta']
                    if not check_exists(f'{self.data_path}proteins/{protein_idx}/{protein_idx}.scratch1d'):
                        raise NoProteinFeaturesException(protein_idx)
                    physiochemical_features = label_physiochemical(fasta)
                    secondary_features = label_secondary(protein_idx)
                    aminoacid_features = label_fasta(fasta)
                    resiwise_features = physiochemical_features + secondary_features + aminoacid_features

                    # Pocket Featurization
                    atomresi_graph = np.load(f'{self.data_path}complexes/{complex_idx}/{complex_idx}.arpeggio.npy')
                    assert atomresi_graph.shape[1] == resiwise_features.shape[0]
                    pocket_fasta, pocket_idx, pocket_features = label_pocket(fasta, atomresi_graph)
                    pocket_pc_features = label_physiochemical(pocket_fasta)
                    pocket_ss_features = label_secondary(protein_idx, pocket_idx)
                    pocketwise_features = pocket_pc_features + pocket_ss_features + pocket_features

                    # etc.
                    ba_value = self.cdf.loc[complex_idx, 'ba_value']


                    self.dataset.append([atomwise_features, resiwise_features, pocketwise_features, ba_value])
                    self.idtuples.append([ligand_idx, protein_idx, complex_idx])
                    if self.clu: self.clu_indices.append(int(self.cdf.loc[complex_idx, self.clu]))

                except Exception as e:
                    self.failed_list.append((ligand_idx, protein_idx, complex_idx, e))

            if not args.prelim_mode: pickle.dump((self.dataset, self.idtuples, self.failed_list), open(DATA_CACHE,'wb'))

        self.indices = [i for i in range(len(self.dataset))]

    def get(self, idx):
        atomwise_features, resiwise_features, pocketwise_features, ba_value = self.dataset[idx]
        ligand_idx, protein_idx, complex_idx = self.idtuples[idx]
        pt_features_size = len(self.protein_onehot)
        
        # use np.zeros for padding
        resiwise_tensor = np.zeros((self.max_seq_len, pt_features_size))
        resiwise_tensor[:len(resiwise_features)] = resiwise_features
        resiwise_tensor = resiwise_tensor.astype(np.float32)

        if self.pkt_window is not None and self.pkt_stride is not None:
            pkt_len = (int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride)) * self.pkt_stride + self.pkt_window)
            pocketwise_tensor = np.zeros((pkt_len, pt_features_size))
            pocketwise_tensor = np.array([pocketwise_tensor[i * self.pkt_stride:i * self.pkt_stride + self.pkt_window] 
                                         for i in range(int(np.ceil((self.max_pkt_len - self.pkt_window) / self.pkt_stride)))])
        else:
            pocketwise_tensor = np.zeros((self.max_pkt_len, pt_features_size))
            pocketwise_tensor[:len(pocketwise_features)] = pocketwise_features
        pocketwise_tensor = pocketwise_tensor.astype(np.float32)

        return [atomwise_features, resiwise_tensor, pocketwise_tensor, ba_value]

class DeepDTADataset(DtiDataset):
    def __init__(self, args):
        super().__init__(args)

        def char_features(char):
            return np.array(onek_encoding_unk(char, CHAR_LIST_Deepdta))

        def resi_features(resi):
            return np.array(onek_encoding_unk(resi, RESI_LIST_Dgraph))

        index = self.cdf.index
        if args.prelim_mode: index=self.cdf.index[:args.prelim_index]
        DATA_CACHE = self.cache_path+f'deepdta_{args.dataset_subsets}.dat'
        if check_exists(DATA_CACHE):
            self.dataset, self.idtuples, self.failed_list = pickle.load(open(DATA_CACHE,'rb'))
            print('Loaded Dataset: ', DATA_CACHE)
            # will include cluster indices later...
        else:
            for idx, complex_idx in enumerate(tqdm(index)):
                try:
                    # Ligand Featurization
                    atomwise_features=[]
                    ligand_idx = self.cdf.loc[complex_idx, 'ligand_id']
                    smiles = self.ldf.loc[ligand_idx, 'smiles']
                    if len(smiles) > 100: raise LongSmilesException
                    for char in smiles:
                        try: atomwise_features.append(char_features(char).reshape(1, -1))
                        except: atomwise_features.append(np.zeros(62).reshape(1, -1))
                    atomwise_features = np.vstack(atomwise_features).astype(int)

                    # Protein Featurization
                    resiwise_features = []
                    protein_idx = self.cdf.loc[complex_idx, 'protein_id']
                    if self.pdf.loc[protein_idx, 'fasta_length'] > 1000:
                        raise FastaLengthException(self.pdf.loc[protein_idx, 'fasta_length'])
                    fasta = self.pdf.loc[protein_idx, 'fasta']
                    for resi in fasta:
                        try: resiwise_features.append(resi_features(resi).reshape(1, -1))
                        except: resiwise_features.append(np.zeros(21).reshape(1, -1))
                    resiwise_features = np.vstack(resiwise_features).astype(int)

                    # etc.
                    ba_value = self.cdf.loc[complex_idx, 'ba_value']

                    self.dataset.append([atomwise_features, resiwise_features, ba_value])
                    self.idtuples.append([ligand_idx, protein_idx, complex_idx])
                    if self.clu: self.clu_indices.append(int(self.cdf.loc[complex_idx, self.clu]))

                except Exception as e:
                    self.failed_list.append((ligand_idx, protein_idx, complex_idx, e))

            if not args.prelim_mode: pickle.dump((self.dataset, self.idtuples, self.failed_list), 
                                                  open(DATA_CACHE,'wb'))
        self.indices = [i for i in range(len(self.dataset))]

    def get(self, idx):
        atomwise_features, resiwise_features, ba_value = self.dataset[idx]
        ligand_idx, protein_idx, complex_idx = self.idtuples[idx]

        return [atomwise_features, resiwise_features, ba_value]

class GraphDTADataset(DtiDataset):
    def __init__(self, args):
        super().__init__(args)

        def atom_features(atom):
            return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST_Dgraph)
                          + onek_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                          + onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                          + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                          + [atom.GetIsAromatic()], dtype=np.float32)


        def resi_features(resi):
            return RESI_DICT_Graph[resi]

        index = self.cdf.index
        if args.prelim_mode: index=self.cdf.index[:args.prelim_index]
        DATA_CACHE = self.cache_path+f'graphdta_{args.dataset_subsets}.dat'
        if check_exists(DATA_CACHE):
            self.dataset, self.idtuples, self.failed_list = pickle.load(open(DATA_CACHE,'rb'))
            print('Loaded Dataset: ', DATA_CACHE)
            # will include cluster indices later...

        else:
            for idx, complex_idx in enumerate(tqdm(index)):
                try:
                    # Ligand Featurization
                    atomwise_features=[]
                    ligand_idx = self.cdf.loc[complex_idx, 'ligand_id']
                    mol = Chem.MolFromSmiles(self.ldf.loc[ligand_idx, 'smiles'])

                    for atom in mol.GetAtoms():
                        try: atomwise_features.append(atom_features(atom).reshape(1, -1))
                        except: atomwise_features.append(np.zeros(78).reshape(1, -1))
                    #import pdb; pdb.set_trace()
                    atomwise_features = np.vstack(atomwise_features)

                    # Protein Featurization
                    resiwise_features = []
                    protein_idx = self.cdf.loc[complex_idx, 'protein_id']
                    if self.pdf.loc[protein_idx, 'fasta_length'] > 1000:
                        raise FastaLengthException(self.pdf.loc[protein_idx, 'fasta_length'])
                    fasta = self.pdf.loc[protein_idx, 'fasta']


                    for resi in fasta:
                        try: resiwise_features.append(resi_features(resi))
                        except: resiwise_features.append(0)
                    resiwise_features = np.vstack(resiwise_features)
                    
                    # Graph-based Data Arrays
                    atomatom_graph = Chem.rdmolops.GetAdjacencyMatrix(mol)
                    
                    # etc.
                    ba_value = self.cdf.loc[complex_idx, 'ba_value']
                    
                    # Sanity Checking
                    assert atomatom_graph.shape[0] == atomwise_features.shape[0]
                    self.dataset.append([atomwise_features, atomatom_graph, resiwise_features, ba_value])
                    self.idtuples.append([ligand_idx, protein_idx, complex_idx])
                    if self.clu: self.clu_indices.append(int(self.cdf.loc[complex_idx, self.clu]))

                except Exception as e:
                    self.failed_list.append((ligand_idx, protein_idx, complex_idx, e))

            if not args.prelim_mode: pickle.dump((self.dataset, self.idtuples, self.failed_list), open(DATA_CACHE,'wb'))

        self.indices = [i for i in range(len(self.dataset))]

    def get(self, idx):
        atomwise_features, atomatom_graph, resiwise_features, ba_value = self.dataset[idx]
        ligand_idx, protein_idx, complex_idx = self.idtuples[idx]

        return [atomwise_features, atomatom_graph, resiwise_features, ba_value]
        
class TransformerCPIDataset(DtiDataset):
    def __init__(self, args):
        super().__init__(args)

        def atom_features(atom):
            hybridizationType = [Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2,
                                'other'] 
                                
            result = onek_encoding_unk(atom.GetSymbol(), ATOM_LIST_Transformer) + \
                     onek_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                     [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                     onek_encoding_unk(atom.GetHybridization(), hybridizationType) + \
                     [atom.GetIsAromatic()] + \
                     onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            try:
                result = result + onek_encoding_unk(atom.GetProp('_CIPCode'),['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
                return np.array(result, dtype = np.float32) 
            except:
                result = result + [False, False] + [atom.HasProp('_ChiralityPossible')]
                return np.array(result, dtype = np.float32) 
                
        def resi_features(resi):
            return RESI_DICT_Graph[resi]

        index = self.cdf.index
        if args.prelim_mode: index=self.cdf.index[:args.prelim_index]
        DATA_CACHE = self.cache_path+f'transformercpi_{args.dataset_subsets}.dat'
        if check_exists(DATA_CACHE):
            self.dataset, self.idtuples, self.failed_list = pickle.load(open(DATA_CACHE,'rb'))
            print('Loaded Dataset: ', DATA_CACHE)
            # will include cluster indices later...
        else:
            for idx, complex_idx in enumerate(tqdm(index)):
                try:
                    # Ligand Featurization
                    atomwise_features=[]
                    ligand_idx = self.cdf.loc[complex_idx, 'ligand_id']
                    mol=Chem.MolFromSmiles(self.ldf.loc[ligand_idx, 'smiles'])

                    for atom in mol.GetAtoms():
                        try: atomwise_features.append(atom_features(atom).reshape(1, -1))
                        except: atomwise_features.append(np.zeros(34).reshape(1, -1))

                    atomwise_features=np.vstack(atomwise_features)
                    
                    # Protein Featurization              
                    model = Word2Vec.load('word2vec_30.model')
                    protein_idx = self.cdf.loc[complex_idx, 'protein_id']
                    if self.pdf.loc[protein_idx, 'fasta_length'] > 1000:
                        raise FastaLengthException(self.pdf.loc[protein_idx, 'fasta_length'])
                    fasta = self.pdf.loc[protein_idx, 'fasta']
                    protein_embedding = get_protein_embedding(model, seq_to_kmers(fasta))
                    resiwise_features=protein_embedding
                    
                    # Graph-based Data Arrays
                    atomatom_graph = Chem.rdmolops.GetAdjacencyMatrix(mol)
                    
                    # etc.
                    ba_value = self.cdf.loc[complex_idx, 'ba_value']
                    
                    # Sanity Checking
                    assert atomatom_graph.shape[0] == atomwise_features.shape[0]
                    self.dataset.append([atomwise_features, atomatom_graph, resiwise_features, ba_value])
                    self.idtuples.append([ligand_idx, protein_idx, complex_idx])
                    if self.clu: self.clu_indices.append(int(self.cdf.loc[complex_idx, self.clu]))

                except Exception as e:
                    self.failed_list.append((ligand_idx, protein_idx, complex_idx, e))
                    
            if not args.prelim_mode: pickle.dump((self.dataset, self.idtuples, self.failed_list), open(DATA_CACHE,'wb'))

        self.indices = [i for i in range(len(self.dataset))]

    def get(self, idx):
        atomwise_features, atomatom_graph, resiwise_features, ba_value = self.dataset[idx]
        ligand_idx, protein_idx, complex_idx = self.idtuples[idx]

        return [atomwise_features, atomatom_graph, resiwise_features, ba_value]


class FastaLengthException(Exception):
    def __init__(self, fasta_length, message="fasta length should not exceed 1000"):
        self.fasta_length = fasta_length
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.fasta_length} -> {self.message}'

class NoProteinGraphException(Exception):
    def __init__(self, protein_idx, message="protein graph structure file not available"):
        self.protein_idx = protein_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.protein_idx} -> {self.message}'

class NoProteinFeaturesException(Exception):
    def __init__(self, protein_idx, message="protein advanced features file not available"):
        self.protein_idx = protein_idx
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.protein_idx} -> {self.message}'
