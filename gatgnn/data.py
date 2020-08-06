import numpy as np
import pandas as pd 
import functools
import torch
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as torch_Dataset
from torch_geometric.data import Data, DataLoader as torch_DataLoader
import sys,json,os
from pymatgen.core.structure import Structure
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SPCL
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

class ELEM_Encoder:
    def __init__(self):
        self.elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 
                        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                        'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 
                        'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 
                        'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 
                        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
        self.e_arr    = np.array(self.elements)

    def encode(self,composition_dict):
        answer   = [0]*len(self.elements)

        elements = [str(i) for i in composition_dict.keys()]
        counts   = [j for j in composition_dict.values()]
        total    = sum(counts)

        for idx in range(len(elements)):
            elem   = elements[idx]
            ratio  = counts[idx]/total
            idx_e  = self.elements.index(elem)
            answer[idx_e] = ratio
        return torch.tensor(answer).float().view(1,-1)
    def decode_pymatgen_num(tensor_idx):
        idx      = (tensor_idx-1).cpu().tolist()
        return self.e_arr[idx]
        

class DATA_normalizer:
    def __init__(self, array):
        tensor = torch.tensor(array)
        self.mean   = torch.mean(tensor)
        self.std    = torch.std(tensor)
    def reg(self,x):
        return x.float()

    def log10(self,x):
        return torch.log10(x)

    def delog10(self,x):
        return 10*x

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean

class METRICS:
    def __init__(self,c_property,epoch,torch_criterion,torch_func,device):
        self.c_property        = c_property
        self.criterion         = torch_criterion
        self.eval_func         = torch_func
        self.dv                = device
        self.training_measure1 = torch.tensor(0.0).to(device)
        self.training_measure2 = torch.tensor(0.0).to(device)
        self.valid_measure1    = torch.tensor(0.0).to(device)
        self.valid_measure2    = torch.tensor(0.0).to(device)

        self.training_counter  = 0
        self.valid_counter     = 0

        self.training_loss1    = []
        self.training_loss2    = []
        self.valid_loss1       = []
        self.valid_loss2       = []
        self.duration          = []
        self.dataframe         = self.to_frame()

    def __str__(self):
        x = self.to_frame() 
        return x.to_string()

    def to_frame(self):
        metrics_df = pd.DataFrame(list(zip(self.training_loss1,self.training_loss2,
                                           self.valid_loss1,self.valid_loss2,self.duration)),
                                           columns=['training_1','training_2','valid_1','valid_2','time'])
        return metrics_df

    def set_label(self,which_phase,graph_data):
        use_label = graph_data.y
        return use_label

    def save_time(self,e_duration):
        self.duration.append(e_duration)


    def __call__(self,which_phase,tensor_pred,tensor_true,measure=1):
        if measure == 1:
            if   which_phase == 'training'  : 
                loss         = self.criterion(tensor_pred,tensor_true)
                self.training_measure1 += loss
            elif which_phase == 'validation':
                loss         = self.criterion(tensor_pred,tensor_true)
                self.valid_measure1    += loss
        else:
            if   which_phase == 'training'  :
                loss         = self.eval_func(tensor_pred,tensor_true) 
                self.training_measure2 += loss
            elif which_phase == 'validation':
                loss         = self.eval_func(tensor_pred,tensor_true)
                self.valid_measure2    += loss
        return loss
    def reset_parameters(self,which_phase,epoch):
        if   which_phase == 'training'  :
            # AVERAGES
            t1 = self.training_measure1/(self.training_counter)
            t2 = self.training_measure2/(self.training_counter)

            self.training_loss1.append(t1.item())
            self.training_loss2.append(t2.item())
            self.training_measure1 = torch.tensor(0.0).to(self.dv)
            self.training_measure2 = torch.tensor(0.0).to(self.dv)
            self.training_counter  = 0 
        else:
            # AVERAGES
            v1 = self.valid_measure1/(self.valid_counter)
            v2 = self.valid_measure2/(self.valid_counter)

            self.valid_loss1.append(v1.item())
            self.valid_loss2.append(v2.item())
            self.valid_measure1    = torch.tensor(0.0).to(self.dv)
            self.valid_measure2    = torch.tensor(0.0).to(self.dv)      
            self.valid_counter     = 0 
    def save_info(self):
        with open('MODELS/metrics_.pickle','wb') as metrics_file:
            pickle.dump(self,metrics_file)
            
class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]

class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CIF_Lister(Dataset):
    def __init__(self,crystals_ids,full_dataset,norm_obj=None,normalization=None,df=None,src=None):
        self.crystals_ids  = crystals_ids
        self.full_dataset  = full_dataset
        self.normalizer    = norm_obj
        self.material_ids  = df.iloc[crystals_ids].values[:,0].squeeze()
        self.normalization = normalization
        self.src           = src

    def __len__(self):
        return len(self.crystals_ids)

    def extract_ids(self,original_dataset):
        names    = original_dataset.iloc[self.crystals_ids]
        return names

    def __getitem__(self, idx):
        i        = self.crystals_ids[idx]
        material = self.full_dataset[i]

        n_features    = material[0][0]
        e_features    = material[0][1]
        e_features    = e_features.view(-1,41) if self.src in ['CGCNN','NEW'] else e_features.view(-1,9)
        a_matrix      = material[0][2]

        groups        = material[1]
        enc_compo     = material[2]
        coordinates   = material[3]
        y             = material[4]

        if   self.normalization == None:    y = y
        elif self.normalization == 'log':   y = self.normalizer.log10(y)
        elif self.normalization == 'classification-1': 
            if   y == 0:    y = torch.tensor([0]).long()
            elif y == 1:    y = torch.tensor([1]).long()
        elif self.normalization == 'classification-0': 
            if   y == 0:    y = torch.tensor([1]).long()
            elif y == 1:    y = torch.tensor([0]).long()      

        graph_crystal = Data(x=n_features,y = y,edge_attr=e_features,edge_index=a_matrix,global_feature = enc_compo,cluster=groups,num_atoms=torch.tensor([len(n_features)]).float(),coords=coordinates,the_idx=torch.tensor([float(i)]))
        return graph_crystal

class CIF_Dataset(Dataset):
    def __init__(self,part_data=None,norm_obj=None,normalization=None,max_num_nbr=12, radius=8, dmin=0, step=0.2,cls_num=3,root_dir = 'DATA/CIF-DATA/'):
        self.root_dir      = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.normalizer    = norm_obj
        self.normalization = normalization
        self.full_data     = part_data
        self.ari           = AtomCustomJSONInitializer(self.root_dir+'atom_init.json')
        self.gdf           = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.clusterizer   = SPCL(n_clusters=cls_num, random_state=None,assign_labels='discretize')
        self.clusterizer2  = KMeans(n_clusters=cls_num, random_state=None)
        self.encoder_elem  = ELEM_Encoder()
        self.update_root   = None

    def __len__(self):
        return len(self.partial_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.full_data.iloc[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        nbr_fea     = self.gdf.expand(nbr_fea)
        g_coords    = crystal.cart_coords
        groups      = [0]*len(g_coords)
        if len(g_coords) > 2:
            try    : groups = self.clusterizer.fit_predict(g_coords)
            except : groups = self.clusterizer2.fit_predict(g_coords)
        groups  = torch.tensor(groups).long()

        atom_fea     = torch.Tensor(atom_fea)
        nbr_fea      = torch.Tensor(nbr_fea)
        nbr_fea_idx  = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))
        target       = torch.Tensor([float(target)])

        coordinates = torch.tensor(g_coords) 
        enc_compo   = self.encoder_elem.encode(crystal.composition)
        return (atom_fea, nbr_fea, nbr_fea_idx),groups,enc_compo,coordinates, target, cif_id,[crystal[i].specie for i in range(len(crystal))]

    def format_adj_matrix(self,adj_matrix):
        size           = len(adj_matrix)
        src_list       = list(range(size))
        all_src_nodes  = torch.tensor([[x]*adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes  = adj_matrix.view(-1).unsqueeze(0)
        return torch.cat((all_src_nodes,all_dst_nodes),dim=0)
