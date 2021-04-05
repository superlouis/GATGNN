import torch, numpy as np
import torch.optim as optim
from   torch.optim import lr_scheduler 
from   torch.nn import Linear, Dropout, Parameter
import torch.nn.functional as F 
import torch.nn as nn

from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax
from torch_geometric.nn       import global_add_pool
from torch_geometric.nn       import GATConv
from torch_scatter            import scatter_add
from torch_geometric.nn.inits import glorot, zeros
torch.cuda.empty_cache()

class COMPOSITION_Attention(torch.nn.Module):
    def __init__(self,neurons):
        '''
        Global-Attention Mechanism based on the crystal's elemental composition
        > Defined in paper as *GI-M1*
        =======================================================================
        neurons : number of neurons to use 
        '''
        super(COMPOSITION_Attention, self).__init__()
        self.node_layer1    = Linear(neurons+103,32)
        self.atten_layer    = Linear(32,1)

    def forward(self,x,batch,global_feat):
        counts      = torch.unique(batch,return_counts=True)[-1]
        graph_embed = global_feat
        graph_embed = torch.repeat_interleave(graph_embed, counts, dim=0)
        chunk       = torch.cat([x,graph_embed],dim=-1)
        x           = F.softplus(self.node_layer1(chunk))
        x           = self.atten_layer(x)
        weights     = softmax(x,batch)
        return weights

class CLUSTER_Attention(nn.Module):
    def __init__(self,neurons_1,neurons_2,num_cluster, cluster_method = 'random'):
        '''
        Global-Attention Mechanism based on clusters (position grouping) of crystals' elements
        > Defined in paper as *GI-M2*, *GI-M3*, *GI-M4*
        ======================================================================================
        neurons_1       : number of neurons to use for layer_1
        neurons_2       : number of neurons to use for the attention-layer
        num_cluster     : number of clusters to use 
        cluster_method  : unpooling method to use 
            - fixed     : (GI-M2)
            - random    : (GI-M3)
            - learnable : (GI-M4)
        '''        
        super(CLUSTER_Attention,self).__init__()
        self.learn_unpool       = Linear(2*neurons_1+3,num_cluster)        
        self.layer_1            = Linear(neurons_1,neurons_2)
        self.negative_slope     = 0.45
        self.atten_layer        = Linear(neurons_2,1)
        self.clustering_method  = cluster_method
        if not self.training: np.random.seed(0)

    
    def forward(self,x,cls,batch):
        r_x     = self.unpooling_featurizer(x,cls,batch)
        r_x     = F.leaky_relu(self.layer_1(r_x),self.negative_slope)
        r_x     = self.atten_layer(r_x)
        weights  = softmax(r_x,batch)
        return weights

    def unpooling_featurizer(self,x,cls,batch):
        g_counts  = torch.unique(batch,return_counts=True)[-1].tolist()
        split_x   = torch.split(x,g_counts)
        split_cls = torch.split(cls,g_counts)
        new_x     = torch.tensor([]).cuda()
        # break batch into graphs
        for i in range(len(split_x)):
            graph_features = split_x[i]
            clus_t         = split_cls[i].view(-1)
            cluster_sum    = scatter_add(graph_features,clus_t,0)
            zero_sum       = torch.zeros_like(cluster_sum)
            if   len(graph_features) == 1: 
                new_x = torch.cat([new_x,cluster_sum],dim=0)
            elif len(graph_features) == 2:
                new_x = torch.cat([new_x,cluster_sum,cluster_sum],dim=0)
            else:
                region_arr  = np.array(clus_t.tolist())
                # choosing unpooling method
                if self.clustering_method == 'fixed':        #--- GI M-2
                    random_sets  = clus_t.tolist()                
                elif self.clustering_method   == 'random':   #--- GI M-3
                    random_sets  = [np.random.choice(np.setdiff1d(region_arr,e)) for e in region_arr]
                elif self.clustering_method == 'learnable':  #--- GI M-4
                    total_feat   = graph_features.sum(dim=0).unsqueeze(0)
                    region_input = torch.cat([graph_features,total_feat,clus_t.unsqueeze(0).float()],dim=-1)
                    random_sets  = torch.argmax(F.softmax(self.learn_unpool(region_input)),dim=1).tolist()

                # normalized-regions
                unique, counts = np.unique(region_arr, return_counts=True) 
                counts         = counts/counts.sum()
                sets_dict      = dict(zip(unique, counts))             
                random_ratio   = torch.tensor([sets_dict[i] for i in random_sets]).cuda()
                random_ratio   = (random_ratio/random_ratio.sum()).view(-1,1)
                
                cluster_sum = cluster_sum[random_sets]
                cluster_sum = cluster_sum*random_ratio
                new_x       = torch.cat([new_x,cluster_sum],dim=0)
        return new_x        

class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False,
                 dropout=0, bias=True, **kwargs):
        '''
        Our Augmented Graph Attention Layer
        > Defined in paper as *AGAT*
        =======================================================================
        in_features    : input-features
        out_features   : output-features
        edge_dim       : edge-features
        heads          : attention-heads
        concat         : to concatenate the attention-heads or sum them
        dropout        : 0
        bias           : True
        '''    
        super(GAT_Crystal, self).__init__(aggr='add',flow='target_to_source', **kwargs)
        self.in_features       = in_features
        self.out_features      = out_features
        self.heads             = heads
        self.concat            = concat
        self.dropout           = dropout
        self.neg_slope         = 0.2
        self.prelu             = nn.PReLU()
        self.bn1               = nn.BatchNorm1d(heads)
        self.W                 = Parameter(torch.Tensor(in_features+edge_dim,heads*out_features))
        self.att               = Parameter(torch.Tensor(1,heads,2*out_features))

        if bias and concat       : self.bias = Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat : self.bias = Parameter(torch.Tensor(out_features))
        else                     : self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x,edge_attr=edge_attr)

    def message(self, edge_index_i, x_i, x_j, size_i,edge_attr): 
        x_i   = torch.cat([x_i,edge_attr],dim=-1)
        x_j   = torch.cat([x_j,edge_attr],dim=-1)
        
        x_i   = F.softplus(torch.matmul(x_i,self.W))
        x_j   = F.softplus(torch.matmul(x_j,self.W))
        x_i   = x_i.view(-1, self.heads, self.out_features)
        x_j   = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1)*self.att).sum(dim=-1))
        alpha = F.softplus(self.bn1(alpha))
        alpha = softmax(alpha,edge_index_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j   = (x_j * alpha.view(-1, self.heads, 1)).transpose(0,1)
        return x_j

    def update(self, aggr_out,x):
        if self.concat is True:    aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:                      aggr_out = aggr_out.mean(dim=0)
        if self.bias is not None:  aggr_out = aggr_out + self.bias
        return aggr_out

class GATGNN(torch.nn.Module):
    def __init__(self,heads,classification=None,neurons=64,nl=3,xtra_layers=True,global_attention='composition',
                 unpooling_technique='random',concat_comp=False,edge_format='CGCNN'):
        super(GATGNN, self).__init__()

        self.n_heads        = heads
        self.classification = True if classification != None else False 
        self.unpooling      = unpooling_technique
        self.g_a            = global_attention
        self.number_layers  = nl
        self.concat_comp    = concat_comp
        self.additional     = xtra_layers   

        n_h, n_hX2          = neurons, neurons*2
        self.neurons        = neurons
        self.neg_slope      = 0.2  

        self.embed_n        = Linear(92,n_h)
        self.embed_e        = Linear(41,n_h) if edge_format in ['CGCNN','NEW'] else Linear(9,n_h)
        self.embed_comp     = Linear(103,n_h)
 
        self.node_att       = nn.ModuleList([GAT_Crystal(n_h,n_h,n_h,self.n_heads) for i in range(nl)])
        self.batch_norm     = nn.ModuleList([nn.BatchNorm1d(n_h) for i in range(nl)])

        self.cluster_att    = CLUSTER_Attention(n_h,n_h,3,self.unpooling)
        self.comp_atten     = COMPOSITION_Attention(n_h)

        if self.concat_comp : reg_h   = n_hX2
        else                : reg_h   = n_h

        if self.additional:
            self.linear1    = nn.Linear(reg_h,reg_h)
            self.linear2    = nn.Linear(reg_h,reg_h)

        if self.classification :    self.out  =  Linear(reg_h,2)
        else:                       self.out  =  Linear(reg_h,1)

    def forward(self,data):
        x, edge_index, edge_attr   = data.x, data.edge_index,data.edge_attr
        batch, global_feat,cluster = data.batch,data.global_feature,data.cluster
        
        x           = self.embed_n(x)
        edge_attr   = F.leaky_relu(self.embed_e(edge_attr),self.neg_slope)

        for a_idx in range(len(self.node_att)):
            x     = self.node_att[a_idx](x,edge_index,edge_attr)
            x     = self.batch_norm[a_idx](x)
            x     = F.softplus(x)
        
        if   self.g_a in ['cluster', 'unpooling', 'clustering']:
            ar        = self.cluster_att(x,cluster,batch)
            x         = (x)*ar
        elif self.g_a =='composition':
            ag        = self.comp_atten(x,batch,global_feat)
            x         = (x)*ag
        
        # CRYSTAL FEATURE-AGGREGATION 
        y         = global_add_pool(x,batch).unsqueeze(1).squeeze()
        if self.concat_comp:
            y     = torch.cat([y,F.leaky_relu(self.embed_comp(global_feat),self.neg_slope)],dim=-1)

        if self.additional:
            y = F.softplus(self.linear1(y))
            y = F.softplus(self.linear2(y))

        if self.classification: y = self.out(y)
        else:                   y = self.out(y).squeeze()
        return y
