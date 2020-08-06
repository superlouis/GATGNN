from gatgnn.data                   import *
from gatgnn.model                  import *
from gatgnn.pytorch_early_stopping import *
from gatgnn.file_setter            import use_property
from gatgnn.utils                  import *

# MOST CRUCIAL DATA PARAMETERS
parser = argparse.ArgumentParser(description='GATGNN')
parser.add_argument('--property', default='bulk-modulus',
                    choices=['absolute-energy','band-gap','bulk-modulus',
                             'fermi-energy','formation-energy',
                             'poisson-ratio','shear-modulus','new-property'],
                    help='material property to train (default: bulk-modulus)')
parser.add_argument('--data_src', default='CGCNN',choices=['CGCNN','MEGNET','NEW'],
                    help='selection of the materials dataset to use (default: CGCNN)')
parser.add_argument('--to_predict', default='mp-1', help='name or id of cif material whose property to predict')

# MOST CRUCIAL MODEL PARAMETERS
parser.add_argument('--num_layers',default=3, type=int,
                    help='number of AGAT layers to use in model (default:3)')
parser.add_argument('--num_neurons',default=64, type=int,
                    help='number of neurons to use per AGAT Layer(default:64)')
parser.add_argument('--num_heads',default=4, type=int,
                    help='number of Attention-Heads to use  per AGAT Layer (default:4)')
parser.add_argument('--use_hidden_layers',default=True, type=bool,
                    help='option to use hidden layers following global feature summation (default:True)')
parser.add_argument('--global_attention',default='composition', choices=['composition','cluster']
                    ,help='selection of the unpooling method as referenced in paper GI M-1 to GI M-4 (default:composition)')
parser.add_argument('--cluster_option',default='fixed', choices=['fixed','random','learnable'],
                    help='selection of the cluster unpooling strategy referenced in paper GI M-1 to GI M-4 (default: fixed)')
parser.add_argument('--concat_comp',default=False, type=bool,
                    help='option to re-use vector of elemental composition after global summation of crystal feature.(default: False)')
parser.add_argument('--train_size',default=0.8, type=float,
                    help='ratio size of the training-set (default:0.8)')
args = parser.parse_args(sys.argv[1:])

# GATGNN --- parameters
crystal_property                     = args.property
data_src                             = args.data_src
material_name                        = args.to_predict

_, _,RSM                             = use_property(crystal_property,data_src, True)
norm_action, classification          = set_model_properties(crystal_property)

number_layers                        = args.num_layers
number_neurons                       = args.num_neurons
n_heads                              = args.num_heads
xtra_l                               = args.use_hidden_layers 
global_att                           = args.global_attention
attention_technique                  = args.cluster_option
concat_comp                          = args.concat_comp

# SETTING UP CODE TO RUN ON GPU
gpu_id = 0
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# MODEL HYPER-PARAMETERS
learning_rate   = 5e-3
batch_size      = 256
test_param      = {'batch_size':batch_size, 'shuffle': False}

# DATALOADER/ TARGET NORMALIZATION
dataset                 = pd.DataFrame()
dataset['material_ids'] = [material_name]
dataset['label']        = [0.00001]
NORMALIZER              = DATA_normalizer(dataset.label.values)

CRYSTAL_DATA          = CIF_Dataset(dataset,**RSM)
CRYSTAL_DATA.root_dir = 'DATA/prediction-directory'
test_idx              = list(range(len(dataset)))
testing_set           = CIF_Lister(test_idx,CRYSTAL_DATA,NORMALIZER,norm_action, df=dataset,src=data_src)


# NEURAL-NETWORK
the_network    = GATGNN(n_heads,classification,neurons=number_neurons,nl=number_layers,xtra_layers=xtra_l,global_attention=global_att,
                                      unpooling_technique=attention_technique,concat_comp=concat_comp,edge_format=data_src)
net            = the_network.to(device)

# LOSS & OPTMIZER & SCHEDULER
if classification == 1: criterion   = nn.CrossEntropyLoss().cuda(); funct = torch_accuracy
else                  : criterion   = nn.SmoothL1Loss().cuda()    ; funct = torch_MAE
optimizer         = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 1e-1)

# LOADING MODEL
net = the_network.to(device)
net.load_state_dict(torch.load(f'TRAINED/{crystal_property}.pt',map_location=device))

# METRICS-OBJECT INITIALIZATION
metrics        = METRICS(crystal_property,0,criterion,funct,device)

print(f'> PREDICTING MATERIAL-PROPERTY ...')
# TESTING PHASE
test_loader    = torch_DataLoader(dataset=testing_set,    **test_param)
net.eval()

for data in test_loader:
    data        = data.to(device)
    with torch.no_grad():
        prediction = net(data)
    print(f'> {crystal_property} of material ({material_name}.cif) = {prediction:.4}')
