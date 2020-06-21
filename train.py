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
                             'poisson-ratio','shear-modulus'],
                    help='material property to train (default: bulk-modulus)')
parser.add_argument('--data_src', default='CGCNN',choices=['CGCNN','MEGNET'],
                    help='selection of the materials dataset to use (default: CGCNN)')

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

args = parser.parse_args(sys.argv[1:])


# GATGNN --- parameters
crystal_property                     = args.property
data_src                             = args.data_src
source_comparison, training_num,RSM  = use_property(crystal_property,data_src)
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

# DATA PARAMETERS
random_num          =  456;random.seed(random_num)

# MODEL HYPER-PARAMETERS
num_epochs      = 500
learning_rate   = 5e-3
batch_size      = 256

stop_patience   = 150
best_epoch      = 1
adj_epochs      = 50
milestones      = [150,250]
train_param     = {'batch_size':batch_size, 'shuffle': True}
valid_param     = {'batch_size':256, 'shuffle': True}

# DATALOADER/ TARGET NORMALIZATION
dataset         = pd.read_csv('DATA/CIF-DATA/id_prop.csv',names=['material_ids','label']).sample(frac=1,random_state=random_num)
NORMALIZER      = DATA_normalizer(dataset.label.values)

CRYSTAL_DATA    = CIF_Dataset(dataset,**RSM)
idx_list        = list(range(len(dataset)))
random.shuffle(idx_list)

train_idx,test_val = train_test_split(idx_list,train_size=training_num,random_state=random_num)
_,       val_idx   = train_test_split(test_val,test_size=0.5,random_state=random_num)

training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,NORMALIZER,norm_action,df=dataset,src=data_src)
validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,NORMALIZER,norm_action,  df=dataset,src=data_src)

# NEURAL-NETWORK
the_network    = GATGNN(n_heads,classification,neurons=number_neurons,nl=number_layers,xtra_layers=xtra_l,global_attention=global_att,
                                      unpooling_technique=attention_technique,concat_comp=concat_comp,edge_format=data_src)
net            = the_network.to(device)

# LOSS & OPTMIZER & SCHEDULER
if classification == 1: criterion   = nn.CrossEntropyLoss().cuda(); funct = torch_accuracy
else                  : criterion   = nn.SmoothL1Loss().cuda()    ; funct = torch_MAE
optimizer         = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 1e-1)
scheduler         = lr_scheduler.MultiStepLR(optimizer, milestones=milestones,gamma=0.3)

# EARLY-STOPPING INITIALIZATION
early_stopping = EarlyStopping(patience=stop_patience, increment=1e-6,verbose=True,save_best=True,classification=classification)

# METRICS-OBJECT INITIALIZATION
metrics        = METRICS(crystal_property,num_epochs,criterion,funct,device)

print(f'> TRAINING MODEL ...')
train_loader   = torch_DataLoader(dataset=training_set,   **train_param)
valid_loader   = torch_DataLoader(dataset=validation_set, **valid_param) 
for epoch in range(num_epochs):
    # TRAINING-STAGE
    net.train()       
    start_time       = time.time()
    for data in train_loader:
        data         = data.to(device)
        predictions  = net(data)
        train_label  = metrics.set_label('training',data)
        loss         = metrics('training',predictions,train_label,1)
        _            = metrics('training',predictions,train_label,2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.training_counter+=1
    metrics.reset_parameters('training',epoch)
    # VALIDATION-PHASE
    net.eval()
    for data in valid_loader:
        data = data.to(device)
        with torch.no_grad():
            predictions    = net(data)
        valid_label        = metrics.set_label('validation',data)
        _                  = metrics('validation',predictions,valid_label,1)
        _                  = metrics('validation',predictions, valid_label,2)

        metrics.valid_counter+=1

    metrics.reset_parameters('validation',epoch)
    scheduler.step()
    end_time         = time.time()
    e_time           = end_time-start_time
    metrics.save_time(e_time)
    
    # EARLY-STOPPING
    early_stopping(metrics.valid_loss2[epoch], net)
    flag_value = early_stopping.flag_value+'_'*(22-len(early_stopping.flag_value))
    if early_stopping.FLAG == True:    estop_val = flag_value
    else:
        estop_val        = '@best: saving model...'; best_epoch = epoch+1
    output_training(metrics,epoch,estop_val,f'{e_time:.1f} sec.')

    if early_stopping.early_stop:
        print("> Early stopping")
        break
# SAVING MODEL
print(f"> DONE TRAINING !")
