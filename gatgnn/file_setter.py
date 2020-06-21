import pandas as pd
import numpy as np


def save_properties():
    df  = pd.read_csv('DATA/data_set.csv')

    # FERMI-ENERGY
    fermener  = df[['material_id','fermi_e']].dropna()
    fermener.to_csv('DATA/properties-reference/fermienergy.csv',index=False,header=False)
    # FORMATION-ENERGY
    formener  = df[['material_id','formation_e']].dropna()
    formener.to_csv('DATA/properties-reference/formationenergy.csv',index=False,header=False)
    # ABSOLUTE-ENERGY
    absener   = df[['material_id','absolute_e']].dropna()
    absener.to_csv('DATA/properties-reference/absoluteenergy.csv',index=False,header=False)
    # BAND-GAP
    bandgap   = df[['material_id','band_gap']].dropna()
    bandgap.to_csv('DATA/properties-reference/bandgap.csv',index=False,header=False)
    # BULK-MODULI
    bulkmod   = df[['material_id','bulk_moduli']].dropna()
    bulkmod.to_csv('DATA/properties-reference/bulkmodulus.csv',index=False,header=False)
    # SHEAR-MODULI
    shearmod  = df[['material_id','shear_moduli']].dropna()
    shearmod.to_csv('DATA/properties-reference/shearmodulus.csv',index=False,header=False)
    # POISSON-RATIO
    poissonr  = df[['material_id','poisson_ratio']].dropna()
    poissonr.to_csv('DATA/properties-reference/poissonratio.csv',index=False,header=False)    
    # METAL-YES/NO
    metal     = df[['material_id','is_metal']].dropna()
    metal.to_csv('DATA/properties-reference/ismetal.csv',index=False,header=False)

    print('> Saved all properties for materials-project data-set')
    exit()


def use_property(property_name,source):

    print('> Preparing dataset to use for Property Prediction. Please wait ...')

    if property_name   in ['band','bandgap','band-gap']:                      filename = 'bandgap.csv'         ;p=1;num_T  = 36720
    elif property_name in ['bulk','bulkmodulus','bulk-modulus','bulk-moduli']:filename = 'bulkmodulus.csv'     ;p=3;num_T  = 4664
    elif property_name in ['energy-1','formationenergy','formation-energy']:  filename = 'formationenergy.csv' ;p=2;num_T  = 60000
    elif property_name in ['energy-2','fermienergy','fermi-energy']:          filename = 'fermienergy.csv'     ;p=2;num_T  = 60000
    elif property_name in ['energy-3','absoluteenergy','absolute-energy']:    filename = 'absoluteenergy.csv'  ;p=2;num_T  = 60000               
    elif property_name in ['shear','shearmodulus','shear-modulus','shear-moduli']:filename = 'shearmodulus.csv';p=4;num_T  = 4664
    elif property_name in ['poisson','poissonratio','poisson-ratio']:         filename = 'poissonratio.csv'    ;p=4;num_T  = 4664
    elif property_name in ['is_metal','is_not_metal']:                        filename = 'ismetal.csv'         ;p=2;num_T  = 55391

    df     = pd.read_csv(f'DATA/properties-reference/{filename}',names=['material_id','value']).replace(to_replace='None',value=np.nan).dropna()

    # CGCNN
    if source == 'CGCNN':
        # SAVING THE PROPERTIES SEPARATELY

        if filename in ['bulkmodulus.csv','shearmodulus.csv','poissonratio.csv']:
            small  = pd.read_csv(f'DATA/cgcnn-reference/mp-ids-3402.csv' ,names=['mp_ids']).values.squeeze()
            df     = df[df.material_id.isin(small)]
            num_T  = 2041
        elif filename == 'bandgap.csv':
            medium = pd.read_csv(f'DATA/cgcnn-reference/mp-ids-27430.csv',names=['mp_ids']).values.squeeze()
            df     = df[df.material_id.isin(medium)]
            num_T  = 16458
        elif filename in ['formationenergy.csv','fermienergy.csv','ismetal.csv','absoluteenergy.csv']:
            large  = pd.read_csv(f'DATA/cgcnn-reference/mp-ids-46744.csv',names=['mp_ids']).values.squeeze()
            df     = df[df.material_id.isin(large)]
            num_T  = 28046
        CIF_dict   = {'radius':8,'step':0.2,'max_num_nbr':12}

    # MEGNET
    elif source  == 'MEGNET':
        megnet_df = pd.read_csv('DATA/megnet-reference/megnet.csv')
        use_ids   = megnet_df[megnet_df.iloc[:,p]==1].material_id.values.squeeze()
        df        = df[df.material_id.isin(use_ids)]
        CIF_dict  = {'radius':4,'step':0.5,'max_num_nbr':16}

    # ADDITIONAL CLEANING
    if p in [3,4]:
        df        = df[df.value>0]
    df.to_csv('DATA/CIF-DATA/id_prop.csv',index=False,header=False)
    print(f'> Dataset for {source}---{property_name} ready !\n\n')
    return source,num_T,CIF_dict
