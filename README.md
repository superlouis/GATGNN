# GATGNN

This software package implements our developed model GATGNN for improved inorganic materials' property prediction. This is the official Pytorch repository.

Property | MAE Performance of our model| Units
------------ | ------------- | -------------
Formation Energy | 0.039 | eV/atom
Absolute Energy | 0.048 | eV/atom
Fermi Energy | 0.33 | eV/atom
Band Gap | 0.31 | eV
Bulk-Moduli | 0.045 | log(GPa)
Shear-Moduli | 0.075 | log(GPa)
Poisson-ratio | 0.029 | -

The package provides 3 major functions:

- Train a GATGNN model for either of the 7 properties referenced above.
- Evaluate the performance of a trained GATGNN model on either of the 7 properties referenced above.
- Predict the property of a given material using its cif file. 

The following paper describes the details of the our framework:
[GLOBAL ATTENTION BASED GRAPH CONVOLUTIONAL NEURAL NETWORKS FOR IMPROVED MATERIALS PROPERTY PREDICTION](https://arxiv.org/pdf/2003.13379.pdf)

Machine Learning and Evolution Laboratory, University of South Carolina <br />
How to cite:<br />
Steph-Yves Louis, Yong Zhao, Alireza Nasiri, Xiran Wong, Yuqi Song, Fei Liu, and Jianjun Hu. "Global Attention based Graph Convolutional Neural Networks for Improved Materials Property Prediction." arXiv preprint arXiv:2003.13379 (2020).

![](front-pic.png)

## Installation
Install any of the relevant packages if not already installed:
* Pytorch (tested on 1.2.0) - preferably version 1.2.0 or later
* Numpy   (tested on 1.17.3)
* Pandas  (tested on 0.21.3) 
* Scikit-learn (tested on 0.21.3) 
* Pytmatgen (tested on 2019.10.16)
* PyTorch-Geometric (tested on 1.1.2)

- Pytorch, Numpy, Pandas, Scikit-learn, and Pymatgen
```bash
pip install torch torchvision 
pip install numpy
pip install pandas
pip install scikit-learn
pip install pymatgen
```
- PyTorch Geometric 
```bash
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```
## Dataset
1. Download the compressed file of our dataset using [this link](https://widgets.figshare.com/articles/12522524/embed?show_title=1)
2. Unzip its content ( a directory named 'DATA')
3. Move the DATA directory in your GATGNN directory. i.e. such that the path GATGNN/DATA now exists.

## Usage
#### Training a new model
Once all the aforementionned requirements are satisfied, one can easily train a new GATGNN by running __train.py__ in the terminal along with the specification of the appropriate flags. At the bare minimum, using --property to specify the property and --data_src to identify the dataset (CGCNN or MEGNET) should be enough to train a robust GATGNN model.
- Example-1. Train a model on the bulk-modulus property using the CGCNN dataset.
```bash
python train.py --property bulk-modulus --data_src CGCNN
```
- Example-2. Train a model on the shear-modulus property using the MEGNET dataset.
```bash
python train.py --property shear-modulus --data_src MEGNET
```
- Example-3. Train a model with 5 layers on the bulk-modulus property using the CGCNN dataset and the global attention technique of fixed cluster unpooling (GI M-2).
```bash
python train.py --property bulk-modulus --data_src CGCNN --num_layers 5 --global_attention cluster --cluster_option fixed
```
The trained model will be automatically saved under the TRAINED directory. *Pay attention to the flags used for they will be needed again to evaluate the model.

#### Evaluating the performance of a trained model
Upon training a GATGNN, one can evaluate its performance using __evaluate.py__ in the terminal exactly the same way as __train.py__. *It is IMPORTANT that one runs __evaluate.py__ with the exact same flags as it was done when prior training the model.*
- Example-1. Evaluate the performance of a model trained on the bulk-modulus property using the CGCNN dataset.
```bash
python evaluate.py --property bulk-modulus --data_src CGCNN
```
- Example-2. Evaluate the performance of a model trained on the shear-modulus property using the MEGNET dataset.
```bash
python evaluate.py --property shear-modulus --data_src MEGNET
```
- Example-3.  Evaluate the performance of a model trained with 5 layers on the bulk-modulus property using the CGCNN dataset and the global attention technique of fixed cluster unpooling (GI M-2).
```bash
python evaluate.py --property bulk-modulus --data_src CGCNN --num_layers 5 --global_attention cluster --cluster_option fixed
```
#### Predicting the property of a single inorganic material using its .cif file
Again, using a trained model one can also predict the property of a single inorganic material using its .cif file. Just follow those 2 steps:
1. Place your .cif file inside the directory DATA/prediction-directory/
1. Run __predict.py__ in a similar fashion as __evaluate.py__ except for the addition of the flag --to_predict which specifies the name of the .cif file.
- Example-1. Predict the bulk-modulus property of a material named mp-1 using the CGCNN graph constructing specifications.
```bash
python predict.py --property bulk-modulus --data_src CGCNN --to_predict mp-1
```
- Example-2. Predict the shear-modulus property of a material named mp-1 using the MEGNET graph constructing specifications.
```bash
python predict.py --property shear-modulus --data_src MEGNET --to_predict mp-1
```
