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

Machine Learning and Evolution Laboratory

University of South Carolina

How to cite:
Louis, Steph-Yves, Yong Zhao, Alireza Nasiri, Xiran Wong, Yuqi Song, Fei Liu, and Jianjun Hu. "Global Attention based Graph Convolutional Neural Networks for Improved Materials Property Prediction." arXiv preprint arXiv:2003.13379 (2020).


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

## Run
Once all the aforementionned requirements are satisfied, one can easily run our module using our train.py and evaluate.py files. As of now, one can readily train and evaluate a model for any of the 7 properties for which we reported our state of the performance. *see results section below* <br />*
__1. train.py__  Use it to train a new model on a given property. <br />*
__2. evaluate.py__ Use it to evaluate a trained model's performance on a given property. <br />*

#### -Instructions to train a new model
a) using your terminal, provide the arguments of 
