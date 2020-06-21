# GATGNN
Official Pytorch repository for our paper *[GLOBAL ATTENTION BASED GRAPH CONVOLUTIONAL NEURAL NETWORKS FOR IMPROVED MATERIALS PROPERTY PREDICTION](https://arxiv.org/pdf/2003.13379.pdf).<br />*
by [Steph-Yves Louis](http://mleg.cse.sc.edu/people.html), et. al. ... 
![](front-pic.png)

## Installation
Please install those relevant packages if not already installed:
* Pytorch (tested on 1.2.0) - preferably version 1.2.0 or later
* Numpy   (tested on 1.17.3)
* Pandas  (tested on 0.21.3) 
* Scikit-learn (tested on 0.21.3) 
* Pytmatgen (tested on 2019.10.16)
* PyTorch-Geometric (tested on 1.1.2)


- Install Pytorch, Numpy, Pandas, Scikit-learn, Pymatgen, and Pytorch-Geometric
```bash
pip install torch torchvision 
pip install numpy
pip install pandas
pip install scikit-learn
pip install pymatgen
```
- Install PyTorch Geometric (tested on 1.1.2), please refer to the offical website for further details
```bash
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```
- Install networkx (tested on 2.3), make sure you are not using networkx 1.x version!
```bash
pip install networkx
