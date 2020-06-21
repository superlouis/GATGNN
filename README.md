# GATGNN
Official Pytorch repository for our paper *[GLOBAL ATTENTION BASED GRAPH CONVOLUTIONAL NEURAL NETWORKS FOR IMPROVED MATERIALS PROPERTY PREDICTION](https://arxiv.org/pdf/2003.13379.pdf).<br />*
by [Steph-Yves Louis](http://mleg.cse.sc.edu/people.html), et. al. ... 
![](front-pic.png)

## Installation
- Install PyTorch (tested on 1.2.0), preferably version 1.2.0 or later
```bash
pip install torch torchvision 
```
- Install Numpy (tested on 1.17.3) 
```bash
pip install numpy
```
- Install Pandas (tested on 0.21.3) 
```bash
pip install pandas
```
- Install Scikit-learn (tested on 0.21.3) 
```bash
pip install scikit-learn
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
