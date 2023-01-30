# 3D-Object-Classification
 
## Dataset 
We employ the [ModelNet40](https://modelnet.cs.princeton.edu/) dataset for our experiments. The dataset contains 12,311 3D models of 40 categories. The models are represented as point clouds with 3D coordinates and 3D normals.
Example of the dataset:

<p align="float">
  <img src="imgs/plant.png" width="200" title="Task">
  <img src="imgs/bottle.png" width="200" title="Task">
</p>


## Architectures

### 1. PointNet
The architecture of PointNet is shown below:

<p align="center">
  <img src="imgs/pnet.png" width="200" title="PointNet">
</p>

### 2. VoxelNet