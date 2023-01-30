# 3D-Object-Classification
 
## Dataset 
We employ the [ModelNet40](https://modelnet.cs.princeton.edu/)[1] dataset for our experiments. The dataset contains 12,311 3D models of 40 categories. The models are represented as point clouds with 3D coordinates and 3D normals.
Example of the dataset:

<p align="float">
  <img src="imgs/meshes/plant.png" width="100" title="Task">
  <img src="imgs/meshes/bottle.png" width="100" title="Task">
    <img src="imgs/meshes/car_2.png" width="100" title="Task">
  <img src="imgs/meshes/stairs.png" width="100" title="Task">

</p>

### Binvox [3][4]
Voxels 

## Architectures

### 1. PointNet
The architecture of PointNet is shown below:

<p align="center">
  <img src="imgs/pnet.png" width="200" title="PointNet">
</p>

### 2. VoxelNet


## References
[1]. K. F. Y. L. Z. X. T. Z. Wu, S. Song and J. Xiao, 3D ShapeNets: A Deep Representation for Volumetric Shapes. Proceedings of 28th IEEE, Conference on Computer Vision and Pattern Recognition (CVPR2015).
[2] Q.-Y. Zhou, J. Park, and V. Koltun, “Open3D: A modern library for 3D data processing,” arXiv:1801.09847, 2018.
[3] P. Min, “binvox.” http://www.patrickmin.com/binvox or https://www.google.com/search?q=binvox, 2004 - 2019. 
[4] F. S. Nooruddin and G. Turk, “Simplification and repair of polygonal models using volumetric techniques,” IEEE Transactions on Visualization and Computer Graphics, vol. 9, no. 2, pp. 191–205, 2003.
[5] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” CoRR, vol. abs/1612.00593, 2016.

