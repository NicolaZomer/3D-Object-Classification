# 3D-Object-Classification

Nowadays, the identification and understanding of 3D objects in real-world environments has a wide range of applications, including robotics and human-computer interaction. This is typically addressed using Deep Learning techniques that deal with 3D volumetric data, as they are generally able to outperform standard Machine Learning tools.
 
 
In this work, we experiment with several architectures based on Convolutional Neural Networks with the aim of classifying 3D objects. We run our tests on the ModelNet40 dataset, one of the most popular benchmark in the context of 3D object recognition. 

First we compare the effectiveness of Point Clouds and Voxel grids, inspecting pros and cons of these representations. We see how, for instance, the more accurate representation obtained via PC does not lead to better performance when dealing with CNNs, unless you have very large memory capacities. 

Then, we build an Autoencoder in order to retrieve an high-dimensional embedding of the input data. We show that the application of simple ML techniques, such as SVM, on these intermediate representations can lead to state-of-the-art performances and codewords could be used for compression purposes. Finally, we provide a visual representation of the encoded features through t-SNE.
 
 
 
## Dataset 
We employ the [ModelNet40](https://modelnet.cs.princeton.edu/)[1] dataset for our experiments. The dataset contains 12,311 3D models of 40 categories. The models are represented as point clouds with 3D coordinates and 3D normals.
Example of the dataset:

<p align="float">
  <img src="imgs/meshes/plant.png" width="100" title="Task">
  <img src="imgs/meshes/bottle.png" width="80" title="Task">
    <img src="imgs/meshes/car_2.png" width="180" title="Task">
  <img src="imgs/meshes/stairs.png" width="180" title="Task">

</p>

### Binvox [3][4]
Voxels indices are listed in the ModelNet40 as .off files, in order to process voxels as 3D volumes we apply a binvox[3] conversion.
The binvox conversion is done using the following command:
```
cd binvox_utils
chmod +x binvox
python3.9 off2binvox.py
```
Note that the binary file binvox in the binvox_utils folder is compiled for OSX, if you are using a different OS you need to compile it from the source code, available [here](https://www.patrickmin.com/binvox/).

### Pipeline
A generic example of the pipeline is shown below:

<p align="center">
  <img src="imgs/nets/pipeline.png" width="1000" title="Task">
>

## Architectures
We test two main architectures for 3D object classification: one based on PointCloud data and one based on Voxel data.
Moreover, we test a further approach, based on autoencoder reconstruction of the input point cloud.

### 1. PointNet
The architecture of PointNet is shown below:

<p align="center">
  <img src="imgs/nets/pnet.png" width="1000" title="PointNet">
</p>


### 2. VoxelNet
Another approach is to use the voxel representation of the 3D models. We use the binvox conversion to convert the .off files into .binvox files. The architecture of VoxelNet is based on convolutional layers, as shown below:
<p align="center">
  <img src="imgs/nets/vnet.png" width="1000" title="voxnet">
</p>


All results are shown in the following table:

<p align="center">
  <img src="imgs/results/supervised.png" width="450" title="results">
</p>

### 3. Autoencoder reconstruction
We train an autoencoder to reconstruct both an input point cloud or a voxel grid.

#### 3.1. Point cloud autoencoder
The architecture of the autoencoder is inspired from Folding Net [6]. The codewords are used as features for the classification task, which can be performed using a simple MLP or SVM.

For visualization purposes, we use T-SNE to reduce the dimensionality of the codewords to 2D. The following figure shows the T-SNE visualization of the codewords of the autoencoder trained on the ModelNet40 dataset:

<p align="center">
  <img src="imgs/tsne_foldingnet.png" width="400" title="tsne">
</p>

#### 3.2. Voxel grid autoencoder
The voxel-based autoencoder is a simple 3D CNN, with the same architecture of the VoxelNet. The results are way worse than the point cloud autoencoder, probably due to the fact that the voxel grid is not a good representation of the 3D shape.


#### 3.3. Results
The results of the autoencoder are shown in the following table:

<p align="center">
  <img src="imgs/results/unsupervised.png" width="750" title="results">
</p>



## References
[1]. K. F. Y. L. Z. X. T. Z. Wu, S. Song and J. Xiao, 3D ShapeNets: A Deep Representation for Volumetric Shapes. Proceedings of 28th IEEE, Conference on Computer Vision and Pattern Recognition (CVPR2015).

[2] Q.-Y. Zhou, J. Park, and V. Koltun, ???Open3D: A modern library for 3D data processing,??? arXiv:1801.09847, 2018.

[3] P. Min, ???binvox.??? http://www.patrickmin.com/binvox or https://www.google.com/search?q=binvox, 2004 - 2019. 

[4] F. S. Nooruddin and G. Turk, ???Simplification and repair of polygonal models using volumetric techniques,??? IEEE Transactions on Visualization and Computer Graphics, vol. 9, no. 2, pp. 191???205, 2003.

[5] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, ???Pointnet: Deep learning on point sets for 3d classification and segmentation,??? CoRR, vol. abs/1612.00593, 2016.

[6] Yang Yaoqing, Feng Chen, Shen Yiru and Tian Dong, *FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation*, on Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences

## inspiring projects
**pointnet2**: https://github.com/charlesq34/pointnet2
