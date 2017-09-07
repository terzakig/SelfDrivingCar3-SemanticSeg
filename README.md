# SelfDriving Car: Road Terrain Classification with Fully Convolutional Neural Networks (Semantic Segmentation)

## Description
This is a fully convolutional neural network based on the VGG network. The purpose of the network is to identify road pixels in images. The encoder part comprises the 7 pre-trained layers of the [VGG net](https://arxiv.org/pdf/1409.1556.pdf) and the decoder  uses successive transposed convolutions with direct convolutional layer connections (to drop the number of output channels from many to 2) from encoding layers 3 and 4. 

The training (and test) data come from the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_road.php) dataset and can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip). Extract the archive in the `data` directory.

## Improvements
A smoothness prior shoould be imposed and tensorflow provides a very convenient way of doing this using `tf.image.total_variation`. The problem is that it requires tuning of the regularization factor and this network was trained in a regular CPU, so experimentation can potentially take up entire days...
