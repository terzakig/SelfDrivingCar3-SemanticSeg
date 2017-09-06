# SelfDriving Car: Road Terrain Classification with Fully Convolutional Neural Networks (Semantic Segmentation)

## Description
This is a fully convolutional neural network based on the VGG network. The purpose of the network is to identify road pixels in images. The encoder part ccomproses the 7 pre-trained VGG layers and the decide is built with successive transposed convolutions with direct convolutional connections from encoding layers 3 and 4. 

## Improvements
A smoothness prior shoould be imposed and tensorflow provides a very convenient way of doing this using `tf.image.total_variation`. The problem is that it requires tuning and this network was trained in a regular CPU, so experimentation can take up entire days...
