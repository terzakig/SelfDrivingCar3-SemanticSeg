import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


IMG_HEIGHT = 160
IMG_WIDTH = 576

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = sess.graph
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    prob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        
    
    return  input_tensor, prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)




def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  
    Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    
    lambda_ = 0.001 # a global regularization factor
    # Layer 75
    # NOTE: This is is something I saw in #    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')the walkthrough.
    # For some reason, we reduce the outputs to 2 right off the bat,
    # (whereas in my original conception, I did this at the very last layer...
    # Anyways, this way the network is much more lighter, as we progress 
    # with only two channels deconvolution after deconvolution layer).
    # 
    # So this is a conv. layer "7.5" which outputs 5x18(x2=num_channels)
    kernel_size_75 = (1, 1) # small kernel as this ia a small image
    strides_75 = (1, 1)
    lambda_75 = lambda_ # regularization factor
    layer_75 = tf.layers.conv2d(vgg_layer7_out,
                                 filters = num_classes,      # reducing channels to num_classes!!!!!!
                                 kernel_size = kernel_size_75,
                                 strides = strides_75,
                                 padding = 'SAME',
                                 trainable = True,
                                 kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_75)
                                 )
    # Layer 45
    # NOTE: This is a simila 1x1 convolutiion on the layer 4 max-pooled output \
    # to match the number of channels with num_classes (i.e., 2)
    # So this is a conv. layer "4.5" which outputs 10x36(x2=num_channels)
    kernel_size_45 = (1, 1) # small kernel as this ia a small image
    strides_45 = (1, 1)
    lambda_45 = lambda_ # regularization factor
    layer_45 = tf.layers.conv2d(vgg_layer4_out,
                                 filters = num_classes,      # reducing channels to num_classes!!!!!!
                                 kernel_size = kernel_size_45,
                                 strides = strides_45,
                                 padding = 'SAME',
                                 trainable = True,
                                 kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_45)
                                 )
    
    # Layer 35
    # NOTE: This is a simila 1x1 convolutiion on the layer 4 max-pooled output \
    # to match the number of channels with num_classes (i.e., 2)
    # So this is a conv. layer "4.5" which outputs 20x72(x2=num_channels)
    kernel_size_35 = (1, 1) # small kernel as this ia a small image
    strides_35 = (1, 1)
    lambda_35 = 0.001 # regularization factor
    layer_35 = tf.layers.conv2d(vgg_layer3_out,
                                 filters = num_classes,      # reducing channels to num_classes!!!!!!
                                 kernel_size = kernel_size_35,
                                 strides = strides_35,
                                 padding = 'SAME',
                                 trainable = True,
                                 kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_35)
                                 )
    
    # Layer 8
    # NOTE: The 7.5 layer dumps out a 5x18(x2) image.
    #       Thus, to upscale we do a (trainable) transpose convolution.
    #       Strides: (2, 2) to upscale by a factor of 2
    #
    # New output: 10x36(x2)
    kernel_size_8 = (4, 4) # small kernel as this ia a small image
    strides_8 = (2, 2)     # 2x2 strides will double the resolution
    lambda_8 = lambda_ # regularization factor
    layer_8 = tf.layers.conv2d_transpose(layer_75,
                                         filters = num_classes,      # reducing channels by /2
                                         kernel_size = kernel_size_8,
                                         strides = strides_8,
                                         padding = 'SAME',
                                         trainable = True,
                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_8)
                                         )
    
    # Layer #9
    # Input shape to layer #9: 10x36(x2)
    # NOTE: This layer dumps out an exact match for the (first two dimensions) output of layer #4
    #
    # New Output: 20x72(x2)
    kernel_size_9 = (4, 4) # small kernel size again...
    strides_9 = (2, 2) # upsample again
    lambda_9 = lambda_ # regularization factor
    combined_input_9 = tf.add(layer_8, layer_45)
    layer_9 = tf.layers.conv2d_transpose(combined_input_9,
                                         filters = num_classes,
                                         kernel_size = kernel_size_9,
                                         strides = strides_9,
                                         padding = 'SAME',
                                         trainable = True,
                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_9)
                                         )
    # Layer #10
    # Input to layer #10: 20x72(x256)
    # NOTE: The size of the layer matches EXACTLY layer #3 output
    # Upsampling by 2
    #
#    # New Output: 40x144(x2)
#    kernel_size_10 = (4, 4) # increase the kernel a bit...
#    strides_10 = (2, 2)
#    lambda_10 = 0.001 # regularization factor
#    combined_input_10 = layer_10 + layer_35
#    layer_10= tf.layers.conv2d_transpose(combined_input_10,
#                                         filters = num_classes,
#                                         kernel_size = kernel_size_10,
#                                         strides = strides_10,
#                                         padding = 'SAME',
#                                         trainable = True,
#                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_10)
#                                         )
#    # Layer #11 Input size: 80x288(x128)
#    # 
#    # New Output: 80x288(x64)
#    kernel_size_11 = (6, 6)
#    strides_11 = (2, 2)
#    lambda_11 = 0.001 # regularization factor
#    layer_11 = tf.layers.conv2d_transpose(layer_10,
#                                         filters = num_classes,
#                                         kernel_size = kernel_size_11,
#                                         strides = strides_11,
#                                         padding = 'SAME',
#                                         trainable = True,
#                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_11)
#                                         )
#    
#    # Layer #12 Input size: 80x288(x2)
#    # 
#    # New Output: 160x576(x 2)
#    kernel_size_12 = (10, 10)
#    strides_12 = (2, 2)
#    lambda_12 = 0.001 # regularization factor
#    rectangular_logits = tf.layers.conv2d_transpose(layer_11 ,
#                                         filters = num_classes,
#                                         kernel_size = kernel_size_12,
#                                         strides = strides_12,
#                                         padding = 'SAME',
#                                         trainable = True,
#                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_12)
#                                         )
    
    # NOTE: Now we do the layer #3 skipping and then upscale by 4 to get the actiual image size
    #       It would be more prudent to this in successive de-convolutions, but that
    #       would take forever to train...
    #    # New Output: 20*2*2*2x72*2*2*2(x2)
    kernel_size_out = (16, 16) # increase the kernel a lot (works better)...
    strides_out = (8, 8)
    lambda_out = lambda_ # regularization factor
    combined_input_out = tf.add(layer_9, layer_35)
    rectangular_logits= tf.layers.conv2d_transpose(combined_input_out,
                                         filters = num_classes,
                                         kernel_size = kernel_size_out,
                                         strides = strides_out,
                                         padding = 'SAME',
                                         trainable = True,
                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(lambda_out)
                                         )
    
            
    return rectangular_logits
tests.test_layers(layers)




def optimize1(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    

    # reshape the output layer into  a flat tensor and call it "logits" (=once again)
    logits_ = tf.reshape(nn_last_layer, (-1, num_classes))
    # Do the same for the ground-truth tensor
    labels_ = tf.to_float( tf.reshape(correct_label, (-1, num_classes)) )
    
    epsilon = tf.constant(1e-4) #spare the logarithm from possible zeros
    softmax = tf.nn.softmax(logits_) + epsilon
    
    cross_entropy = -tf.reduce_sum( labels_ * tf.log(softmax), reduction_indices=[1] )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    #tf.add_to_collection('losses', cross_entropy_mean)
    loss_operation = cross_entropy_mean
    #loss_operation = tf.add_n(tf.get_collection('losses'), name='total_loss')

    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    
    return logits_, training_operation, loss_operation
tests.test_optimize(optimize1)
#

def optimize2(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    

    # reshape the output layer into  a flat tensor and call it "logits" (=once again)
    logits_ = tf.reshape(nn_last_layer, (-1, num_classes))
    # Do the same for the ground-truth tensor
    labels_ = tf.reshape(correct_label, (-1, num_classes))
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits_, labels = labels_)
    loss_operation = tf.reduce_mean(cross_entropy)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    training_operation = optimizer.minimize(loss_operation)
    
    return logits_, training_operation, loss_operation
tests.test_optimize(optimize2)
#

#
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
  
    # running the generator once just to get the batch number
    batches = get_batches_fn(batch_size)
    num_batches = len(list(batches))
        
    for epoch in range(epochs):
        print("Training epoch : ", epoch)    
        batch_index = 0;
        for (X_train, Y_train) in get_batches_fn(batch_size):
            # train
            print("Processing batch : ", batch_index + 1, " of " , num_batches)
            sess.run(train_op, {input_image: X_train, keep_prob: 0.5, correct_label: Y_train})
            print("Done...")
            batch_index += 1
            # print loss in the last bacth in this epoch
            if (batch_index % 5 == 0):
                print("Average cross-entropy error in last batch: ", sess.run(cross_entropy_loss, {input_image: X_train, keep_prob: 1.0, correct_label: Y_train}) )
                
        batch_index = 0
        # clear the respective collection in the graph (i.e., the average entropy losses of the images in the batch)
        tf.get_collection('losses').clear()
        
tests.test_train_nn(train_nn)
#
#
def run():
    tf.reset_default_graph()
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
        
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
    
    epochs = 10
    learning_rate = 0.00007
    batch_size = 8
    
    with tf.Session() as sess:
        
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        #
        # 1. Load vgg
        input_tensor, prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor = load_vgg(sess, vgg_path)
    
        # 2. Get the tensor of the last layer (rectangular logits) 
        nn_last_layer_tensor = layers(layer3_out_tensor, layer4_out_tensor, layer7_out_tensor, num_classes)
        
        
        # 3. Get the cross-entropy error and training operations
        # create the correct_label placeholder
        correct_label = tf.placeholder(dtype = tf.int8, shape = (None, None, None, 2))
        logits_, train_op, cross_entropy_loss = optimize1(nn_last_layer_tensor , correct_label, learning_rate, num_classes)
        
        sess.run(tf.global_variables_initializer() )
#        batch_x, batch_y = generator.__next__()
#        print("Input shape ", batch_y.shape)
#        sess.run(tf.global_variables_initializer())
#        print("Layer #3 output shape ", sess.run(nn_last_layer_tensor, {input_tensor: [batch_x[0]], prob_tensor: 1.0} ).shape)
#        print("Layer #4 output shape ", sess.run(layer4_out_tensor, {input_tensor: [batch_x[0]], prob_tensor: 1.0} ).shape)
#        print("Layer #7 output shape ", sess.run(layer7_out_tensor, {input_tensor: [batch_x[0]], prob_tensor: 1.0} ).shape)
#        
        #Train NN using the train_nn function
        
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_tensor,
             correct_label, prob_tensor, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir='data/data_road/testing', 
                                      data_dir=data_dir, 
                                      sess=sess, 
                                      image_shape = image_shape, 
                                      logits = logits_, 
                                      keep_prob = prob_tensor, 
                                      input_image = input_tensor)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
