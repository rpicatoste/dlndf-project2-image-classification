
import tensorflow as tf
import pickle
import problem_unittests as tests
import helper



# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load( open('.\cifar-10-batches-py\preprocess_validation.p', mode='rb') )

#image_shape  = list( valid_features.shape )


def neural_net_image_input( image_shape ):
    
    image_shape  = list( image_shape )
    image_shape.insert(0, None)
    x = tf.placeholder(tf.float32, image_shape, name = 'x')
    
    return x


def neural_net_label_input( n_classes ):

    y = tf.placeholder(tf.float32, [None, n_classes], name = 'y')
    
    return y
    
def neural_net_keep_prob_input():
    
    return tf.placeholder(tf.float32, name = 'keep_prob')


tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    
#def new_conv_layer(input,              # The previous layer.
#                   num_input_channels, # Num. channels in prev. layer.
#                   filter_size,        # Width and height of each filter.
#                   num_filters,        # Number of filters.
#                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    input_channels = x_tensor.shape[3].value
    shape = [conv_ksize[0], conv_ksize[1], input_channels , conv_num_outputs]

    # Create new weights aka. filters with the given shape.
    weights = tf.Variable( tf.truncated_normal(shape, stddev = 0.05) )

    # Create new biases, one for each filter.
    biases = tf.Variable( tf.constant(0.05, shape = [conv_num_outputs]) )

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input = x_tensor,
                         filter = weights,
                         strides = [1, conv_strides[0], conv_strides[1], 1],
                         padding = 'SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Pooling
    layer = tf.nn.max_pool(value = layer,
                           ksize = [1, pool_ksize[0], pool_ksize[1], 1],
                           strides = [1, pool_strides[0], pool_strides[1], 1],
                           padding = 'SAME')

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer


tests.test_con_pool(conv2d_maxpool)


def flatten( x_tensor ):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    
    # Get the shape of the input layer.
    layer_shape = x_tensor.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]
    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    layer_flat = tf.reshape(x_tensor, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat
    

tests.test_flatten(flatten)

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """

#def new_fc_layer(input,          # The previous layer.
#                 num_inputs,     # Num. inputs from prev. layer.
#                 num_outputs,    # Num. outputs.
#                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    num_inputs = x_tensor.shape[1].value
    weights = tf.Variable( tf.truncated_normal([num_inputs, num_outputs], stddev = 0.05) )
    biases = tf.Variable( tf.constant(0.05, shape = [num_outputs]) )
    
    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(x_tensor, weights) + biases

    return layer

tests.test_fully_conn(fully_conn)
