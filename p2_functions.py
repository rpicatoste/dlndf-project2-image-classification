
import problem_unittests as tests
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import time
from datetime import timedelta
import helper

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    x_norm = x.reshape(x.size)
    
    x_max = max(x_norm)
    x_min = min(x_norm)
    x_range = x_max-x_min
    x_norm = (x_norm - x_min)/(x_range)    
    
    return x_norm.reshape(x.shape)



lb = LabelBinarizer()
lb_initialized = False

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    global lb_initialized
    if(lb_initialized):
        pass
    else:    
        lb.fit(x)
        lb_initialized = True
    
    return lb.transform(x)



# Unit testing
tests.test_normalize(normalize)
tests.test_one_hot_encode(one_hot_encode)



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

    # Shape of the filter-weights for the convolution.
    input_channels = x_tensor.shape[3].value
    shape = [conv_ksize[0], conv_ksize[1], input_channels , conv_num_outputs]

    # Create new weights aka. filters with the given shape.
    weights = tf.Variable( tf.truncated_normal(shape, stddev = 0.05), name = 'weights' )

    # Create new biases, one for each filter.
    biases = tf.Variable( tf.constant(0.05, shape = [conv_num_outputs]), name = 'biases' )

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input = x_tensor,
                         filter = weights,
                         strides = [1, conv_strides[0], conv_strides[1], 1],
                         padding = 'SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer = tf.nn.bias_add(layer, biases)
#    layer += biases

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

    # Create new weights and biases.
    num_inputs = x_tensor.shape[1].value
    weights = tf.Variable( tf.truncated_normal([num_inputs, num_outputs], stddev = 0.05), name = 'weights' )
    biases = tf.Variable( tf.constant(0.05, shape = [num_outputs]), name = 'biases' )
    
    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(x_tensor, weights) + biases

    return layer

tests.test_fully_conn(fully_conn)


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return fully_conn(x_tensor, num_outputs)


tests.test_output(output)


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    conv_1 =    conv2d_maxpool(x,       16,         [3,3],      [1,1],          [1,1],      [1,1])
    conv_1 =    tf.nn.dropout( conv_1, keep_prob )
    conv_2 =    conv2d_maxpool(conv_1,  32,         [3,3],      [1,1],          [2,2],      [2,2])
    conv_2 =    tf.nn.dropout( conv_2, keep_prob )
    conv_last = conv2d_maxpool(conv_2,  64,         [5,5],      [2,2],          [2,2],      [2,2])
    conv_last = tf.nn.dropout( conv_last, keep_prob )
    # Apply a Flatten Layer
    x_flat = flatten( conv_last )

    # Apply 1, 2, or 3 Fully Connected Layers
    # Play around with different number of outputs
    full_1 =    fully_conn(x_flat, 256)
    full_1 =    tf.nn.dropout( full_1, keep_prob )
    full_last = fully_conn( full_1, 64)
    full_last = tf.nn.dropout( full_last, keep_prob )

#    full_last = fully_conn( x_flat, 256)
#    full_last = tf.nn.dropout( full_last, keep_prob )
    
    
    # Apply an Output Layer
    # Set this to the number of classes
    # TODO: 10 is hard coded... pass somehow
    output_layer = output( full_last, 10 )
    
    return output_layer


# Train once
def train_neural_network_once(session, optimizer, keep_probability, feature_batch, label_batch, x, y, keep_prob):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    
    dict_feed_train = { x: feature_batch, y: label_batch, keep_prob: keep_probability }
    session.run(optimizer, feed_dict = dict_feed_train)
    
def print_stats(session, feature_batch, label_batch, cost, accuracy, x, y, keep_prob):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    feed_dict_fwd = { x: feature_batch, y: label_batch, keep_prob: 1.0 }
    
    
    cost = session.run(cost, feed_dict = feed_dict_fwd)
    acc = session.run(accuracy, feed_dict = feed_dict_fwd)
    
    print('Cost: {:05.4}, Acc: {:.1%}'.format(cost, acc) )


def train_neural_network_full(optimizer, cost, accuracy, x, y, keep_prob, keep_probability, n_batches, batch_size, shuffle_data, epochs, load_data = False, file_path = './training_progress/saved_progress'):
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initializing the variables
        if load_data:
            print('Continue a started training...')
            saver.restore(sess, file_path)
        else:
            print('Starting training...')
            sess.run(tf.global_variables_initializer())
        
        start_time = time.time()
        
        # Training cycle
        for epoch in range(epochs):
            
            for batch_i in range(1, n_batches + 1):
                start_time_batch = time.time()
                
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size, shuffle_data = True):
                    
                    train_neural_network_once(sess, optimizer, keep_probability, batch_features, batch_labels, x, y, keep_prob)
                    
                end_time_batch = time.time()
                time_dif = str(timedelta(seconds = int(round(end_time_batch - start_time_batch))))
                print('Epoch {:>2}, time  {} sec, CIFAR-10 Batch {}:  '.format(epoch + 1, time_dif, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy, x, y, keep_prob)        
    
        # Print the time-usage.
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds = int(round(time_dif)))))
        
        save_path = saver.save(sess, file_path)
 
        
        