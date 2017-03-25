
import tensorflow as tf
import pickle
import problem_unittests as tests
import helper
import time
from datetime import timedelta



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
    conv_1 =    conv2d_maxpool(x,       32,         [3,3],      [1,1],          [1,1],      [1,1])
    conv_2 =    conv2d_maxpool(conv_1,  64,         [3,3],      [1,1],          [2,2],      [1,1])
    conv_last = conv2d_maxpool(conv_2,  128,        [5,5],      [2,2],          [2,2],      [1,1])

    # Apply a Flatten Layer
    x_flat = flatten( conv_last )

    # Apply 1, 2, or 3 Fully Connected Layers
    # Play around with different number of outputs
#    full_1 =    fully_conn(x_flat, 128)
    full_last = fully_conn(x_flat, 64)
    
    # Apply an Output Layer
    # Set this to the number of classes
    # TODO: 10 is hard coded... pass somehow
    output_layer = output( full_last, 10 )
    
    return output_layer


##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = y) )
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)

# Train once

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    
    dict_feed_train = { x: feature_batch, y: label_batch }
    session.run(optimizer, feed_dict = dict_feed_train)
    

tests.test_train_nn(train_neural_network)


#%%
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    feed_dict_fwd = { x: feature_batch, y: label_batch}
    
    
    cost = session.run(cost, feed_dict = feed_dict_fwd)
    acc = session.run(accuracy, feed_dict = feed_dict_fwd)
    
    print('Cost: {:05.4}, Acc: {:.1%}'.format(cost, acc) )
    

#session = tf.Session()
#print_stats(session, valid_features, valid_labels, cost, accuracy)
#session.close()


#%% Tune Parameters and run in a single batch of the data
epochs = 10
batch_size = 64
keep_probability = 1.0

print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    start_time = time.time()
    
    # Training cycle
    for epoch in range(epochs):
        start_time_batch = time.time()
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            
        end_time_batch = time.time()
        time_dif = str(timedelta(seconds = int(round(end_time_batch - start_time_batch))))
        print('Epoch {:>2}, time  {} sec, CIFAR-10 Batch {}:  '.format(epoch + 1, time_dif, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)        

    # Print the time-usage.
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds = int(round(time_dif)))))

#%% Run in all the batches

save_model_path = './training_progress/saved_progress'

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

#%% Continue a training
if 0:
    #%%
    save_model_path = './training_progress/saved_progress'
    
    print('Continue a started training...')
    with tf.Session() as sess:
        # Initializing the variables
        saver.restore(sess, save_model_path)
        
        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
                
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)
    
#%% Test the model

import tensorflow as tf
import pickle
import helper
import random

# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './training_progress/saved_progress'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('.\cifar-10-batches-py\preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {:.1%}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model()