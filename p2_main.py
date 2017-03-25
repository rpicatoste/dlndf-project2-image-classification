
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