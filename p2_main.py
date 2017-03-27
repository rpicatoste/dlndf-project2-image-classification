
import tensorflow as tf
import pickle
import problem_unittests as tests
import helper
import time
from datetime import timedelta
from p2_functions import *#neural_net_image_input, neural_net_label_input, neural_net_keep_prob_input, conv_net

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load( open('.\cifar-10-batches-py\preprocess_validation.p', mode='rb') )

# Build the Neural Network
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
tests.test_train_nn(train_neural_network_once, x, y, keep_prob)

#%% Tune Parameters and run in a single batch of the data
epochs = 2
batch_size = 64*4
keep_probability = 1.0

print('Checking the Training on a Single Batch...')
n_batches = 1
shuffle_data = True

train_neural_network_full(optimizer, cost, accuracy, x, y, keep_prob, keep_probability, n_batches, batch_size, shuffle_data, epochs)


#%% Run in all the batches

save_model_path = './training_progress/saved_progress'
epochs = 2
batch_size = 64*4
keep_probability = 1.0
n_batches = 5
shuffle_data = True
train_neural_network_full(optimizer, cost, accuracy, x, y, keep_prob, keep_probability, n_batches, batch_size, shuffle_data, epochs, False, save_model_path)



#%% Continue a training
if 0:
    #%%
    
    save_model_path = './training_progress/saved_progress'
    epochs = 2
    batch_size = 64*4
    keep_probability = 1.0
    n_batches = 5
    shuffle_data = True
    train_neural_network_full(optimizer, cost, accuracy, x, y, keep_prob, keep_probability, n_batches, batch_size, shuffle_data, epochs, load_data = True, file_path = save_model_path)

        
    
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