
import tensorflow as tf
import pickle
import problem_unittests as tests
import helper
import time
from datetime import timedelta
from p2_functions import *#neural_net_image_input, neural_net_label_input, neural_net_keep_prob_input, conv_net

# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load( open('.\cifar-10-batches-py\preprocess_validation.p', mode='rb') )

#%%
# Build the Neural Network
# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()
train_phase = tf.placeholder(tf.bool, name='train_phase')

# Model
logits = conv_net(x, keep_prob, train_phase)

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

import os
directory = 'training_progress'
if not os.path.exists(directory):
    os.makedirs(directory)
	
#%% Tune Parameters and run in a single batch of the data
epochs = 1
batch_size = 64*4
keep_probability = 0.5

print('Checking the Training on a Single Batch...')
n_batches = 1
shuffle_data = True
                            
train_neural_network_full(optimizer, cost, accuracy, x, y, keep_prob,
                          keep_probability, 
                          n_batches, batch_size, shuffle_data, train_phase,
                          valid_features, valid_labels,
                          epochs)

#%% Run in all the batches

save_model_path = './training_progress/saved_progress'
epochs = 1
batch_size = 64*4
keep_probability = 0.5
n_batches = 5
shuffle_data = True

train_neural_network_full(optimizer, cost, accuracy, x, y, keep_prob,
                          keep_probability, 
                          n_batches, batch_size, shuffle_data, train_phase,
                          valid_features, valid_labels,
                          epochs, load_data = False, file_path = save_model_path)

test_model( batch_size )



#%% Continue a training
if 0:
    #%%
    
    save_model_path = './training_progress/saved_progress'
    epochs = 10
    batch_size = 64*4
    keep_probability = 0.5
    n_batches = 5
    shuffle_data = True
    
    train_neural_network_full(optimizer, cost, accuracy, x, y, keep_prob,
                              keep_probability, 
                              n_batches, batch_size, shuffle_data, train_phase,
                              valid_features, valid_labels,
                              epochs, load_data = True, file_path = save_model_path)
        
    test_model( batch_size )




