# Deep Learning Nanodegree Foundations - Project 2 - Image Classification

The notebook can be run directly and should do the whole process.

The code attached can be run as well, I used spyder to write it, and running p2_main.py the whole project will run.

I added shuffling each time an epoch is finished, and the improvement was noticeable. 

I tried dropout (there is a separate html with the result), without noticing a big improvement. 
In fact, to get the same level of accuracy as without it, I had to run 50 extra epochs. 
Even increasing the network (in the lessons it is mentioned that if dropout is not working for you, you should consider
increasing your network), I only noticed that it get faster to a similar performance (approx 71 %).

# Resubmission 1 notes

I corrected the use of the function to print the stats, which was the most important in order to print the test performance. 
I had also to update the function since the test data is much bigger than the batches used during training.

I started using properly the dropout, and also batch normalization as suggested. 

Any further advise is always welcome!

# Resubmission 2 notes

As requested I swapped relu and pooling. But I would like to emphasize that the reason to have them in the reversed order was because
I followed the next statement from Siraj's live session about this:

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.
    
    
# Resubmission 3 notes

The comment made by reviewer 3 was addressed but he/she missed the fact that I call print_stats with train values between batches to check training progress and with the global variables valid_features and valid_labels when each epoch is finished. This is noted in the "Resubmission 1 notes" of the README.md.

NOTE: I don't use the python "global" word because I prefer to pass values as parameter in the function using it. 
