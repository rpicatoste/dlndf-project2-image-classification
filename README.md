# Deep Learning Nanodegree Foundations - Project 2 - Image Classification

The notebook can be run directly and should do the whole process.

The code attached can be run as well, I used spyder to write it, and running p2_main.py the whole project will run.

I added shuffling each time an epoch is finished, and the improvement was noticeable. 

I tried dropout (there is a separate html with the result), without noticing a big improvement. 
In fact, to get the same level of accuracy as without it, I had to run 50 extra epochs. 
Even increasing the network (in the lessons it is mentioned that if dropout is not working for you, you should consider
increasing your network), I only noticed that it get faster to a similar performance (approx 71 %).

# Resubmission notes

I corrected the use of the function to print the stats, which was the most important in order to print the test performance. 
I had also to update the function since the test data is much bigger than the batches used during training.

I started using properly the dropout, and also batch normalization as suggested. 

Any further advise is always welcome!