
import problem_unittests as tests
import numpy as np
from sklearn.preprocessing import LabelBinarizer

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



import problem_unittests as tests
# Unit testing
tests.test_normalize(normalize)
tests.test_one_hot_encode(one_hot_encode)