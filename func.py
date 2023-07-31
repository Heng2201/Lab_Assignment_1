import numpy as np

# this function is to reshape the 2-D image into 1-D image
def resize_img(x):
    return np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

# this function implements one hot encoding 
def one_hot(x):
    one_hot = np.zeros((x.shape[0], 10))
    one_hot[np.arange(x.shape[0]), x] = 1

    return one_hot
