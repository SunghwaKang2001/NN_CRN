import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import utils

def downsampled_mnist() : 
    mnist = fetch_openml('mnist_784')
    N = 14780-1000
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    selected_indices = np.where((y == 0) | (y == 1))
    x = np.array(x)[selected_indices]
    y = np.array(y)[selected_indices]

    label_x = utils.downscale_images(x[:N])
    label_x = label_x.reshape(label_x.shape[0],-1)
    label_y = utils.one_hot_encoding(y[:N])

    test_x = utils.downscale_images(x[N:])
    test_x = test_x.reshape(test_x.shape[0],-1)
    test_y = utils.one_hot_encoding(y[N:])

    return label_x, label_y, test_x, test_y, N

def iris() : 
    iris = datasets.load_iris()
    enc = OneHotEncoder()
    label_x, test_x, label_y, test_y = train_test_split(iris.data, enc.fit_transform(iris.target.reshape(-1,1)).toarray(), test_size=0.1)
    N = len(label_x)
    return label_x, label_y, test_x, test_y, N

def half_sine() : 
    N=13
    label_y = [[np.sin(i/4)] for i in range(13)]
    label_x = [[i/4] for i in range(13)]
    test_x = [[(2*i+1)/8] for i in range(12)]
    test_y = [[np.sin((2*i+1)/8)] for i in range(12)]
    return label_x, label_y, test_x, test_y, N

def XOR() : 
    N=4
    label_x = [[0,0],[0,1],[1,0],[1,1]]
    label_y = [[0],[1],[1],[0]]
    test_x = label_x
    test_y = label_y
    return label_x, label_y, test_x, test_y, N
