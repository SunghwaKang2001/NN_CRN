import numpy as np
from skimage.transform import resize

def one_hot_encoding(labels, num_classes=2):
    one_hot_vectors = []
    for label in labels:
        label_int = int(label)
        if label_int < 0 or label_int >= num_classes:
            raise ValueError("Label {} is out of range for the specified number of classes ({}).".format(label_int, num_classes))
        one_hot_vector = [0] * num_classes
        one_hot_vector[label_int] = 1
        one_hot_vectors.append(one_hot_vector)
    return one_hot_vectors

def downscale_images(images):
    downscaled_images = np.zeros((images.shape[0], 8, 8))
    for i in range(images.shape[0]):
        downscaled_images[i] = resize(images[i].reshape(28, 28), (8, 8), anti_aliasing=True)
    return downscaled_images

def mean_error(W1,W2,b1,b2,input,output,H) : 
    error = 0
    for i in range(len(input)) : 
        z1 = np.dot(W1,input[i])+b1
        y1 = (z1+np.sqrt(np.power(z1,2)+4*H))/2 
        z2 = np.dot(W2,y1)+b2
        y2 = (z2+np.sqrt(np.power(z2,2)+4*H))/2 
        error = error+((y2-output[i])**2).sum()
    return error/len(input)

def mean_error_for_leaky(W1,W2,b1,b2,input,output,alpha,beta) : 
    error = 0
    for i in range(len(input)) : 
        z1 = np.dot(W1,input[i])+b1
        y1 = (np.abs(z1)+z1)/2*alpha + (-np.abs(z1)+z1)/2*beta
        z2 = np.dot(W2,y1)+b2
        y2 = (np.abs(z2)+z2)/2*alpha + (-np.abs(z2)+z2)/2*beta
        error = error+((y2-output[i])**2).sum()

    return error/len(input)