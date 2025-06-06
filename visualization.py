import numpy as np
import matplotlib.pyplot as plt
import NN_CRN
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import utils


def XOR_smoothed_ReLU(network, train_loss, accuracy, params, N, epoch, noise_controller) : 
    #Train loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training loss(H='+str(network.H)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training loss(H='+str(network.H)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training loss(H='+str(network.H)+') with input noise '+str(network.noise))
    plt.ylim(-0.01,1)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training loss(H='+str(network.H)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training loss(H='+str(network.H)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training loss(H='+str(network.H)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Accuracy graph
    time2 = [time[i*100] for i in range(int(len(time)/100))] 
    plt.plot(time2, accuracy[:int(N*epoch/100)], 'r', label = 'accuracy')
    if(noise_controller == 0) : plt.title('Accuracy(H='+str(network.H)+') with no noise')
    elif(noise_controller == 1) : plt.title('Accuracy(H='+str(network.H)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Accuracy(H='+str(network.H)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Accuracy(H='+str(network.H)+') with input noise '+str(network.noise))
    plt.ylim(-0.05,1.05)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Accuracy(H='+str(network.H)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Accuracy(H='+str(network.H)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Accuracy(H='+str(network.H)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Accuracy(H='+str(network.H)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Visualization of learning process for XOR dataset
    fig = plt.figure(figsize=(10,50))
    xaxis = np.arange(-1,2,0.1)
    yaxis = np.arange(-1,2,0.1)
    xaxis, yaxis = np.meshgrid(xaxis, yaxis)
    for i in range(5) : 
        z1 = [0]*network.n2
        z = 0
        ax = fig.add_subplot(5,1,i+1)
        for j in range(network.n2) : 
            z1[j] = (params[0][(i)*(int(epoch*N/5))])[j][0]*xaxis+(params[0][(i)*(int(epoch*N/5))])[j][1]*yaxis+(params[2][(i)*(int(epoch*N/5))])[j]
            z1[j] = (z1[j]+np.sqrt(np.power(z1[j],2)+4*network.H))/2
                
        for j in range(network.n2) : 
            z = z+(params[1][(i)*(int(epoch*N/5))])[0][j]*z1[j]
        z = z+params[3][(i)*(int(epoch*N/5))]
        z = (z+np.sqrt(np.power(z,2)+4*network.H))/2
        labels = np.round(np.arange(-1,2,0.1),1)
        df = pd.DataFrame(z, index=labels, columns=labels)
        df = df.iloc[::-1,:]
        ax = sns.heatmap(df)
    plt.show()
    return

def XOR_Leaky_ReLU(network, train_loss, accuracy, params, N, epoch, noise_controller) : 
    #Train loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise '+str(network.noise))
    plt.ylim(-0.01,1)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Accuracy graph
    time2 = [time[i*100] for i in range(int(len(time)/100))] 
    plt.plot(time2, accuracy[:int(N*epoch/100)], 'r', label = 'accuracy')
    if(noise_controller == 0) : plt.title('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with no noise')
    elif(noise_controller == 1) : plt.title('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with input noise '+str(network.noise))
    plt.ylim(-0.05,1.05)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Accuracy(H='+str(network.alpha)+',beta='+str(network.beta)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Visualization of learning process for XOR dataset
    fig = plt.figure(figsize=(10,50))
    xaxis = np.arange(-1,2,0.1)
    yaxis = np.arange(-1,2,0.1)
    xaxis, yaxis = np.meshgrid(xaxis, yaxis)
    for i in range(5) : 
        ax = fig.add_subplot(5,1,i+1)
        z1 = [0]*network.n2
        z = 0
        for j in range(network.n2) : 
            z1[j] = (params[0][(i)*(int(epoch*N/5))])[j][0]*xaxis+(params[0][(i)*(int(epoch*N/5))])[j][1]*yaxis+(params[2][(i)*(int(epoch*N/5))])[j]
            z1[j] = (z1[j])*(network.alpha)*(z1[j]>0) + (z1[j])*(network.beta)*(z1[j]<0)
               
        for j in range(network.n2) : 
            z = z+(params[1][(i)*(int(epoch*N/5))])[0][j]*z1[j]
        z = z+params[3][(i)*(int(epoch*N/5))]
        z = z*(network.alpha)*(z>0) + z*(network.beta)*(z<0)

        labels = np.round(np.arange(-1,2,0.1),1)
        df = pd.DataFrame(z, index=labels, columns=labels)
        df = df.iloc[::-1,:]
        ax = sns.heatmap(df)
    plt.show()
    return

def sine_smoothed_ReLU(network, train_loss, noise_controller) : 
    #Train loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training loss(H='+str(network.H)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training loss(H='+str(network.H)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training loss(H='+str(network.H)+') with input noise '+str(network.noise))
    plt.ylim(-0.01,1)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training loss(H='+str(network.H)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training loss(H='+str(network.H)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training loss(H='+str(network.H)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()
    return

def sine_Leaky_ReLU(network, train_loss, noise_controller) : 
    #Train loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise '+str(network.noise))
    plt.ylim(-0.01,1)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training loss(H='+str(network.alpha)+',beta='+str(network.beta)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()
    return

def iris_smoothed_ReLU(network, train_loss, validation_loss, accuracy, N, epoch, noise_controller) : 
    #Train loss + validation loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, validation_loss,'b', label = 'validation')
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training and validation loss(H='+str(network.H)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training and validation loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training and validation loss(H='+str(network.H)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training and validation loss(H='+str(network.H)+') with input noise '+str(network.noise))

    plt.ylim(-0.01,5)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training and validation loss(H='+str(network.H)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training and validation loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training and validation loss(H='+str(network.H)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training and validation loss(H='+str(network.H)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Accuracy graph
    time2 = [time[i*100] for i in range(int(len(time)/100))] 
    plt.plot(time2, accuracy[:int(N*epoch/100)], 'r', label = 'accuracy')
    if(noise_controller == 0) : plt.title('Accuracy(H='+str(network.H)+') with no noise')
    elif(noise_controller == 1) : plt.title('Accuracy(H='+str(network.H)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Accuracy(H='+str(network.H)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Accuracy(H='+str(network.H)+') with input noise '+str(network.noise))
    plt.ylim(-0.05,1.05)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Accuracy(H='+str(network.H)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Accuracy(H='+str(network.H)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Accuracy(H='+str(network.H)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Accuracy(H='+str(network.H)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Visualization of dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df['species'] = y
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    sns.set(font_scale=1.5) 
    g=sns.pairplot(iris_df, hue='species', palette='husl')
    g.fig.suptitle('Iris Dataset',y=1.02)
    plt.savefig('IRIS dataset.svg', dpi=500, bbox_inches='tight')
    plt.show()

    #Visualization of trained model
    iris = load_iris()
    X = iris.data
    y = []
    for i in range(len(X)) : 
        z1 = np.dot((network.W1p-network.W1n),np.array(X[i]))+(network.b1p-network.b1n)
        y1 = (z1+np.sqrt(np.power(z1,2)+4*network.H))/2
        z2 = np.dot((network.W2p-network.W2n),y1)+(network.b2p-network.b2n)
        y2 = (z2+np.sqrt(np.power(z2,2)+4*network.H))/2
        y.append(np.argmax(y2))
    feature_names = iris.feature_names
    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df['species'] = y
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    sns.set(font_scale=1.5) 
    g=sns.pairplot(iris_df, hue='species', palette='husl')
    g.fig.suptitle('Iris Dataset classified by NN',y=1.02)
    plt.savefig('IRIS classification.svg', dpi=500, bbox_inches='tight')
    plt.show()
    return

def iris_Leaky_ReLU(network, train_loss, validation_loss, accuracy, N, epoch, noise_controller) : 
    #Train loss + validation loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, validation_loss,'b', label = 'validation')
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise '+str(network.noise))

    plt.ylim(-0.01,5)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Accuracy graph
    time2 = [time[i*100] for i in range(int(len(time)/100))] 
    plt.plot(time2, accuracy[:int(N*epoch/100)], 'r', label = 'accuracy')
    if(noise_controller == 0) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise')
    elif(noise_controller == 1) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise '+str(network.noise))
    plt.ylim(-0.05,1.05)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Visualization of dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df['species'] = y
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    sns.set(font_scale=1.5) 
    g=sns.pairplot(iris_df, hue='species', palette='husl')
    g.fig.suptitle('Iris Dataset',y=1.02)
    plt.savefig('IRIS dataset.svg', dpi=500, bbox_inches='tight')
    plt.show()

    #Visualization of trained model
    iris = load_iris()
    X = iris.data
    y = []
    for i in range(len(X)) : 
        z1 = np.dot((network.W1p-network.W1n),np.array(X[i]))+(network.b1p-network.b1n)
        y1 = network.alpha*(z1>0)*z1+network.beta*(z1<0)*z1
        z2 = np.dot((network.W2p-network.W2n),y1)+(network.b2p-network.b2n)
        y2 = network.alpha*(z2>0)*z2+network.beta*(z2<0)*z2
        y.append(np.argmax(y2))
    feature_names = iris.feature_names
    iris_df = pd.DataFrame(X, columns=feature_names)
    iris_df['species'] = y
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    sns.set(font_scale=1.5) 
    g=sns.pairplot(iris_df, hue='species', palette='husl')
    g.fig.suptitle('Iris Dataset classified by NN',y=1.02)
    plt.savefig('IRIS classification.svg', dpi=500, bbox_inches='tight')
    plt.show()
    return

def MNIST_smoothed_ReLU(network, train_loss, validation_loss, accuracy,N,epoch,noise_controller) : 

    #Train loss + validation loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, validation_loss,'b', label = 'validation')
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training and validation loss(H='+str(network.H)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training and validation loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training and validation loss(H='+str(network.H)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training and validation loss(H='+str(network.H)+') with input noise '+str(network.noise))
    plt.ylim(-0.01,5)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training and validation loss(H='+str(network.H)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training and validation loss(H='+str(network.H)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training and validation loss(H='+str(network.H)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training and validation loss(H='+str(network.H)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Accuracy graph
    time2 = [time[i*100] for i in range(int(len(time)/100))] 
    plt.plot(time2, accuracy[:int(N*epoch/100)], 'r', label = 'accuracy')
    if(noise_controller == 0) : plt.title('Accuracy(H='+str(network.H)+') with no noise')
    elif(noise_controller == 1) : plt.title('Accuracy(H='+str(network.H)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Accuracy(H='+str(network.H)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Accuracy(H='+str(network.H)+') with input noise '+str(network.noise))
    plt.ylim(-0.05,1.05)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Accuracy(H='+str(network.H)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Accuracy(H='+str(network.H)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Accuracy(H='+str(network.H)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Accuracy(H='+str(network.H)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()


    #Visualization of dataset
    # load MNIST dataset
    mnist = fetch_openml('mnist_784')
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    selected_indices = np.where((y == 0) | (y == 1))
    x = np.array(x)[selected_indices]
    y = np.array(y)[selected_indices]
    test_x = utils.downscale_images(x[N:])
    test_x = test_x.reshape(test_x.shape[0],-1)
    y = y[N:]


    # PCA model initialization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(test_x)
    colors = ['red', 'blue']

    # visualization
    plt.figure(figsize=(8, 7))
    for i in range(2):
        indices = np.where(y == i)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=colors[i], label=str(i), alpha=0.7)

    plt.title('MNIST Dataset Visualization with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Digit')
    plt.savefig('MNIST dataset.svg', dpi=500)
    plt.show()

    #Visualization of trained model
    X = test_x
    y = []

    for i in range(1000) : 
        z1 = np.dot((network.W1p-network.W1n),test_x[i])+(network.b1p-network.b1n)
        y1 = (z1+np.sqrt(np.power(z1,2)+4*network.H))/2
        z2 = np.dot((network.W2p-network.W2n),y1)+(network.b2p-network.b2n)
        y2 = (z2+np.sqrt(np.power(z2,2)+4*network.H))/2
        y.append(np.argmax(y2))
    y = np.array(y)

    # PCA model initialization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    colors = ['orange', 'green']

    # visualization
    plt.figure(figsize=(8, 7))
    for i in range(2):
        indices = np.where(y == i)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=colors[i], label=str(i), alpha=0.7)

    plt.title('MNIST Dataset Visualization with PCA(classified with NN)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Digit')
    plt.savefig('MNIST classification.svg', dpi=500, bbox_inches='tight')
    plt.show()
    return 

def MNIST_Leaky_ReLU(network, train_loss, validation_loss, accuracy,N,epoch,noise_controller) : 

    #Train loss + validation loss graph
    time = [i*network.timelen for i in range(len(train_loss))]
    plt.plot(time, validation_loss,'b', label = 'validation')
    plt.plot(time, train_loss, 'r', label = 'train')
    if(noise_controller == 0) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise')
    elif(noise_controller == 1) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise '+str(network.noise))
    plt.ylim(-0.01,5)
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Training and validation loss(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()

    #Accuracy graph
    time2 = [time[i*100] for i in range(int(len(time)/100))] 
    plt.plot(time2, accuracy[:int(N*epoch/100)], 'r', label = 'accuracy')
    if(noise_controller == 0) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise')
    elif(noise_controller == 1) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise))
    elif(noise_controller == 2) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise))
    elif(noise_controller == 3) : plt.title('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise '+str(network.noise))
    plt.ylim(-0.05,1.05)
    plt.xlabel('time')
    plt.ylabel('accuracy')
    plt.legend()
    if(noise_controller == 0) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with no noise.svg', dpi=500)
    elif(noise_controller == 1) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with reaction rate noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 2) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with layer noise '+str(network.noise)+'.svg', dpi=500)
    elif(noise_controller == 3) :plt.savefig('Accuracy(alpha='+str(network.alpha)+',beta='+str(network.beta)+') with input noise'+str(network.noise)+'.svg', dpi=500)
    plt.show()


    #Visualization of dataset
    # load MNIST dataset
    mnist = fetch_openml('mnist_784')
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int')
    selected_indices = np.where((y == 0) | (y == 1))
    x = np.array(x)[selected_indices]
    y = np.array(y)[selected_indices]
    test_x = utils.downscale_images(x[N:])
    test_x = test_x.reshape(test_x.shape[0],-1)
    y = y[N:]


    # PCA model initialization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(test_x)
    colors = ['red', 'blue']

    # visualization
    plt.figure(figsize=(8, 7))
    for i in range(2):
        indices = np.where(y == i)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=colors[i], label=str(i), alpha=0.7)

    plt.title('MNIST Dataset Visualization with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Digit')
    plt.savefig('MNIST dataset.svg', dpi=500)
    plt.show()

    #Visualization of trained model
    X = test_x
    y = []

    for i in range(1000) : 
        z1 = np.dot((network.W1p-network.W1n),test_x[i])+(network.b1p-network.b1n)
        y1 = (z1>0)*network.alpha*z1+(z1<0)*network.beta*z1
        z2 = np.dot((network.W2p-network.W2n),y1)+(network.b2p-network.b2n)
        y2 = (z2>0)*network.alpha*z2+(z2<0)*network.beta*z2
        y.append(np.argmax(y2))
    y = np.array(y)

    # PCA model initialization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    colors = ['orange', 'green']

    # visualization
    plt.figure(figsize=(8, 7))
    for i in range(2):
        indices = np.where(y == i)
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=colors[i], label=str(i), alpha=0.7)

    plt.title('MNIST Dataset Visualization with PCA(classified with NN)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Digit')
    plt.savefig('MNIST classification.svg', dpi=500, bbox_inches='tight')
    plt.show()
    return 

