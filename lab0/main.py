import struct
import numpy as np
import matplotlib.pyplot as plt
from preprocess import getXmean
from preprocess import centralized
from preprocess import mean
from preprocess import PCA1
from sklearn.decomposition import PCA
from knn import Knn
import os
IMAGEROW=28
IMAGECOL=28



def load_mnist(root='D:/mnist/'):
    trainlabels_path=os.path.join(root, 'train-labels.idx1-ubyte')
    trainimages_path=os.path.join(root, 'train-images.idx3-ubyte')
    testlabels_path=os.path.join(root, 't10k-labels.idx1-ubyte')
    testimages_path=os.path.join(root, 't10k-images.idx3-ubyte')
    with open(trainlabels_path, 'rb') as lapath:
        magic, n = struct.unpack('>II', lapath.read(8))
        y_train = np.fromfile(lapath, dtype=np.uint8).reshape(60000,)
    with open(trainimages_path, 'rb') as impath:
        magic, num, rows, cols = struct.unpack('>IIII', impath.read(16))
        x_train = np.fromfile(impath,dtype=np.uint8).reshape(60000,28*28)
    with open(testlabels_path, 'rb') as lapath1:
        magic, n = struct.unpack('>II', lapath1.read(8))
        y_test = np.fromfile(lapath1,dtype=np.uint8).reshape(10000,)
    with open(testimages_path, 'rb') as impath1:
        magic, num, rows, cols = struct.unpack('>IIII', impath1.read(16))
        x_test = np.fromfile(impath1,dtype=np.uint8).reshape(10000,28*28)
    #x_test=x_test[:1000]
    #y_test=y_test[:1000]
    return x_train,y_train,x_test,y_test
    # TODO Load the MNIST dataset
    # 1. Download the MNIST dataset from
    #    http://yann.lecun.com/exdb/mnist/
    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.

    # Input:
    # root: str, the directory of mnist

    # Output:
    # X_train: np.array, shape (6e4, 28, 28)
    # y_train: np.array, shape (6e4,)
    # X_test: np.array, shape (1e4, 28, 28)
    # y_test: np.array, shape (1e4,)

    # Hint:
    # 1. Use np.fromfile to load the MNIST dataset(notice offset).
    # 2. Use np.reshape to reshape the MNIST dataset.

    # YOUR CODE HERE
    # raise NotImplementedError
    ...

    # End of todo


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    pca=PCA(0.85)
    pca.fit(X_train)
    mean_image=getXmean(X_train)
    X_train=centralized(X_train,mean_image)
    X_test=centralized(X_test, mean_image)
    X_train_reduction=pca.transform(X_train)
    X_test_reduction=pca.transform(X_test)
    knn = Knn()
    knn.fit(X_train_reduction, y_train)
    y_pred = knn.predict(X_test_reduction)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
