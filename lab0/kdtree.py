import os
import struct
import numpy as np
import itertools
from datetime import datetime
from sklearn.decomposition import PCA
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def getXmean(data):
    data=np.reshape(data,(data.shape[0],-1))
    mean_image=np.mean(data,axis=0)
    return mean_image

def centralized(data, mean_image):
    data = data.reshape((data.shape[0], -1))
    data = data.astype(np.float64)
    data -= mean_image
    return data

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
    return x_train,y_train,x_test,y_test


class node:
    def __init__(self, point, label):
        self.left = None
        self.right = None
        self.point = point
        self.label = label  #由于按树存储的时候数据点顺序打乱了，这里将label也存进树里面。
        self.parent = None
        pass

    def set_left(self, left):
        if left == None: pass
        left.parent = self
        self.left = left

    def set_right(self, right):
        if right == None: pass
        right.parent = self
        self.right = right


def median(lst):
    m = len(lst) // 2
    return lst[m], m


def build_kdtree(data, d):
    data = sorted(data, key=lambda x: x[next(d)])
    p, m = median(data)
    tree = node(p[:-1], p[-1])

    del data[m]

    #递归查询新节点该存放的位置，同时也递归的切分区域
    if m > 0: tree.set_left(build_kdtree(data[:m], d))
    if len(data) > 1: tree.set_right(build_kdtree(data[m:], d))
    return tree

#计算距离
def distance(a, b):
    diff = a - b
    squaredDiff = diff ** 2
    return np.sum(squaredDiff)


def search_kdtree(tree, d, target, k):
    den = next(d)
    #直到搜索到不存在更近的节点才停止。
    if target[den] < tree.point[den]:
        if tree.left != None:
            return search_kdtree(tree.left, d, target, k)
    else:
        if tree.right != None:
            return search_kdtree(tree.right, d, target, k)

    #持续更新距离最近的k个节点
    def update_best(t, best):
        if t == None: return
        label = t.label
        t = t.point
        d = distance(t, target)
        for i in range(k):
            if d < best[i][1]:
                for j in range(0, i):
                    best[j][1] = best[j+1][1]
                    best[j][0] = best[j+1][0]
                    best[j][2] = best[j+1][2]
                best[i][1] = d
                best[i][0] = t
                best[i][2] = label
    best = []
    for i in range(k):
        best.append( [tree.point, 100000.0, 10] )
    while (tree.parent != None):
        update_best(tree.parent.left, best)
        update_best(tree.parent.right, best)
        tree = tree.parent
    return best
def testHandWritingClass():
    train_x, train_y, test_x, test_y = load_mnist()
    l = min(train_x.shape[0], train_y.shape[0])
    rows = train_x.shape[1]
    #将训练集的标签存到train_x中，一遍一同存储到kd树中。
    for i in range(l):
        train_x[i, -1] = train_y[i]

    densim = itertools.cycle(range(0, rows-1))
    ## step 2: training...
    mnist_tree = build_kdtree(train_x, densim)

    ## step 3: testing
    a = datetime.now()
    numTestSamples = test_x.shape[0]
    matchCount = 0
    test_num = numTestSamples
    K = 6
    for i in range(test_num):
        best_k = search_kdtree(mnist_tree, densim, test_x[i, :-1], K)
        #计算数量最大的label。
        classCount = {}
        for j in range(K):
            voteLabel = best_k[j][2]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        maxCount = 0
        predict = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                predict = key
        if predict == test_y[i]:
            matchCount += 1
        if i % 100 == 0:
            print ("完成%d张图片"%(i))
    accuracy = float(matchCount) / test_num
    b = datetime.now()
    print ("一共运行了%d秒"%((b-a).seconds))

    ## step 4: show the result
    print ('The classify accuracy is: %.2f%%' % (accuracy * 100))

if __name__ == '__main__':
    testHandWritingClass()

