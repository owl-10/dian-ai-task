import numpy as np
def getXmean(data):
    data=np.reshape(data,(data.shape[0],-1))
    mean_image=np.mean(data,axis=0)
    return mean_image

def centralized(data, mean_image):
    data = data.reshape((data.shape[0], -1))
    data = data.astype(np.float64)
    data -= mean_image
    return data
def  mean(data):
    data-= np.mean(data)
    return data
def PCA1(X):
    X -= np.mean(X, axis = 0) # 对数据进行零中心化(重要)
    cov = np.dot(X.T, X) / X.shape[0]
    U,S,V = np.linalg.svd(cov)
    Xrot = np.dot(X,U)
    Xrot_reduced = np.dot(X, U[:,:100])
    return Xrot_reduced