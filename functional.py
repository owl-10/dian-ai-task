import numpy as np
from .modules import Module

def softmax(self,x):
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        out=1/(1+np.exp(-x))
        self.sigmoidout=out
        return out
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.
        dx =dy*(1-self.sigmoidout)*self.sigmoidout
        return dx
        ...

        # End of todo


class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.
        out=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        self.tanhout=out
        return out
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.
        dx=dy*(1-self.tanhout*self.tanhout)
        return dx
        ...

        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        self.x=np.maximum(0,x)
        out=self.x
        return out
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        dx=dy
        dx[self.x<=0]=0
        return dx
        ...

        # End of todo


class Softmax(Module):


    def forward(self, x):
        # TODO Implement forward propogation
        # of Softmax function.
        self.softmaxout=softmax(x)
        return self.softmaxout
        ...

        # End of todo

    def backward(self, dy):
        dx = self.softmaxout * dy
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.softmaxout * sumdx
        return dx
        # Omitted.
        ...


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.
        self.probs=probs
        self.targets=softmax(targets)
        self.loss=CrossEntropyLoss.__call__(self.targets,self.probs)
        return self.loss
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.
        batch_size = self.probs.shape[0]
        dx = (self.targets - self.probs) / batch_size
        return dx
        ...

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.

        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.

        ...

        # End of todo
