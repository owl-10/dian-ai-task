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
        self.x=x
        self.tanhout=np.tanh(x)
        return self.tanhout
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.
        dx=dy*(1-np.power(self.tanhout,2))
        return dx
        ...

        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        self.x=x
        self.relu_out=np.maximum(0,x)
        return self.relu_out
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        dx=np.where(self.x > 0, dy, 0)
        return dx
        ...

        # End of todo


class Softmax(Module):


    def forward(self, x):
        # TODO Implement forward propogation
        # of Softmax function.
        e_x = np.exp(x)
        self.softmaxout= e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.softmaxout
        ...

        # End of todo

    def backward(self, dy):

        return dy
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
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.
        e_x=np.exp(probs)
        self.probs=e_x/np.sum(e_x,axis=1,keepdims=True)
        self.targets=targets.astype(np.int)
        self.value=np.sum(-np.eye(self.n_classes)[self.targets]*np.log(self.probs))/self.targets.shape[0]
        return self
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.
        return self.probs-np.eye(self.n_classes)[self.targets]
        ...

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.
        super(CrossEntropyLoss, self).__call__(probs, targets)

        self.value = np.sum(-np.eye(self.n_classes)[targets] * np.log(probs)) / targets.shape[0]

        return self
        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.
        return self.probs - np.eye(self.n_classes)[self.targets]
        ...

        # End of todo
