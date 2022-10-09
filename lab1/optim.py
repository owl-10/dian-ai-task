from .tensor import Tensor
from .modules import Module


class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):

        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.
        for attr in vars(module).values():
            if isinstance(attr, Module):
                self._step_module(attr) #递归调用
            if isinstance(attr, Tensor):
                if hasattr(attr, 'grad'):
                    self._update_weight(attr) #更新weight
            if isinstance(attr, list):
                for elem in attr:
                    self._step_module(elem)
        ...

        # End of todo

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum

    def _update_weight(self, tensor):

        # TODO Update the weight of tensor
        # in SGD manner.
        if 'v' in vars(tensor):
            tensor.v = self.momentum * tensor.v + self.lr * tensor.grad
        else:
            tensor.v = self.lr * tensor.grad
        tensor -= tensor.v
        ...

        # End of todo


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.
        pass

        # End of todo

    def _update_weight(self, tensor):

        # TODO Update the weight of
        # tensor in Adam manner.
        pass
        ...

        # End of todo
