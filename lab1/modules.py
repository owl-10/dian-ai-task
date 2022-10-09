import numpy as np
from itertools import product
from . import tensor
import math

class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # w[0] for bias and w[1:] for weight
        self.w = tensor.tensor((in_length + 1, out_length))

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.

        self.x=x
        return np.dot(x,self.w[1:])+self.w[0]
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        self.w.grad=np.vstack((np.sum(dy,axis=0),np.dot(self.x.T, dy)))
        return np.dot(dy,self.w[1:].T)
        ...

        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.
        self.running_mean = np.zeros((length,))
        self.running_var = np.zeros((length,))
        self.gamma = tensor.ones((length,))
        self.beta = tensor.zeros((length,))
        self.momentum = momentum
        self.eps = 1e-5
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.
        if self.training:
            self.mean = np.mean(x, axis = 0)
            self.var = np.var(x, axis = 0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
            self.x = (x - self.mean) / np.sqrt(self.var + self.eps)
        else:
            self.x = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.x + self.beta
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.
        self.gamma.grad = np.sum(dy * self.x, axis=0)
        self.beta.grad = np.sum(dy, axis=0)
        N = dy.shape[0]
        dy *= self.gamma
        dx = N * dy - np.sum(dy, axis=0) - self.x * np.sum(dy * self.x, axis=0)
        return dx / N / np.sqrt(self.var + self.eps)
        ...

        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=True):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.
        self.in_channels=in_channels
        self.channels=channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.bias=True
        self.weights_scale = math.sqrt(self.kernel_size*self.kernel_size*self.in_channels)
        self.weights = np.random.standard_normal((self.kernel_size, self.kernel_size, self.in_channels, self.channels)) // self.weights_scale
        self.bias = np.random.standard_normal(self.channels) // self.weights_scale
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        B, C, H, W = x.shape
        eta = np.zeros((B, self.channels, (H-self.kernel_size)//self.stride+1, (H-self.kernel_size)//self.stride+1))

        col_weights = self.weights.reshape([-1, self.channels])
        conv_out = np.zeros(eta.shape)

        for i in range(B):
            img_i = x[i] # (C, H, W)
            col_img_i = Conv2d_im2col.forward(img_i)
            conv_out[i] = np.reshape(np.dot(col_img_i, col_weights)+self.bias, eta[0].shape)

        return conv_out
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.
        pass
        ...

        # End of todo


class Conv2d_im2col(Conv2d):

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        image_col = []
        for i in range(0, x.shape[1]-self.kernel_size+1, self.stride):
            for j in range(0, x.shape[2]-self.kernel_size+1, self.stride):
                col = x[:, i:i+self.kernel_size, j:j+self.kernel_size].reshape([-1])
                image_col.append(col)
        return np.array(image_col)
        ...

        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_height = 2
        self.w_width =2
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.
        self.x = x
        self.in_height = x.shape[0]
        self.in_width = x.shape[1]
        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1
        out = np.zeros((self.out_height, self.out_width))
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = np.mean(x[start_i: end_i, start_j: end_j])
        return out
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.
        dx = tensor.ones_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                dx[start_i: end_i, start_j: end_j] = dy[i, j] / (self.w_width * self.w_height)
        return dx
        ...

        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.
        self.kernel_size = kernel_size
        self.w_height = 2
        self.w_width = 2
        self.stride = stride
        self.x = None
        self.in_height = None
        self.in_width = None
        self.out_height = None
        self.out_width = None
        self.arg_max = None
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.
        self.x = x
        self.in_height = np.shape(x)[0]
        self.in_width = np.shape(x)[1]
        self.out_height = int((self.in_height - self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride) + 1
        out = tensor.zeros((self.out_height, self.out_width))
        self.arg_max = tensor.ones_like(out, dtype=np.int32)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = tensor.max(x[start_i: end_i, start_j: end_j])
                self.arg_max[i, j] = tensor.argmax(x[start_i: end_i, start_j: end_j])
        self.arg_max = self.arg_max
        return out
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.
        dx = tensor.ones_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                index = tensor.unravel_index(self.arg_max[i, j], self.kernel_size)
                dx[start_i:end_i, start_j:end_j][index] = dy[i, j] #
        return dx
        ...

        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

        ...

        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.
        c = (1 - self.p)
        self._mask = np.random.uniform(size=x.shape) > self.p
        c = self._mask
        return x * c
        ...

        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.
        return dy* self._mask
        ...

        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()
