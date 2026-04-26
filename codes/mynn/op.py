from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, 'Linear.backward called before Linear.forward.'
        assert grad.shape[0] == self.input.shape[0]

        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((1, out_channels, 1, 1))
        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}

        self.input_shape = None
        self.input_padded = None
        self.windows = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        assert X.ndim == 4
        assert X.shape[1] == self.in_channels

        self.input_shape = X.shape
        if self.padding > 0:
            X_padded = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            X_padded = X
        self.input_padded = X_padded

        k = self.kernel_size
        windows = np.lib.stride_tricks.sliding_window_view(X_padded, (k, k), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        self.windows = windows

        out = np.einsum('nchwkl,ockl->nohw', windows, self.W)
        return out + self.b

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        assert self.windows is not None and self.input_shape is not None
        assert grads.shape[1] == self.out_channels

        self.grads['W'] = np.einsum('nohw,nchwkl->ockl', grads, self.windows)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True)

        grad_input_padded = np.zeros_like(self.input_padded)
        out_H, out_W = grads.shape[2], grads.shape[3]
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                grad_window = np.einsum('nohw,oc->nchw', grads, self.W[:, :, i, j])
                grad_input_padded[
                    :,
                    :,
                    i:i + self.stride * out_H:self.stride,
                    j:j + self.stride * out_W:self.stride,
                ] += grad_window

        if self.padding > 0:
            return grad_input_padded[
                :,
                :,
                self.padding:-self.padding,
                self.padding:-self.padding,
            ]
        return grad_input_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    """
    Flatten image-like tensors into [batch_size, features].
    """
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        assert self.input_shape is not None
        return grads.reshape(self.input_shape)

class MaxPool2D(Layer):
    """
    Non-overlapping 2D max pooling for image-like tensors.
    """
    def __init__(self, pool_size=2) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.input_shape = None
        self.max_mask = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert X.ndim == 4
        N, C, H, W = X.shape
        assert H % self.pool_size == 0 and W % self.pool_size == 0

        self.input_shape = X.shape
        out_H = H // self.pool_size
        out_W = W // self.pool_size
        windows = X.reshape(N, C, out_H, self.pool_size, out_W, self.pool_size)
        pooled = np.max(windows, axis=(3, 5))
        self.max_mask = windows == pooled[:, :, :, None, :, None]
        return pooled

    def backward(self, grads):
        assert self.input_shape is not None and self.max_mask is not None
        N, C, H, W = self.input_shape
        out_H = H // self.pool_size
        out_W = W // self.pool_size
        grad_windows = np.zeros((N, C, out_H, self.pool_size, out_W, self.pool_size))
        max_count = np.sum(self.max_mask, axis=(3, 5), keepdims=True)
        grad_windows += self.max_mask * grads[:, :, :, None, :, None] / max_count
        return grad_windows.reshape(self.input_shape)

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.optimizable = False
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        labels = labels.astype(np.int64)
        assert predicts.ndim == 2
        assert predicts.shape[0] == labels.shape[0]
        assert predicts.shape[1] <= self.max_classes

        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]

        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts

        self.probs = np.clip(probs, 1e-12, 1.0)
        losses = -np.log(self.probs[np.arange(batch_size), labels])
        return np.mean(losses)
    
    def backward(self):
        # first compute the grads from the loss to the input
        assert self.probs is not None and self.labels is not None
        batch_size = self.labels.shape[0]

        if self.has_softmax:
            self.grads = self.probs.copy()
            self.grads[np.arange(batch_size), self.labels] -= 1
            self.grads /= batch_size
        else:
            self.grads = np.zeros_like(self.predicts)
            self.grads[np.arange(batch_size), self.labels] = -1 / self.probs[np.arange(batch_size), self.labels]
            self.grads /= batch_size

        # Then send the grads to model for back propagation
        if self.model is not None:
            self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
