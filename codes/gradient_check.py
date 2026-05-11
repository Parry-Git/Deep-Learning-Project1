"""Numerical gradient check for all core operators."""
import numpy as np
import mynn as nn
from mynn.op import Linear, conv2D, MaxPool2D, MultiCrossEntropyLoss, softmax


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f()
        x[idx] = old - eps
        fm = f()
        grad[idx] = (fp - fm) / (2 * eps)
        x[idx] = old
        it.iternext()
    return grad


def rel_error(a, b):
    return np.max(np.abs(a - b) / (np.maximum(np.abs(a) + np.abs(b), 1e-12)))


def check_linear():
    np.random.seed(42)
    X = np.random.randn(4, 5)
    layer = Linear(5, 3, initialize_method=lambda size: np.random.randn(*size) * 0.1)
    out = layer.forward(X)
    grad_out = np.random.randn(*out.shape)

    layer.backward(grad_out)
    analytic_gW = layer.grads['W'].copy()
    analytic_gb = layer.grads['b'].copy()
    analytic_gX = layer.backward(grad_out).copy()

    num_gW = numerical_gradient(lambda: np.sum(layer.forward(X) * grad_out), layer.W)
    num_gb = numerical_gradient(lambda: np.sum(layer.forward(X) * grad_out), layer.b)
    num_gX = numerical_gradient(lambda: np.sum(Linear.forward(layer, X) * grad_out), X)

    print(f"  Linear dW rel_error: {rel_error(analytic_gW, num_gW):.2e}")
    print(f"  Linear db rel_error: {rel_error(analytic_gb, num_gb):.2e}")
    print(f"  Linear dX rel_error: {rel_error(analytic_gX, num_gX):.2e}")
    return rel_error(analytic_gW, num_gW), rel_error(analytic_gX, num_gX)


def check_cross_entropy():
    np.random.seed(42)
    logits = np.random.randn(4, 10)
    labels = np.array([2, 5, 0, 7])
    loss_fn = MultiCrossEntropyLoss(model=None, max_classes=10)

    loss_fn.forward(logits, labels)
    analytic = loss_fn.backward().copy()

    def f():
        return loss_fn.forward(logits, labels)

    num = numerical_gradient(f, logits)
    err = rel_error(analytic, num)
    print(f"  CrossEntropy dlogits rel_error: {err:.2e}")
    return err


def check_conv2d():
    np.random.seed(42)
    X = np.random.randn(2, 1, 6, 6)
    layer = conv2D(1, 2, 3, stride=1, padding=1,
                   initialize_method=lambda size: np.random.randn(*size) * 0.1)
    out = layer.forward(X)
    grad_out = np.random.randn(*out.shape)

    analytic_gX = layer.backward(grad_out).copy()
    analytic_gW = layer.grads['W'].copy()

    num_gW = numerical_gradient(lambda: np.sum(layer.forward(X) * grad_out), layer.W)
    num_gX = numerical_gradient(lambda: np.sum(layer.forward(X) * grad_out), X)

    print(f"  conv2D dW rel_error: {rel_error(analytic_gW, num_gW):.2e}")
    print(f"  conv2D dX rel_error: {rel_error(analytic_gX, num_gX):.2e}")
    return rel_error(analytic_gW, num_gW), rel_error(analytic_gX, num_gX)


def check_maxpool():
    np.random.seed(42)
    X = np.random.randn(2, 2, 4, 4)
    layer = MaxPool2D(pool_size=2)
    out = layer.forward(X)
    grad_out = np.random.randn(*out.shape)

    analytic_gX = layer.backward(grad_out).copy()
    num_gX = numerical_gradient(lambda: np.sum(layer.forward(X) * grad_out), X)

    err = rel_error(analytic_gX, num_gX)
    print(f"  MaxPool2D dX rel_error: {err:.2e}")
    return err


if __name__ == '__main__':
    print("Linear:")
    check_linear()
    print("Softmax CrossEntropy:")
    check_cross_entropy()
    print("conv2D:")
    check_conv2d()
    print("MaxPool2D:")
    check_maxpool()
    print("\nAll gradient checks passed.")
