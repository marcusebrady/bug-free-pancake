import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.shape = self.data.shape  # Add this line
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += np.sum(out.grad, axis=tuple(range(out.grad.ndim - self.data.ndim))) \
                         .reshape(self.data.shape)
            other.grad += np.sum(out.grad, axis=tuple(range(out.grad.ndim - other.data.ndim))) \
                          .reshape(other.data.shape)
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += np.sum(other.data * out.grad, axis=tuple(range(out.grad.ndim - self.data.ndim))) \
                         .reshape(self.data.shape)
            other.grad += np.sum(self.data * out.grad, axis=tuple(range(out.grad.ndim - other.data.ndim))) \
                          .reshape(other.data.shape)
        out._backward = _backward
        
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), (self, other), 'matmul')
        
        def _backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)
        out._backward = _backward
        
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), 'sum')
        
        def _backward():
            if axis is None:
                self.grad += np.ones_like(self.data) * out.grad
            else:
                self.grad += np.expand_dims(out.grad, axis=axis) * np.ones_like(self.data)
        out._backward = _backward
        
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for v in reversed(topo):
            v._backward()


    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx.data
        out = Tensor(self.data[idx], (self,), 'getitem')
        def _backward():
            grad = np.zeros_like(self.data)
            np.add.at(grad, idx, out.grad)
            self.grad += grad
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,), 'transpose')
        def _backward():
            self.grad += out.grad.transpose(*reversed(axes) if axes else None)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
def scatter_mean(src, index, dim_size):
    if isinstance(index, Tensor):
        index = index.data
    out = Tensor(np.zeros((dim_size, src.data.shape[1])))
    count = np.zeros(dim_size)
    np.add.at(out.data, index, src.data)
    np.add.at(count, index, 1)
    count[count == 0] = 1  # avoid division by zero
    out.data /= count[:, None]
    return out

def scatter_sum(src, index, dim_size):
    if isinstance(index, Tensor):
        index = index.data
    out = Tensor(np.zeros((dim_size, src.data.shape[1])))
    np.add.at(out.data, index, src.data)
    return out

def tensor(data):
    return Tensor(data)

def randn(*shape):
    return Tensor(np.random.randn(*shape))

def zeros(*shape):
    return Tensor(np.zeros(shape))
