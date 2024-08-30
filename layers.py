import numpy as np
from autograd import Tensor
from functools import wraps

def init_weights(shape, method='xavier_uniform'):
    if method == 'xavier_uniform':
        bound = np.sqrt(6.0 / sum(shape))
        return Tensor(np.random.uniform(-bound, bound, shape))
    elif method == 'xavier_normal':
        std = np.sqrt(2.0 / sum(shape))
        return Tensor(np.random.normal(0, std, shape))
    elif method == 'he_uniform':
        bound = np.sqrt(6.0 / shape[0])
        return Tensor(np.random.uniform(-bound, bound, shape))
    elif method == 'he_normal':
        std = np.sqrt(2.0 / shape[0])
        return Tensor(np.random.normal(0, std, shape))
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def with_weight_init(init_method='xavier_uniform'):
    def decorator(init_func):
        @wraps(init_func)
        def wrapper(self, *args, **kwargs):
            init_func(self, *args, **kwargs)
            for name, param in vars(self).items():
                if name.startswith('W_'):
                    shape = param.data.shape
                    setattr(self, name, init_weights(shape, method=init_method))
        return wrapper
    return decorator

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = Tensor(np.zeros_like(p.data))
    
    def parameters(self):
        return [getattr(self, name) for name in vars(self) if name.startswith('W_') or name.startswith('b_')]

class Linear(Module):
    @with_weight_init()
    def __init__(self, in_features, out_features):
        self.W_linear = Tensor(np.zeros((in_features, out_features)))
        self.b_linear = Tensor(np.zeros(out_features))
    
    def __call__(self, x):
        return x.matmul(self.W_linear) + self.b_linear

class GraphConv(Module):
    @with_weight_init()
    def __init__(self, in_features, out_features):
        self.W_conv = Tensor(np.zeros((in_features, out_features)))
        self.b_conv = Tensor(np.zeros(out_features))
    
    def __call__(self, x, adj):
        return adj.matmul(x).matmul(self.W_conv) + self.b_conv


