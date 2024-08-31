import numpy as np
from autograd import Tensor, scatter_sum, scatter_mean
from typing import Optional, Tuple
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

class MessagePassing(Module):
    def __init__(self, aggr='mean'):
        super().__init__()
        self.aggr = aggr

    def propagate(self, x, edge_index, edge_attr=None):
        # Create messages
        messages = self.message(x, edge_index, edge_attr)
        aggregated = self.aggregate(messages, edge_index)
        x = self.update(x, aggregated)
        
    
        if edge_attr is not None:
            edge_attr = self.update_edges(x, edge_index, edge_attr)
        
        return x, edge_attr

    def message(self, x, edge_index, edge_attr=None):
        return x[edge_index[1]]

    def aggregate(self, messages, edge_index):
        if self.aggr == "mean":
            return scatter_mean(messages, edge_index[0], dim_size=0)
        elif self.aggr == "sum":
            return scatter_sum(messages, edge_index[0], dim_size=0)

    def update(self, x, aggregated):
        return aggregated

    def update_edges(self, x, edge_index, edge_attr):
      
        return edge_attr

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(x, edge_index, edge_attr)


class Linear(Module):
    @with_weight_init()
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        self.W_linear = Tensor(np.zeros((in_features, out_features)))
        if self.bias:
            self.b_linear = Tensor(np.zeros(out_features))

    def __call__(self, x):
        output = x.matmul(self.W_linear)
        if self.bias:
            output = output + self.b_linear
        return output


class GraphConv(MessagePassing):
    @with_weight_init()
    def __init__(self, in_features: int, out_features: int):
        super().__init__(aggr='add')
        self.in_features = in_features
        self.out_features = out_features
        self.W_conv = Tensor(np.zeros((in_features, out_features)))
        self.b_conv = Tensor(np.zeros(out_features))
    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward(x, edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j.matmul(self.W_conv)
    
    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out + self.b_conv

class GraphSAGE(MessagePassing):
    @with_weight_init()
    def __init__(self, in_features, out_features, aggr='mean', project=False, normalize=False, bias=True):
        super().__init__(aggr)
        self.in_features = in_features
        self.out_features = out_features
        self.project = project
        self.normalize = normalize

        if self.project:
            self.lin_proj = Linear(in_features, in_features)

        self.lin_l = Linear(in_features, out_features, bias=bias)
        self.lin_r = Linear(in_features, out_features, bias=False)

    def __call__(self):
        pass



        
