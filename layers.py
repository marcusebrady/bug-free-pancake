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
    def __init__(self, aggr='sum'):
        super().__init__()
        self.aggr = aggr
    def propagate(self, edge_index, x, edge_attr=None):
        src, dst = edge_index.data[0], edge_index.data[1]
        x_j = x[Tensor(src)]
        messages = self.message(x_j, x, edge_index, edge_attr)
        aggregated = self.aggregate(messages, Tensor(dst), dim_size=x.data.shape[0])  # Use x.data.shape
        return self.update(aggregated, x)

    def message(self, x_j: Tensor, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        return x_j

    def aggregate(self, messages, index, dim_size):
        if self.aggr == "mean":
            return scatter_mean(messages, index, dim_size=dim_size)
        elif self.aggr == "sum":
            return scatter_sum(messages, index, dim_size=dim_size)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggr}")

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return aggr_out

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x, edge_attr)


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
        super().__init__(aggr='sum')
        self.in_features = in_features
        self.out_features = out_features
        self.W_conv = Tensor(np.zeros((in_features, out_features)))
        self.b_conv = Tensor(np.zeros(out_features))

    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward(x, edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.propagate(edge_index, x)
    
    def message(self, x_j: Tensor, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        return x_j.matmul(self.W_conv)
    
    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
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

    def __call__(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.forward(x, edge_index)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.project:
            x = self.lin_proj(x).relu()

        out = self.propagate(edge_index, x=x)
        out = self.lin_l(out) + self.lin_r(x)

        if self.normalize:
            out = out / (out.pow(2).sum(axis=1, keepdims=True).sqrt() + 1e-6)

        return out

    def message(self, x_j: Tensor, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        return x_j

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return aggr_out

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
#https://arxiv.org/pdf/1706.08566
#below uses this paper

class ContinuousFilterConv(Module):
    @with_weight_init()
    def __init__(self, num_features: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.num_features = num_features
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        self.filter_network = Sequential(
                Linear(num_rbf, num_features),
                shifted_softplus,
                Linear(num_features, num_features)
            )


    def __call__(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_index.data[:, 0], edge_index.data[:, 1]
        rbf_features = rbf_expansion(edge_attr, self.num_rbf, self.cutoff)
        filters = self.filter_network(rbf_features)
        filters = filters * cosine_cutoff(edge_attr, self.cutoff)
        x_j = x[Tensor(src)]
        messages = x_j * filters
        return scatter_sum(messages, Tensor(dst), dim_size=x.shape[0])

#Message and Update use: https://arxiv.org/pdf/2102.03150
class MessageBlock(Module):
    @with_weight_init()
    def __init__(self, num_features: int, num_rbf: int):
        super().__init__()
        self.conv_filter = ContinuousFilterConv(num_features, num_rbf, cutoff=5.0)
        self.phi = Sequential(
            Linear(num_features, num_features),
            shifted_softplus,
            Linear(num_features, num_features * 3)
        )

    def __call__(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index.data[:, 0], edge_index.data[:, 1]
        r_ij = edge_attr
        
        s_j = s[Tensor(src)]
        v_j = v[Tensor(src)]
        
        phi_output = self.phi(s_j)
        phi_s, phi_vv, phi_vs = phi_output.split(3, axis=-1)
        
        filters = self.conv_filter(s_j, edge_index, edge_attr)
        Ws, Wvv, Wvs = filters.split(3, axis=-1)
        
        delta_s = scatter_sum(phi_s * Ws, Tensor(dst), dim_size=s.shape[0])
        delta_v = scatter_sum(v_j * phi_vv.unsqueeze(-1) * Wvv.unsqueeze(-1), Tensor(dst), dim_size=v.shape[0])
        delta_v += scatter_sum(phi_vs.unsqueeze(-1) * Wvs.unsqueeze(-1) * (r_ij / (r_ij.norm(axis=-1, keepdims=True) + 1e-10)), Tensor(dst), dim_size=v.shape[0])
        
        return delta_s, delta_v

class UpdateBlock(Module):
    @with_weight_init()
    def __init__(self, num_features: int):
        super().__init__()
        self.update_s = Linear(num_features * 2, num_features)
        self.update_v = Linear(num_features, num_features)
        self.U = Linear(num_features, num_features, bias=False)
        self.V = Linear(num_features, num_features, bias=False)
        self.a = Sequential(
            Linear(num_features * 2, num_features),
            shifted_softplus,
            Linear(num_features, num_features * 3)
        )

    def __call__(self, s: Tensor, v: Tensor, delta_s: Tensor, delta_v: Tensor) -> Tuple[Tensor, Tensor]:
        v_norm = v.norm(axis=-1)
        a_input = s.concatenate(v_norm, axis=-1)
        a_output = self.a(a_input)
        ass, asv, avv = a_output.split(3, axis=-1)
        
        s_out = ass * s + asv * (self.U(v) * self.V(v)).sum(axis=-1)
        v_out = avv.unsqueeze(-1) * self.U(v)
        
        return s_out + delta_s, v_out + delta_v

def rbf_expansion(distances: Tensor, num_rbf: int, cutoff: float) -> Tensor:
    centers = Tensor(np.linspace(0, cutoff, num_rbf))
    return (-0.5 * ((distances.unsqueeze(-1) - centers) / (cutoff / num_rbf))**2).exp()

def cosine_cutoff(r: Tensor, cutoff: float) -> Tensor:
    return 0.5 * (Tensor(np.cos(np.pi * r.data / cutoff)) + 1.0) * (r.data < cutoff)


def shifted_softplus(x: Tensor) -> Tensor:
    return Tensor(np.log(0.5 * np.exp(x.data) + 0.5))

def compute_edges(positions: Tensor, cutoff: float) -> Tuple[Tensor, Tensor]:
    num_atoms = positions.shape[0]
    distances = compute_distances(positions)
    edge_index = np.array(np.where(distances.data < cutoff)).T
    edge_attr = distances[Tensor(edge_index[:, 0]), Tensor(edge_index[:, 1])]
    return Tensor(edge_index), edge_attr

def compute_distances(positions: Tensor) -> Tensor:
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    return (diff ** 2).sum(axis=-1).sqrt()

