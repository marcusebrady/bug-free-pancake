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

    def __init__(self):
        self.training = True

    def train(self, mode=True):
        self.training = mode
        for name, value in vars(self).items():
            if isinstance(value, Module):
                value.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = Tensor(np.zeros_like(p.data))
    
    def parameters(self):
        params = []
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def named_parameters(self):
        named_params = []
        for name, value in vars(self).items():
            if isinstance(value, Tensor):
                named_params.append((name, value))
            elif isinstance(value, Module):
                for sub_name, sub_param in value.named_parameters():
                    named_params.append((f"{name}.{sub_name}", sub_param))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        for sub_name, sub_param in item.named_parameters():
                            named_params.append((f"{name}[{i}].{sub_name}", sub_param))
        return named_params


class MessagePassing(Module):
    def __init__(self, aggr='sum'):
        super().__init__()
        self.aggr = aggr
    def propagate(self, edge_index, x, edge_attr=None):
        src, dst = edge_index.data[0], edge_index.data[1]
        x_j = x[Tensor(src)]
        messages = self.message(x_j, x, edge_index, edge_attr)
        aggregated = self.aggregate(messages, Tensor(dst), dim_size=x.data.shape[0])  
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
    def parameters(self):
        return [self.W_linear] + ([self.b_linear] if self.bias else [])

    def named_parameters(self):
        params = [('W_linear', self.W_linear)]
        if self.bias:
            params.append(('b_linear', self.b_linear))
        return params

'''
class Linear3D(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.W_linear = Tensor(np.zeros((in_features, out_features)))
        if self.bias:
            self.b_linear = Tensor(np.zeros(out_features))

    def __call__(self, x):
        print(f"Input x shape: {x.shape}")
        
        batch_size, num_nodes, _ = x.shape
        x_reshaped = x.view(batch_size * num_nodes, self.in_features)
        output = x_reshaped.matmul(self.W_linear)
        output = output.view(batch_size, num_nodes, self.out_features)
        
        if self.bias:
            output = output + self.b_linear
        
        print(f"Output shape: {output.shape}")
        return output
'''

class Linear3D(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.W_linear = Tensor(np.zeros((in_features, out_features)))
        if self.bias:
            self.b_linear = Tensor(np.zeros(out_features))

    def __call__(self, x):
        print(f"Input x shape: {x.shape}")
        
        batch_size, num_nodes, _ = x.shape
        x_reshaped = x.view(batch_size * num_nodes, self.in_features)
        output = x_reshaped.matmul(self.W_linear)
        output = output.view(batch_size, num_nodes, self.out_features)
        
        if self.bias:
            output = output + self.b_linear
        
        print(f"Output shape: {output.shape}")
        return output

    def parameters(self):
        return [self.W_linear] + ([self.b_linear] if self.bias else [])

    def named_parameters(self):
        params = [('W_linear', self.W_linear)]
        if self.bias:
            params.append(('b_linear', self.b_linear))
        return params




class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if isinstance(layer, Module):
                params.extend(layer.parameters())
        return params

    def named_parameters(self):
        named_params = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Module):
                for name, param in layer.named_parameters():
                    named_params.append((f"layer[{i}].{name}", param))
        return named_params

#https://arxiv.org/pdf/1706.08566
#below uses this paper
# returns filters now to pass to Message Block
#returns three filters (ws, Wvv, Wvs)
class ContinuousFilterConv(Module):
    def __init__(self, num_features: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.num_features = num_features
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        self.filter_network = Sequential(
            Linear(num_rbf, num_features),
            shifted_softplus,
            Linear(num_features, num_features * 3)
        )

    def __call__(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        rbf_features = rbf_expansion(edge_attr, self.num_rbf, self.cutoff)
        
        filters = self.filter_network(rbf_features)
        
        edge_distances = edge_attr.norm(axis=-1)
        cutoff = cosine_cutoff(edge_distances, self.cutoff)
        
        num_edges = edge_attr.shape[0]
        filters = filters.view(num_edges, 3, self.num_features)
        
    
        cutoff = cutoff.view(num_edges, 1, 1)
        cutoff = cutoff.expand(num_edges, 3, self.num_features)
        
        filters = filters * cutoff
        
        return filters
        
    def parameters(self):
        return self.filter_network.parameters()

    def named_parameters(self):
        return [('filter_network.' + name, param) for name, param in self.filter_network.named_parameters()]

#Message and Update use: https://arxiv.org/pdf/2102.03150
#Seaparte processing for s/v value featuresuses eq (8) in paper

class MessageBlock(Module):
    def __init__(self, num_features: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.num_features = num_features
        self.conv_filter = ContinuousFilterConv(num_features, num_rbf, cutoff)
        self.phi = Sequential(
            Linear(num_features, num_features),
            shifted_softplus,
            Linear(num_features, num_features * 3)
        )

    def __call__(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        src, dst = edge_index.data[0], edge_index.data[1]
        
        s_j = s[Tensor(src)]
        v_j = v[Tensor(src)]          
        phi_output = self.phi(s_j)
        phi_s, phi_vv, phi_vs = phi_output.split(3, axis=-1)
        
        filters = self.conv_filter(s_j, edge_index, edge_attr)
        Ws, Wvv, Wvs = filters.split(3, axis=1)
        
        num_edges = phi_vv.shape[0]
        
        # (Equation 7)
        delta_s = scatter_sum(phi_s * Ws.view(num_edges, self.num_features), Tensor(dst), dim_size=s.shape[0])
        
        # (Equation 8)
        phi_vv_3d = phi_vv.view(num_edges, self.num_features, 1)  
        Wvv_3d = Wvv.view(num_edges, self.num_features, 1)          
        delta_v1 = v_j * phi_vv_3d * Wvv_3d
        delta_v1 = scatter_sum(delta_v1, Tensor(dst), dim_size=v.shape[0])
        
        r_ij = edge_attr / (edge_attr.norm(axis=-1, keepdims=True) + 1e-10)
        phi_vs_3d = phi_vs.view(num_edges, self.num_features, 1)  
        Wvs_3d = Wvs.view(num_edges, self.num_features, 1)  
        r_ij_3d = r_ij.view(num_edges, 1, 3) 
        delta_v2 = phi_vs_3d * Wvs_3d * r_ij_3d
        delta_v2 = scatter_sum(delta_v2, Tensor(dst), dim_size=v.shape[0])
        
        delta_v = delta_v1 + delta_v2
        
        return delta_s, delta_v

    def parameters(self):
        return list(self.conv_filter.parameters()) + list(self.phi.parameters())

    def named_parameters(self):
        return (
            [('conv_filter.' + name, param) for name, param in self.conv_filter.named_parameters()] +
            [('phi.' + name, param) for name, param in self.phi.named_parameters()]
        )

    def _debug_shapes(self, s, v, edge_index, edge_attr):
        src, dst = edge_index.data[0], edge_index.data[1]
        s_j = s[Tensor(src)]
        v_j = v[Tensor(src)]
        phi_output = self.phi(s_j)
        phi_s, phi_vv, phi_vs = phi_output.split(3, axis=-1)
        filters = self.conv_filter(s_j, edge_index, edge_attr)
        Ws, Wvv, Wvs = filters.split(3, axis=1)
        
        print(f"s shape: {s.shape}")
        print(f"v shape: {v.shape}")
        print(f"edge_index shape: {edge_index.shape}")
        print(f"edge_attr shape: {edge_attr.shape}")
        print(f"s_j shape: {s_j.shape}")
        print(f"v_j shape: {v_j.shape}")
        print(f"phi_output shape: {phi_output.shape}")
        print(f"phi_s shape: {phi_s.shape}")
        print(f"phi_vv shape: {phi_vv.shape}")
        print(f"phi_vs shape: {phi_vs.shape}")
        print(f"filters shape: {filters.shape}")
        print(f"Ws shape: {Ws.shape}")
        print(f"Wvv shape: {Wvv.shape}")
        print(f"Wvs shape: {Wvs.shape}")
        
        num_edges = phi_vv.shape[0]
        phi_vv_3d = phi_vv.view(num_edges, self.num_features, 1)
        Wvv_3d = Wvv.view(num_edges, self.num_features, 1)
        phi_vs_3d = phi_vs.view(num_edges, self.num_features, 1)
        Wvs_3d = Wvs.view(num_edges, self.num_features, 1)
        r_ij = edge_attr / (edge_attr.norm(axis=-1, keepdims=True) + 1e-10)
        r_ij_3d = r_ij.view(num_edges, 1, 3)
        
        print(f"phi_vv_3d shape: {phi_vv_3d.shape}")
        print(f"Wvv_3d shape: {Wvv_3d.shape}")
        print(f"phi_vs_3d shape: {phi_vs_3d.shape}")
        print(f"Wvs_3d shape: {Wvs_3d.shape}")
        print(f"r_ij shape: {r_ij.shape}")
        print(f"r_ij_3d shape: {r_ij_3d.shape}")
        
        print(f"delta_s shape: {scatter_sum(phi_s * Ws.view(num_edges, self.num_features), Tensor(dst), dim_size=s.shape[0]).shape}")
        print(f"delta_v1 intermediate shape: {(v_j * phi_vv_3d * Wvv_3d).shape}")
        print(f"delta_v1 shape: {scatter_sum(v_j * phi_vv_3d * Wvv_3d, Tensor(dst), dim_size=v.shape[0]).shape}")
        print(f"delta_v2 shape: {scatter_sum(phi_vs_3d * Wvs_3d * r_ij_3d, Tensor(dst), dim_size=v.shape[0]).shape}")

        print(f"delta_s shape: {scatter_sum(phi_s * Ws.view(num_edges, self.num_features), Tensor(dst), dim_size=s.shape[0]).shape}")
        print(f"delta_v1 intermediate shape: {(v_j * phi_vv_3d * Wvv_3d).shape}")
        delta_v1 = scatter_sum(v_j * phi_vv_3d * Wvv_3d, Tensor(dst), dim_size=v.shape[0])
        print(f"delta_v1 shape: {delta_v1.shape}")
        delta_v2 = scatter_sum(phi_vs_3d * Wvs_3d * r_ij_3d, Tensor(dst), dim_size=v.shape[0])
        print(f"delta_v2 shape: {delta_v2.shape}")
        print(f"delta_v shape: {(delta_v1 + delta_v2).shape}")

class UpdateBlock(Module):
    @with_weight_init()
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.U = Linear3D(3, num_features, bias=False)
        self.V = Linear3D(3, num_features, bias=False)
        self.a = Sequential(
            Linear(num_features * 2, num_features),
            shifted_softplus,
            Linear(num_features, num_features * 3)
        )

    def __call__(self, s: Tensor, v: Tensor, delta_s: Tensor, delta_v: Tensor) -> Tuple[Tensor, Tensor]:
        print(f"Input shapes: s: {s.shape}, v: {v.shape}, delta_s: {delta_s.shape}, delta_v: {delta_v.shape}")

        
        U_v = self.U(v)
        V_v = self.V(v)
        print(f"U_v shape: {U_v.shape}, V_v shape: {V_v.shape}")

        V_v_norm = V_v.norm(axis=-1)
        print(f"V_v_norm shape: {V_v_norm.shape}")

        a_input = s.concatenate([V_v_norm], axis=-1)
        print(f"a_input shape: {a_input.shape}")

        a_output = self.a(a_input)
        print(f"a_output shape: {a_output.shape}")

        ass, asv, avv = a_output.split(3, axis=-1)
        print(f"ass shape: {ass.shape}, asv shape: {asv.shape}, avv shape: {avv.shape}")

        UV_scalar_prod = (U_v * V_v).sum(axis=-1)
        print(f"UV_scalar_prod shape: {UV_scalar_prod.shape}")

  
        delta_s_u = ass * s + asv * UV_scalar_prod
        print(f"delta_s_u shape: {delta_s_u.shape}")

        delta_v_u = avv.unsqueeze(-1) * U_v
        
        delta_v_u_proj = delta_v_u.matmul(self.U.W_linear.transpose(1, 0))
        print(f"delta_v_u_proj shape: {delta_v_u_proj.shape}")

    
        s_out = s + delta_s_u + delta_s
        v_out = v + delta_v_u_proj + delta_v
        print(f"Final output shapes: s_out: {s_out.shape}, v_out: {v_out.shape}")

        return s_out, v_out

    def parameters(self):
        return (
            list(self.U.parameters()) +
            list(self.V.parameters()) +
            list(self.a.parameters())
        )

    def named_parameters(self):
        return (
            [('U.' + name, param) for name, param in self.U.named_parameters()] +
            [('V.' + name, param) for name, param in self.V.named_parameters()] +
            [('a.' + name, param) for name, param in self.a.named_parameters()]
        )

def rbf_expansion(distances: Tensor, num_rbf: int, cutoff: float) -> Tensor:
    centers = Tensor(np.linspace(0, cutoff, num_rbf))
    distances = distances.norm(axis=-1, keepdims=True)
    return (-0.5 * ((distances.unsqueeze(-1) - centers) / (cutoff / num_rbf))**2).exp()

def cosine_cutoff(r: Tensor, cutoff: float) -> Tensor:
    return 0.5 * (Tensor(np.cos(np.pi * r.data / cutoff)) + 1.0) * (r < cutoff)


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

