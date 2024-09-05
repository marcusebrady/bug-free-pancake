from typing import Tuple
from autograd import Tensor
from layers import Module, Linear, MessageBlock, UpdateBlock

class PaiNNLayer(Module):
    def __init__(self, num_features: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.message_block = MessageBlock(num_features, num_rbf, cutoff)
        self.update_block = UpdateBlock(num_features)

    def __call__(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        delta_s, delta_v = self.message_block(s, v, edge_index, edge_attr)
        s_out, v_out = self.update_block(s, v, delta_s, delta_v)
        return s_out, v_out

class PaiNN(Module):
    def __init__(self, num_features: int, num_layers: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.layers = [PaiNNLayer(num_features, num_rbf, cutoff) for _ in range(num_layers)]

    def __call__(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            s, v = layer(s, v, edge_index, edge_attr)
        return s, v

class ScalarOutputNetwork(Module):
    def __init__(self, num_features: int, num_outputs: int):
        super().__init__()
        self.linear1 = Linear(num_features, num_features)
        self.linear2 = Linear(num_features, num_outputs)

    def __call__(self, s: Tensor) -> Tensor:
        x = self.linear1(s).relu()
        return self.linear2(x)

class VectorOutputNetwork(Module):
    def __init__(self, num_features: int, num_outputs: int):
        super().__init__()
        self.linear1 = Linear(num_features, num_features)
        self.linear2 = Linear(num_features, num_outputs * 3)
        self.num_outputs = num_outputs

    def __call__(self, v: Tensor) -> Tensor:
        x = self.linear1(v.sum(axis=-1)).relu()
        return self.linear2(x).view(*x.shape[:-1], self.num_outputs, 3)

class CompletePaiNN(Module):
    def __init__(self, num_features: int, num_layers: int, num_rbf: int, cutoff: float, 
                 num_scalar_outputs: int, num_vector_outputs: int):
        super().__init__()
        self.painn = PaiNN(num_features, num_layers, num_rbf, cutoff)
        self.scalar_output = ScalarOutputNetwork(num_features, num_scalar_outputs)
        self.vector_output = VectorOutputNetwork(num_features, num_vector_outputs)

    def __call__(self, s: Tensor, v: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        s, v = self.painn(s, v, edge_index, edge_attr)
        scalar_pred = self.scalar_output(s)
        vector_pred = self.vector_output(v)
        return scalar_pred, vector_pred
