import numpy as np
from autograd import Tensor
from layers import Linear, GraphConv, MessagePassing

def test_linear():
    print("Testing Linear layer...")
    linear = Linear(3, 2)
    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    output = linear(x)
    print(f"Input shape: {x.data.shape}")
    print(f"Output shape: {output.data.shape}")
    print(f"Output: {output.data}")
    assert output.data.shape == (2, 2), "Output shape is incorrect"
    print("Linear layer test passed!")

def test_graph_conv():
    print("\nTesting GraphConv layer...")
    conv = GraphConv(3, 2)
    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    edge_index = Tensor(np.array([[0, 1], [1, 2], [2, 0]]).T)  # 3 edges
    output = conv(x, edge_index)
    print(f"Input shape: {x.data.shape}")
    print(f"Edge index shape: {edge_index.data.shape}")
    print(f"Output shape: {output.data.shape}")
    print(f"Output: {output.data}")
    assert output.data.shape == (3, 2), "Output shape is incorrect"
    print("GraphConv layer test passed!")

if __name__ == "__main__":
    test_linear()
    test_graph_conv()
