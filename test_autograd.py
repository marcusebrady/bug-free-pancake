import numpy as np
import time
from autograd import Tensor, tensor, randn, zeros

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Comparisons will be skipped.")

def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} execution time: {execution_time:.6f} seconds")
        return (result, execution_time)
    return wrapper

@time_function
def test_addition():
    print("\nTesting Addition:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a + b
    c.backward()
    print(f"a: {a.data}")
    print(f"b: {b.data}")
    print(f"c (a + b): {c.data}")
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")
    assert np.allclose(c.data, np.array([[6, 8], [10, 12]]))
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.ones_like(b.data))
    print("Addition test passed")

    if TORCH_AVAILABLE:
        print("\nPyTorch Comparison:")
        start_time = time.time()
        a_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        b_torch = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
        c_torch = a_torch + b_torch
        c_torch.sum().backward()
        end_time = time.time()
        torch_time = end_time - start_time
        print(f"PyTorch result: {c_torch.detach().numpy()}")
        print(f"PyTorch gradient of a: {a_torch.grad.numpy()}")
        print(f"PyTorch gradient of b: {b_torch.grad.numpy()}")
        print(f"PyTorch execution time: {torch_time:.6f} seconds")
        return torch_time
    return None

@time_function
def test_multiplication():
    print("\nTesting Multiplication:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a * b
    c.backward()
    print(f"a: {a.data}")
    print(f"b: {b.data}")
    print(f"c (a * b): {c.data}")
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")
    assert np.allclose(c.data, np.array([[5, 12], [21, 32]]))
    assert np.allclose(a.grad, b.data)
    assert np.allclose(b.grad, a.data)
    print("Multiplication test passed")

    if TORCH_AVAILABLE:
        print("\nPyTorch Comparison:")
        start_time = time.time()
        a_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        b_torch = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
        c_torch = a_torch * b_torch
        c_torch.sum().backward()
        end_time = time.time()
        torch_time = end_time - start_time
        print(f"PyTorch result: {c_torch.detach().numpy()}")
        print(f"PyTorch gradient of a: {a_torch.grad.numpy()}")
        print(f"PyTorch gradient of b: {b_torch.grad.numpy()}")
        print(f"PyTorch execution time: {torch_time:.6f} seconds")
        return torch_time
    return None

@time_function
def test_matmul():
    print("\nTesting Matrix Multiplication:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a.matmul(b)
    c.backward()
    print(f"a: {a.data}")
    print(f"b: {b.data}")
    print(f"c (a.matmul(b)): {c.data}")
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")
    assert np.allclose(c.data, np.array([[19, 22], [43, 50]]))
    assert np.allclose(a.grad, np.array([[11, 15], [11, 15]]))
    assert np.allclose(b.grad, np.array([[4, 4], [6, 6]]))
    print("Matrix multiplication test passed")

    if TORCH_AVAILABLE:
        print("\nPyTorch Comparison:")
        start_time = time.time()
        a_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        b_torch = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
        c_torch = torch.matmul(a_torch, b_torch)
        c_torch.sum().backward()
        end_time = time.time()
        torch_time = end_time - start_time
        print(f"PyTorch result: {c_torch.detach().numpy()}")
        print(f"PyTorch gradient of a: {a_torch.grad.numpy()}")
        print(f"PyTorch gradient of b: {b_torch.grad.numpy()}")
        print(f"PyTorch execution time: {torch_time:.6f} seconds")
        return torch_time
    return None

@time_function
def test_broadcasting():
    print("\nTesting Broadcasting:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([1, 2])
    c = a + b
    c.backward()
    print(f"a: {a.data}")
    print(f"b: {b.data}")
    print(f"c (a + b): {c.data}")
    print(f"Gradient of a: {a.grad}")
    print(f"Gradient of b: {b.grad}")
    assert np.allclose(c.data, np.array([[2, 4], [4, 6]]))
    assert np.allclose(a.grad, np.ones_like(a.data))
    assert np.allclose(b.grad, np.array([2, 2]))
    print("Broadcasting test passed")

    if TORCH_AVAILABLE:
        print("\nPyTorch Comparison:")
        start_time = time.time()
        a_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
        b_torch = torch.tensor([1, 2], dtype=torch.float32, requires_grad=True)
        c_torch = a_torch + b_torch
        c_torch.sum().backward()
        end_time = time.time()
        torch_time = end_time - start_time
        print(f"PyTorch result: {c_torch.detach().numpy()}")
        print(f"PyTorch gradient of a: {a_torch.grad.numpy()}")
        print(f"PyTorch gradient of b: {b_torch.grad.numpy()}")
        print(f"PyTorch execution time: {torch_time:.6f} seconds")
        return torch_time
    return None

@time_function
def test_tanh():
    print("\nTesting Tanh:")
    a = Tensor([-2, -1, 0, 1, 2])
    b = a.tanh()
    b.backward()
    print(f"a: {a.data}")
    print(f"b (tanh(a)): {b.data}")
    print(f"Gradient of a: {a.grad}")
    expected_data = np.tanh(a.data)
    expected_grad = 1 - np.tanh(a.data)**2
    assert np.allclose(b.data, expected_data)
    assert np.allclose(a.grad, expected_grad)
    print("Tanh test passed")

    if TORCH_AVAILABLE:
        print("\nPyTorch Comparison:")
        start_time = time.time()
        a_torch = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, requires_grad=True)
        b_torch = torch.tanh(a_torch)
        b_torch.sum().backward()
        end_time = time.time()
        torch_time = end_time - start_time
        print(f"PyTorch result: {b_torch.detach().numpy()}")
        print(f"PyTorch gradient of a: {a_torch.grad.numpy()}")
        print(f"PyTorch execution time: {torch_time:.6f} seconds")
        return torch_time
    return None

if __name__ == "__main__":
    (torch_add_time, add_time) = test_addition()
    (torch_mul_time, mul_time) = test_multiplication()
    (torch_matmul_time, matmul_time) = test_matmul()
    (torch_broadcast_time, broadcast_time) = test_broadcasting()
    (torch_tanh_time, tanh_time) = test_tanh()
    print("\nAll tests passed!")

    if TORCH_AVAILABLE:
        print("\nExecution Time Comparison:")
        print(f"Addition: Custom: {add_time:.6f}s, PyTorch: {torch_add_time:.6f}s")
        print(f"Multiplication: Custom: {mul_time:.6f}s, PyTorch: {torch_mul_time:.6f}s")
        print(f"Matrix Multiplication: Custom: {matmul_time:.6f}s, PyTorch: {torch_matmul_time:.6f}s")
        print(f"Broadcasting: Custom: {broadcast_time:.6f}s, PyTorch: {torch_broadcast_time:.6f}s")
        print(f"Tanh: Custom: {tanh_time:.6f}s, PyTorch: {torch_tanh_time:.6f}s")
