import numpy as np
import timeit
import matplotlib.pyplot as plt
from autograd import Tensor, tensor, randn, zeros
from typing import Callable, List, Tuple
import statistics
import os
import csv

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Comparisons will be skipped.")

def time_function(func: Callable, *args, number=1000, repeat=5) -> Tuple[float, float, float]:
    times = timeit.repeat(lambda: func(*args), number=number, repeat=repeat)
    return min(times) / number, statistics.mean(times) / number, statistics.stdev(times) / number

def compare_tensors(a: Tensor, b: np.ndarray, tolerance=1e-5) -> bool:
    return np.allclose(a.data, b, atol=tolerance)

def benchmark_operation(operation: str, custom_op: Callable, torch_op: Callable, 
                        sizes: List[int], num_runs: int = 5) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    custom_times_min, custom_times_mean, custom_times_std = [], [], []
    torch_times_min, torch_times_mean, torch_times_std = [], [], []

    for size in sizes:
        print(f"Benchmarking {operation} with size {size}x{size}")
        a = randn(size, size)
        b = randn(size, size)

        custom_min, custom_mean, custom_std = time_function(custom_op, a, b, number=num_runs)
        custom_times_min.append(custom_min)
        custom_times_mean.append(custom_mean)
        custom_times_std.append(custom_std)

        if TORCH_AVAILABLE:
            a_torch = torch.tensor(a.data, requires_grad=True)
            b_torch = torch.tensor(b.data, requires_grad=True)
            torch_min, torch_mean, torch_std = time_function(torch_op, a_torch, b_torch, number=num_runs)
            torch_times_min.append(torch_min)
            torch_times_mean.append(torch_mean)
            torch_times_std.append(torch_std)

    return custom_times_min, custom_times_mean, custom_times_std, torch_times_min, torch_times_mean, torch_times_std

def plot_results(sizes: List[int], custom_times: List[float], torch_times: List[float], 
                 custom_std: List[float], torch_std: List[float], operation: str):
    plt.figure(figsize=(12, 6))
    plt.errorbar(sizes, custom_times, yerr=custom_std, label='Custom Autograd', marker='o', capsize=5)
    if TORCH_AVAILABLE:
        plt.errorbar(sizes, torch_times, yerr=torch_std, label='PyTorch', marker='s', capsize=5)
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title(f'Performance Comparison: {operation}')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    

    os.makedirs('autograd_bench', exist_ok=True)
    plt.savefig(f'autograd_bench/{operation.lower().replace(" ", "_")}_comparison.png')
    plt.close()

def save_results_to_csv(sizes: List[int], custom_times: List[float], torch_times: List[float], operation: str):
    os.makedirs('autograd_bench', exist_ok=True)
    with open(f'autograd_bench/{operation.lower().replace(" ", "_")}_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Size', 'Custom Autograd', 'PyTorch'])
        for size, custom, torch in zip(sizes, custom_times, torch_times):
            writer.writerow([size, custom, torch if TORCH_AVAILABLE else 'N/A'])

def test_addition():
    print("\nTesting Addition:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a + b
    c.backward()
    assert compare_tensors(c, np.array([[6, 8], [10, 12]]))
    assert compare_tensors(a.grad, np.ones_like(a.data))
    assert compare_tensors(b.grad, np.ones_like(b.data))
    print("Addition test passed")

def test_multiplication():
    print("\nTesting Multiplication:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a * b
    c.backward()
    assert compare_tensors(c, np.array([[5, 12], [21, 32]]))
    assert compare_tensors(a.grad, b.data)
    assert compare_tensors(b.grad, a.data)
    print("Multiplication test passed")

def test_matmul():
    print("\nTesting Matrix Multiplication:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a.matmul(b)
    c.backward()
    assert compare_tensors(c, np.array([[19, 22], [43, 50]]))
    assert compare_tensors(a.grad, np.array([[11, 15], [11, 15]]))
    assert compare_tensors(b.grad, np.array([[4, 4], [6, 6]]))
    print("Matrix multiplication test passed")

def test_broadcasting():
    print("\nTesting Broadcasting:")
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([1, 2])
    c = a + b
    c.backward()
    assert compare_tensors(c, np.array([[2, 4], [4, 6]]))
    assert compare_tensors(a.grad, np.ones_like(a.data))
    assert compare_tensors(b.grad, np.array([2, 2]))
    print("Broadcasting test passed")

def test_tanh():
    print("\nTesting Tanh:")
    a = Tensor([-2, -1, 0, 1, 2])
    b = a.tanh()
    b.backward()
    expected_data = np.tanh(a.data)
    expected_grad = 1 - np.tanh(a.data)**2
    assert compare_tensors(b, expected_data)
    assert compare_tensors(a.grad, expected_grad)
    print("Tanh test passed")

def run_benchmarks():
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    operations = [
        ("Addition", lambda a, b: a + b, lambda a, b: a + b),
        ("Multiplication", lambda a, b: a * b, lambda a, b: a * b),
        ("Matrix Multiplication", lambda a, b: a.matmul(b), lambda a, b: torch.matmul(a, b)),
        ("Tanh", lambda a, b: a.tanh(), lambda a, b: torch.tanh(a)),
        ("Broadcasting Addition", lambda a, b: a + b[:, 0].unsqueeze(1), lambda a, b: a + b[:, 0].unsqueeze(1)),
    ]

    for op_name, custom_op, torch_op in operations:
        custom_min, custom_mean, custom_std, torch_min, torch_mean, torch_std = benchmark_operation(op_name, custom_op, torch_op, sizes)
        plot_results(sizes, custom_mean, torch_mean, custom_std, torch_std, op_name)
        save_results_to_csv(sizes, custom_mean, torch_mean, op_name)


    plt.figure(figsize=(15, 10))
    for i, (op_name, _, _) in enumerate(operations):
        custom_times = []
        torch_times = []
        with open(f'autograd_bench/{op_name.lower().replace(" ", "_")}_results.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  
            for row in reader:
                custom_times.append(float(row[1]))
                if TORCH_AVAILABLE:
                    torch_times.append(float(row[2]))
        plt.plot(sizes, custom_times, label=f'{op_name} (Custom)', marker='o')
        if TORCH_AVAILABLE:
            plt.plot(sizes, torch_times, label=f'{op_name} (PyTorch)', linestyle='--', marker='s')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: All Operations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('autograd_bench/all_operations_comparison.png')
    plt.close()

if __name__ == "__main__":
    test_addition()
    test_multiplication()
    test_matmul()
    test_broadcasting()
    test_tanh()
    print("\nAll tests passed!")

    run_benchmarks()
    print("\nBenchmarking completed. Check 'plots' directory for performance graphs and 'results' directory for CSV files.")
