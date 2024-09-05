import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data)
        self.shape = self.data.shape  # Add this line
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor(data={self.data}), requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = None
        for child in self._prev:
            child.zero_grad()

    #inspire:https://github.com/smolorg/smolgrad/blob/master/smolgrad/core/engine.py#L194
    def _preprocess_binop(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Handle scalar multiplication
        if self.shape == () or other.shape == ():
            return self, other

        # Try to broadcast shapes
        try:
            broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {self.shape} and {other.shape}")

        self_broadcasted = self.broadcast_to(broadcast_shape)
        other_broadcasted = other.broadcast_to(broadcast_shape)
        
        return self_broadcasted, other_broadcasted

    def broadcast_to(self, shape):
        if self.shape == shape:
            return self
        
        data = np.broadcast_to(self.data, shape)
        out = Tensor(data, _children=(self,), _op="broadcast")
        
        def _backward():
            axes = tuple(range(len(shape) - len(self.shape))) + \
                   tuple(i for i, (a, b) in enumerate(zip(shape[len(shape) - len(self.shape):], self.shape)) if a != b)
            self.grad += np.sum(out.grad, axis=axes).reshape(self.shape)
        
        out._backward = _backward
        return out

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return Tensor(self.data < other)
        elif isinstance(other, Tensor):
            return Tensor(self.data < other.data)
        else:
            raise TypeError(f"Unsupported operand type for <: '{type(self)}' and '{type(other)}'")

    def view(self, *shape):
        # Ensure that the new shape has the same number of elements as the original tensor
        if np.prod(shape) != np.prod(self.shape):
            raise ValueError(f"Cannot reshape tensor of shape {self.shape} to shape {shape}")
        
        # Reshape the data
        reshaped_data = self.data.reshape(shape)
        out = Tensor(reshaped_data, _children=(self,), _op='view')
        
        # Define the backward pass for the view operation
        def _backward():
            self.grad += out.grad.reshape(self.shape)
        
        out._backward = _backward
        return out


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
        self, other = self._preprocess_binop(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            if self.grad is not None:
                self.grad += other.data * out.grad
            if other.grad is not None:
                other.grad += self.data * out.grad
        out._backward = _backward
        
        return out


    def __rmul__(self, other):
        return self * other
        



    def __sub__(self, other):
        self, other = self._preprocess_binop(other) 
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += np.sum(out.grad, axis=tuple(range(out.grad.ndim - self.data.ndim))) \
                         .reshape(self.data.shape)
            other.grad += np.sum(out.grad, axis=tuple(range(out.grad.ndim - other.data.ndim))) \
                          .reshape(other.data.shape)
        out._backward = _backward
        
        return out
    def expand(self, *shape):
        if len(shape) < len(self.shape):
            raise ValueError(f"Expanded shape {shape} must have at least {len(self.shape)} dimensions")
        
        new_shape = list(shape)
        for i, (new_dim, old_dim) in enumerate(zip(reversed(shape), reversed(self.shape))):
            if new_dim != old_dim and old_dim != 1:
                raise ValueError(f"Expanded shape {shape} is not compatible with current shape {self.shape}")
            if old_dim == 1:
                new_shape[-(i+1)] = new_dim
        
        expanded_data = np.broadcast_to(self.data, new_shape)
        out = Tensor(expanded_data, _children=(self,), _op='expand')
        
        def _backward():
            sum_axes = tuple(i for i, (new_dim, old_dim) in enumerate(zip(new_shape, self.shape)) if new_dim != old_dim)
            self.grad += np.sum(out.grad, axis=sum_axes).reshape(self.shape)
        
        out._backward = _backward
        return out
 

    def pow(self, exponent):
        out = Tensor(np.power(self.data, exponent), (self,), 'pow')
        
        def _backward():
            self.grad += (exponent * np.power(self.data, exponent - 1) * out.grad)
        out._backward = _backward
        
        return out

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        
        return out


    def sqrt(self):
        out = Tensor(np.sqrt(self.data), (self,), 'sqrt')
        
        def _backward():
            self.grad += 0.5 / np.sqrt(self.data) * out.grad
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

    def backward(self, gradient=None):
            if not self.requires_grad:
                return

            if gradient is None:
                gradient = np.ones_like(self.data)
            
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            
            self.grad += gradient

            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)
            
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
        if not axes:
            axes = tuple(reversed(range(len(self.shape))))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        
        # Handle partial axis specification
        if len(axes) < len(self.shape):
            remaining_axes = [i for i in range(len(self.shape)) if i not in axes]
            axes = list(axes) + remaining_axes
        
        if len(axes) != len(self.shape):
            raise ValueError(f"Invalid number of axes. Got {len(axes)}, expected {len(self.shape)}")
        
        out = Tensor(self.data.transpose(*axes), (self,), 'transpose')
        
        def _backward():
            # Compute the inverse permutation
            inverse_axes = [0] * len(axes)
            for i, axis in enumerate(axes):
                inverse_axes[axis] = i
            self.grad += out.grad.transpose(*inverse_axes)
        
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out


    def unsqueeze(self, dim):
        new_shape = list(self.data.shape)
        new_shape.insert(dim, 1)
        out = Tensor(self.data.reshape(new_shape), (self,), 'unsqueeze')
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
    
        return out
    def squeeze(self, dim=None):
        if dim is None:
            new_shape = tuple(s for s in self.data.shape if s != 1)
        else:
            new_shape = list(self.data.shape)
            if new_shape[dim] == 1:
                new_shape.pop(dim)
        
        out = Tensor(self.data.reshape(new_shape), (self,), 'squeeze')
        
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        
        return out

    def concatenate(self, tensors, axis=0):
        tensors = [self] + [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        shape_check = [t.shape[:axis] + t.shape[axis + 1:] for t in tensors]
        if not all(s == shape_check[0] for s in shape_check):
            raise ValueError("All tensors must have the same shape except in the concatenation axis.")
        data = np.concatenate([t.data for t in tensors], axis=axis)
        out = Tensor(data, tensors, 'concatenate')
        
        def _backward():
            grads = [np.zeros_like(t.data) for t in tensors]
            splits = np.cumsum([t.shape[axis] for t in tensors[:-1]])
            split_gradients = np.split(out.grad, splits, axis=axis)
            for i, grad in enumerate(split_gradients):
                grads[i] += grad
            for t, g in zip(tensors, grads):
                t.grad += g

        out._backward = _backward
        return out
    '''
    def norm(self, axis=None, keepdims=False):
        out = Tensor(np.linalg.norm(self.data, axis=axis, keepdims=keepdims), (self,), 'norm')
        
        def _backward():
            if axis is None:
                scale = self.data / (out.data + 1e-8)  # Add small epsilon to avoid division by zero
            else:
                scale = self.data / (np.expand_dims(out.data, axis=axis) + 1e-8)
            self.grad += scale * out.grad
        out._backward = _backward
        
        return out
    '''


    def norm(self, axis=None, keepdims=False):
        # Manually compute the squared values
        squared_data = self.data ** 2

        # Use the custom sum method from the Tensor class
        summed_tensor = Tensor(squared_data).sum(axis=axis, keepdims=keepdims)

        # Compute the square root to get the norm
        norm_data = np.sqrt(summed_tensor.data)

        # Create the output tensor
        out = Tensor(norm_data, (self,), 'norm')

        # Backward method to calculate the gradients
        def _backward():
            if axis is None:
                scale = self.data / (out.data + 1e-8)  # Add small epsilon to avoid division by zero
            else:
                expanded_out_data = np.expand_dims(out.data, axis=axis)
                scale = self.data / (expanded_out_data + 1e-8)
            self.grad += scale * out.grad

        out._backward = _backward

        return out



    def split(self, num_splits, axis=-1):
        splits = np.array_split(self.data, num_splits, axis=axis)
        return [Tensor(split) for split in splits]


    def cross(self, other):
        assert self.data.shape[-1] == 3 and other.data.shape[-1] == 3, "Cross product is only defined for 3D vectors"
        out = Tensor(np.cross(self.data, other.data), (self, other), 'cross')
        def _backward():
            self.grad += np.cross(other.data, out.grad)
            other.grad += np.cross(out.grad, self.data)
        out._backward = _backward
        return out

    def outer(self, other):
        out = Tensor(np.outer(self.data, other.data), (self, other), 'outer')
        def _backward():
            self.grad += np.sum(out.grad * other.data, axis=-1)
            other.grad += np.sum(out.grad * self.data, axis=0)
        out._backward = _backward
        return out
def scatter_mean(src, index, dim_size):
    if isinstance(index, Tensor):
        index = index.data
    if len(src.shape) > 2:
        # Reshape src to 2D if it's higher dimensional
        src_2d = src.view(-1, src.shape[-1])
    else:
        src_2d = src
    out = Tensor(np.zeros((dim_size, src_2d.shape[-1])))
    count = np.zeros(dim_size)
    np.add.at(out.data, index, src_2d.data)
    np.add.at(count, index, 1)
    count[count == 0] = 1  
    out.data /= count[:, None]
    return out



def scatter_sum(src, index, dim_size):
    if isinstance(index, Tensor):
        index = index.data
    if len(src.shape) == 3:
        out = Tensor(np.zeros((dim_size, src.shape[1], src.shape[2])))
    else:
        out = Tensor(np.zeros((dim_size, src.shape[-1])))
    np.add.at(out.data, index, src.data)
    return out

def tensor(data):
    return Tensor(data)

def randn(*shape):
    return Tensor(np.random.randn(*shape))

def zeros(*shape):
    return Tensor(np.zeros(shape))
