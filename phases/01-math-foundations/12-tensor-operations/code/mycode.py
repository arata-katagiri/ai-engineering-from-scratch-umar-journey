from functools import reduce
from itertools import product
import numpy as np
import torch

class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)

            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)
    
    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        flat_index = sum(i * stride for i, stride in zip(indices, self._strides))

        return self._data[flat_index]
    
    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)

        flat_index = sum(i * stride for i, stride in zip(indices, self._strides))

        self._data[flat_index] = value
    
    def __repr__(self) -> str:
        def _build(data, shape):
            if len(shape) == 0:
                return repr(data[0])
            if len(shape) == 1:
                return repr(data)
            
            chunk_size = reduce(lambda a, b: a * b, shape[1:], 1)
            chunks = [data[i*chunk_size : (i+1)*chunk_size] for i in range(shape[0])]
            inner = [_build(chunk, shape[1:]) for chunk in chunks]
            return '[' + ', '.join(inner) + ']'
        
        return f"Tensor({_build(self._data, self._shape)})"
    
    def __add__(self, other):
        sum_data = []
        if isinstance(other, Tensor):
            for i in range(len(self._data)):
                sum_data.append(self._data[i] + other._data[i])
        else:
            for i in range(len(self._data)):
                sum_data.append(self._data[i] + other)
        
        return Tensor(sum_data, self._shape)

    def __mul__(self, other):
        mul_data = []
        if isinstance(other, Tensor):
            for i in range(len(self._data)):
                mul_data.append(self._data[i] * other._data[i])
        else:
            for i in range(len(self._data)):
                mul_data.append(self._data[i] * other)
        
        return Tensor(mul_data, self._shape)
    
    def __sub__(self, other):
        return self + other * (-1)
    
    def tensorsum(self, axis):
        order = list(range(len(self._shape)))
        order.pop(axis)
        order.insert(0, axis)
        permuted = self.permute(order)
        result_shape = permuted._shape[1:]
        result = []
        for idx in product(*[range(s) for s in result_shape]):
            total = sum(permuted[(i,) + idx] for i in range(permuted._shape[0]))
            result.append(total)

        return Tensor(result, list(result_shape))
    
    def mean(self, axis):
        return self.tensorsum(axis) * (1 / self._shape[axis])
    
    def tensormax(self, axis):
        order = list(range(len(self._shape)))
        order.pop(axis)
        order.insert(0, axis)
        permuted = self.permute(order)
        result_shape = permuted._shape[1:]
        result = []
        for idx in product(*[range(s) for s in result_shape]):
            total = max(permuted[(i,) + idx] for i in range(permuted._shape[0]))
            result.append(total)
        
        return Tensor(result, list(result_shape))
    
    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i+1] * shape[i+1]
        return tuple(strides)
    
    def _flatten_nested(self, data):
        flat_data = []
        shape = []
        data_len = []
        for i in range(len(data)):
            if isinstance(data[i], list):
                data[i], data_len = self._flatten_nested(data[i])
                flat_data.extend(data[i])
                
            else:
                flat_data.append(data[i])

        shape.insert(0, len(data))
        shape.extend(data_len)
        
        return flat_data, shape
    
    def reshape(self, *new_shape):
        tensor = Tensor(self._data, new_shape)

        return tensor
    
    def squeeze(self):
        new_shape = tuple(x for x in self._shape if x != 1)
        return self.reshape(*new_shape)
    
    def unsqueeze(self, index):
        new_shape = list(self._shape)
        new_shape.insert(index, 1) # type: ignore
        return self.reshape(*new_shape)
    
    def permute(self, *trans):
        new_data = []
        new_shape = []
        for i in trans:
            new_shape.append(self._shape[i])
        
        for idx in list(product(*[range(s) for s in new_shape])):
            old_idx = [0] * len(trans)
            for i, dim in enumerate(trans):
                old_idx[dim] = idx[i]
            new_data.append(self[tuple(old_idx)])
        
        return Tensor(new_data, new_shape)
    
    def transpose(self, *trans):
        order = list(range(len(self._shape)))
        order[trans[0]], order[trans[1]] = order[trans[1]], order[trans[0]]
        return self.permute(*order)

def einsum(string, *tensors):
    splitted = string.split('->')
    inputs = splitted[0].split(',')
    output = splitted[1]

    all_input = ''.join(inputs)
    sum_indices = set(all_input) - set(output)

    size_dict = {}
    for input, tensor in zip(inputs, tensors):
        for pos, letter in enumerate(input):
            size_dict[letter] = tensor._shape[pos]

    output_shape = tuple(size_dict[l] for l in output)
    result = [0] * reduce(lambda a, b: a*b, output_shape, 1)

    all_indices = ''.join(set(all_input))
    for combo in product(*[range(size_dict[l]) for l in all_indices]):
        idx = dict(zip(all_indices, combo))
        val = 1

        out_idx = tuple(idx[letter] for letter in output)
        flat_out = sum(i * s for i, s in zip(out_idx, Tensor._compute_strides(output_shape)))

        for tensor, spec in zip(tensors, inputs):
            element_idx = tuple(idx[letter] for letter in spec)
            val *= tensor[element_idx]
        
        result[flat_out] += val

    return Tensor(result, list(output_shape))

def softmax(tensor):
    row_size = tensor._shape[-1]
    new_data = []
    for i in range(0, len(tensor._data), row_size):
        row = tensor._data[i:i+row_size]
        new_row = []
        total = sum([np.exp(row[j]) for j in range(len(row))])
        for i in range(len(row)):
            new_row.append(np.exp(row[i]) / total)
        
        new_data.extend(new_row)
    
    return Tensor(new_data, tensor._shape)

def attention(Q, K, V):
    # QK = einsum("ij,kj->ik", Q, K)
    # soft = softmax(QK * (1 / K._shape[-1] ** 0.5))
    # return einsum("ik,kj->ij", soft, V)

    QK = einsum("bhtd, bhsd->bhts", Q, K)
    soft = softmax(QK * (1 / K._shape[-1] ** 0.5))
    return einsum("bhts, bhsd->bhtd", soft, V)

def attention_shape_tracker(batch_size, seq_len, embed_dim, num_heads):
    print("Input X:", [batch_size, seq_len, embed_dim])
    print("Q/K/V projection:", [batch_size, seq_len, embed_dim])
    print("Head split:", [batch_size, seq_len, num_heads, embed_dim // num_heads])
    print("After transpose:", [batch_size, num_heads, seq_len, embed_dim // num_heads])
    print("Attention scores (QK):", [batch_size, num_heads, seq_len, seq_len])
    print("Softmax weights:", [batch_size, num_heads, seq_len, seq_len])
    print("Weighted sum:", [batch_size, num_heads, seq_len, embed_dim // num_heads])
    print("Head merge:", [batch_size, seq_len, embed_dim])
    print("Output projection:", [batch_size, seq_len, embed_dim])

# t = Tensor(list(range(12)), shape=(2, 6))
# r = t.reshape((3, 4))
# r = t.reshape((-1, 3))

# t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
# s = t.squeeze()
# v = Tensor([1, 2, 3])
# u = v.unsqueeze(0)

# mat = Tensor(list(range(6)), shape=(2, 3))
# tr = mat.transpose(0, 1)

# t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
# perm = t4d.permute((0, 2, 3, 1))

# a = Tensor([[1, 2], [3, 4]])
# b = Tensor([[10, 20], [30, 40]])
# c = a + b
# d = a * 2
# s = a.tensorsum(axis=0)

# activations = np.random.randn(4, 3)
# bias = np.array([0.1, 0.2, 0.3])
# result = activations + bias

# images = np.random.randn(2, 3, 4, 4)
# scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
# result = images * scale

# a = np.array([1, 2, 3]).reshape(-1, 1)
# b = np.array([10, 20, 30, 40]).reshape(1, -1)
# outer = a * b

# a = np.array([1.0, 2.0, 3.0])
# b = np.array([4.0, 5.0, 6.0])
# dot = np.einsum("i,i->", a, b)

# A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
# B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
# matmul = np.einsum("ik,kj->ij", A, B)

# batch_A = np.random.randn(4, 3, 5)
# batch_B = np.random.randn(4, 5, 2)
# batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)

# B, H, T, D = 2, 4, 8, 16
# E = H * D

# X = np.random.randn(B, T, E)
# W_q = np.random.randn(E, E) * 0.02

# Q = np.einsum("bte,ek->btk", X, W_q)
# Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

# scores = np.einsum("bhtd, bhsd->bhts", Q, K) / np.sqrt(D)
# weights = softmax(scores, axis=-1)
# attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

# concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
# output = np.einsum("bte,ek->btk", concat, W_o)

attention_shape_tracker(2, 8, 256, 4)

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(t.shape)
print(t.stride())
print(t.is_contiguous())

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj-ij", A, B)