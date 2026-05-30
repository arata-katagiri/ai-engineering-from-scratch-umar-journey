from functools import reduce
from itertools import product
import numpy as np

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
            if len(shape) == 1:
                return repr(data)
            
            chunk_size = reduce(lambda a, b: a * b, shape[1:], 1)
            chunks = [data[i*chunk_size : (i+1)*chunk_size] for i in range(shape[0])]
            inner = [_build(chunk, shape[1:]) for chunk in chunks]
            return '[' + ', '.join(inner) + ']'
        
        return f"Tensor({_build(self._data, self._shape)})"
    
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
        new_shape.insert(index, 1)
        return self.reshape(*new_shape)
    
    def transpose(self, *trans):
        new_data = []
        new_shape = []
        for i in trans:
            new_shape.append(self._shape[i])
        
        for idx in list(product(*[range(s) for s in new_shape])):
            new_data = self[*idx]

t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))

t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)