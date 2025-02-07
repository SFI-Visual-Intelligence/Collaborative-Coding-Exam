import numpy as np 

a = np.random.rand(28,28,3)
a = np.mean(a, axis=2, keepdims=True)
print(a.shape)