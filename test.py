import numpy as np
import matplotlib.pyplot as plt

def RELU(z_mtx):
    out = np.array(np.zeros(z_mtx.size)).reshape(z_mtx.shape)
    out.reshape(z_mtx.shape)
    for idx in range(0, out.shape[0]):
        out[idx, 0] = max(z_mtx[idx, 0], 0)
    return out

def soft_max(z_mtx):
    sum = 0.0
    sigma_vct = []
    for z in z_mtx:
        sum += np.exp(z)
    for z in z_mtx:
        sigma_vct.append(z / sum)
    return np.array(sigma_vct)

a = np.array([1, 2, 3, 4, 5], np.float64)
b = np.array(np.ones(5))
# a = a.astype(np.float)

print(a)


#
# for itm in a:
#     print(itm)

# w1_mtx = np.array(np.zeros((10, 2)), np.float64)
