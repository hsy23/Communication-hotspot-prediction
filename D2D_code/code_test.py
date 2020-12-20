import numpy as np

d2d_net_key = np.load("D2D__del1_keyinfo.npy")
d2d_net_per = np.load("D2D__del1_perinfo.npy")

print(d2d_net_key[0][2])
print(d2d_net_per)