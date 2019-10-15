import numpy as np
from skimage import transform

'''
a = np.load('gt_eigen_depths.npy')
na = []
for i in range(len(a)):
	na.append(transform.downscale_local_mean(np.array([a[i]]), factors=(1, 3, 3))[0])
np.save('gt_eigen_depths_scaled.npy', na)

b = np.load('gt_eigen_depths.npy')
nb = []
for i in range(len(b)):
	nb.append(transform.downscale_local_mean(np.array([b[i]]), factors=(1, 3, 3))[0])
np.save('gt_eigen_masked_depths_scaled.npy', nb)
'''

c = np.load('eigen_imgs.npy')
nc = []
for i in range(len(c)):
	tmp = transform.downscale_local_mean(c[i], factors=(3, 3, 1))
	nc.append(tmp.astype(np.float32))

np.save('eigen_imgs_scaled.npy', nc)

