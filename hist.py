import numpy as np
from scipy import  misc
from scipy import  ndimage
from matplotlib import pyplot as plt

image = misc.imread('unity.png', mode="L")

hist, bin_edges = np.histogram(image, bins='auto')
bin_centres = 0.5*(bin_edges[:1] + bin_edges[1:])

im_min = image.min()
im_max = image.max()
im_mean = image.mean()

print ("Min : ",im_min)
print ("Max : ",im_max)
print ("Mean : ",im_mean)

plt.plot(bin_centres, hist, lw=2)
plt.show()