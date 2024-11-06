import cv2
from scipy import misc
import util
import os.path
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
filename =  sys.argv[1]
img = misc.imread(filename)
img = cv2.GaussianBlur(img, (17, 17), 0)
plt.imsave('./fig/blurred.png', img, cmap = cm.gray)