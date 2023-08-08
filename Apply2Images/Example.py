# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:35:06 2023

@author: LENOVO
"""

import time
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.data import binary_blobs, \
    camera, horse, astronaut, cat
from skimage.util import random_noise, img_as_float
from skimage.color import rgb2gray
from skimage.io import imread
import Q_p
import test_functions as test
from Delay_CNN2images import DCNN_2_image
# import sys
# sys.path.insert(1, 'D:/Documents/un/python/SC-CNN')
# sys.path.insert(1, 'D:/Documents/un/python/\
#                     SC-CNN/image_example/\
#                         edge_detection/\
#                             grey image')


delta_t = 0.05
t = 6
p = 3
K = 4
Z_k = Q_p.Z_N_group(p, K)


def f(x):
    return 0.5 * (abs(x + 1) - abs(x - 1))


image_saint = 'D:/Documents/un/python/SC-CNN/image_example/edge_detection/grey image/saint_Paul.png'
""" Import and tranform image """
image_read = imread(image_saint)
image = rgb2gray(image_read)

image = img_as_float(resize(image, (3**6, 3**6)))

image = p**K*(image * 2 - 1)
plt.imshow(image, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

""" J, A, B,  and Z  operators """
# J operator


# J = test.char_function(0, 0, p)
J = 0


# Feedback

A = 2*test.char_function(0, -K, p)
# Feedforward


def B(x):
    if Q_p.norm_p(x, p) == 0:
        return (p**(K-2)-1)
    elif 0 < Q_p.norm_p(x, p) <= p**(-K+2):
        return (-1)
    else:
        return 0


# Threshold
def Z(x):
    return -1


def X_0(x, t):
    return 0


start = time.time()
DCNN_2_image(image, J=J, A=A, B=B, Z=Z,
             t=t, delta_t=delta_t,
             Z_k=Z_k, X_0=X_0,
             delay=-2, L=1,
             split_image=True,
             screem_shot=True,
             reduce=False,
             without_label=True)
end = time.time()
print("Time ejecution ", end-start)
