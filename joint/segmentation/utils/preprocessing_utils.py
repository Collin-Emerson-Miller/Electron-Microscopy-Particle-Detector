from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import re
import shutil

import numpy as np
import math
import matplotlib.pyplot as plt


def load_image(foldername, name, img_path, gt_path):
    
    if os.path.exists(foldername):
        shutil.rmtree(foldername)

    os.makedirs(foldername)
    
    img = cv2.imread(img_path, 0)
    gt = cv2.imread(gt_path, 0)
    
    plt.imsave(os.path.join(foldername, "img"), img, cmap='gray')
    plt.imsave(os.path.join(foldername, "gt"), gt, cmap='gray')    

    os.system("gmic -v -1 " + gt_path + " -channels 1 -threshold 10% -negative -label_fg 0,0 -o -.asc | tail -n +2 | awk '{ for (i = 1; i<=NF; i++) {x[$i] += i; y[$i] += NR; n[$i]++; } } END { for (v in x) print x[v]/n[v],y[v]/n[v] }' > " + foldername + "/seeds.txt")

    seeds = []
    f = open(foldername + "/seeds.txt", 'r')
    for line in f:
        y = int(float(re.split(' ', line)[0]))
        x = int(float(re.split(' ', line)[1]))
        seed = (x - 1, y - 1)
        seeds.append(seed)

    seeds = seeds[1:]
    
    return img, gt, seeds



def crop_2d(image, top_left_corner, height, width):
    """
    Returns a crop of an image.

    Args:
        image: The original image to be cropped.
        top_left_corner: The coordinates of the top left corner of the image.
        height: The hight of the crop.
        width: The width of the crop.

    Returns:
        A cropped version of the original image.
    """

    x_start = top_left_corner[0]
    y_start = top_left_corner[1]
    x_end = x_start + width
    y_end = y_start + height

    return image[x_start:x_end, y_start:y_end, ...]


# In[ ]:


def pad_for_window(img, height, width, padding_type='reflect'):
    npad = ((height // 2, width // 2), (height // 2, width // 2), (0, 0))
    return np.pad(img, npad, padding_type)


# In[ ]:


def prepare_input_images(img, height=15, width=15):
    """
    Preprocess images to be used in the prediction of the edges.

    Args:
        image (numpy.array):
    """

    # Standardize input
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)

    padded_image = pad_for_window(img, height, width)

    images = []

    for index in np.ndindex(img.shape[:-1]):
        images.append(crop_2d(padded_image, index, height, width))

    return np.stack(images)


# In[ ]:


def create_batches(x, max_batch_size=32):
    """

    Args:
        x: A numpy array of the input data
        y: A numpy array of the output
        max_batch_size: The maximum elements in each batch.

    Returns: A list of batches.

    """

    batches = math.ceil(x.shape[0] / max_batch_size)
    x = np.array_split(x, batches)

    return x

