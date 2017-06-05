#!usr/bin/env python
#-*- coding: utf-8 -*-
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# settings for LBP
radius = 3
n_points = 8 * radius


# 读取图像
image = cv2.imread('cat.jpg')

#显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(131)
plt.imshow(image1)

# 转换为灰度图显示
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(132)
plt.imshow(image, cmap='gray')

# 处理
lbp = local_binary_pattern(image, n_points, radius)

plt.subplot(133)
plt.imshow(lbp, cmap='gray')
plt.show()

print(lbp)