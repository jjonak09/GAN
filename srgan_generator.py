from keras.models import load_model
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import cv2
from skimage.io import imsave
import tensorflow as tf

z_dim = 100
SAVE_DIR = "result_img/"

# DRAGAN

# model = load_model('model-1000-epoch.h5')
# noise = np.random.normal(size=(1, z_dim)).astype('float32')
# gen_imgs = model.predict(noise)
# gen_imgs = ((gen_imgs - gen_imgs.min()) * 255 /
#             (gen_imgs.max() - gen_imgs.min())).astype(np.uint8)
# gen_imgs = gen_imgs.reshape(1, -1, 64, 64, 3).swapaxes(1,
#                                                        2).reshape(1*64, -1, 3)
# plt.imshow(Image.fromarray(gen_imgs))
# plt.imsave('srgan/result64.jpg', gen_imgs)
# plt.close()

# SRGAN

model = load_model('model-16000epoch.h5')
image = []
img = Image.open('result-1.jpg')
# img = Image.open('result_img/images.png')

img = img.convert('RGB')
img = np.array(img).astype(np.float)
image.append(img)
image = np.array(image) / 127.5 - 1
print(image.shape)
gen_image = model.predict(image)

gen_image = ((gen_image - gen_image.min()) * 255 /
             (gen_image.max() - gen_image.min())).astype(np.uint8)
gen_image = gen_image.reshape(1, -1, 256, 256, 3).swapaxes(1,
                                                           2).reshape(1 * 256, -1, 3).astype(np.uint8)
plt.imshow(Image.fromarray(gen_image))
plt.imsave('srgan/result256.jpg', gen_image)
plt.close()
