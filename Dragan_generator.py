from __future__ import print_function, division
import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Cropping2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import cv2

conv_init = RandomNormal(0, 0.02)
z_dim = 100


def build_generator():

    model = Sequential()

    model.add(Dense(512 * 4 * 4, activation="relu",input_dim=z_dim,  kernel_initializer=conv_init))
    model.add(Reshape((4, 4, 512)))
    model.add(Conv2DTranspose(256, 4, strides=2, padding="same",
                              kernel_initializer=conv_init))  # 8x8x256
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(128, 4, strides=2, padding="same",
                              kernel_initializer=conv_init,))  # 16x16x128
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(64, 4,  strides=2, padding="same",
                              kernel_initializer=conv_init))  # 32x32x64
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(3, 4,  strides=2,
                              padding="same",
                              kernel_initializer=conv_init))  # 64x64x3
    model.add(Activation("tanh"))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    model = load_model('./dcgan-2900-epoch.h5')

    return model


def main():
    model = build_generator()

    r, c = 5,5
    noise = np.random.normal(0, 1, (r*c, z_dim))
    gen_imgs = model.predict(noise)
    gen_imgs = (0.5 * gen_imgs + 0.5)


    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("gen_images/epoch2900.png")
    plt.close()

    # # ---------------------
    # # Latent space
    # # ---------------------
    # print("Generating interpolations...")
    # check_noise = np.random.uniform(-1, 1, (5*5, 100))
    # steps = 30
    # latentStart = np.expand_dims(check_noise[5], axis=0)
    # latentEnd = np.expand_dims(check_noise[10], axis=0)

    # # startImg = model.predict(latentStart)
    # # endImg = model.predict(latentEnd)

    # vectors = []

    # alphaValues = np.linspace(0, 1, steps)
    # for alpha in alphaValues:
    #     vector = latentStart * (1 - alpha) + latentEnd * alpha
    #     vectors.append(vector)

    # vectors = np.array(vectors)

    # resultLatent = None
    # resultImage = None

    # for i, vec in enumerate(vectors):
    #     gen_img = np.squeeze(model.predict(vec), axis=0)
    #     gen_img = (0.5 * gen_img + 0.5) * 255
    #     gen_img = gen_img.reshape(
    #         gen_img.shape[1], gen_img.shape[2], gen_img.shape[0])
    #     interpolatedImage = cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR)
    #     interpolatedImage = interpolatedImage.astype(np.uint8)
    #     resultImage = interpolatedImage if resultImage is None else np.hstack(
    #         [resultImage, interpolatedImage])

    # cv2.imwrite("latent/" + "latent_1.png", resultImage)

    '''
    noise = np.random.normal(size=(1,z_dim)).astype('float32')
    gen_imgs = model.predict(noise)
    gen_imgs = ((gen_imgs - gen_imgs.min()) * 255 / (gen_imgs.max() - gen_imgs.min())).astype(np.uint8)
    gen_imgs = gen_imgs.reshape(1, -1, 64, 64, 3).swapaxes(1,
                                                        2).reshape(1*64, -1, 3)
    plt.imshow(Image.fromarray(gen_imgs))
    plt.imsave('result_img/result.jpg',gen_imgs)
    plt.close()
    '''


main()
