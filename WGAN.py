from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal

import keras.backend as K

import matplotlib.pyplot as plt

import sys

import numpy as np

class WGAN():
    def __init__(self):
        self.image_size = 64
        self.channels = 3
        self.img_shape = (self.image_size, self.image_size, self.channels)
        self.z_dim = 100
        self.conv_init = RandomNormal(0, 0.02)
        self.n_critic = 5
        self.clip = 0.01
        optimizer = RMSprop(lr = 0.00005)


        # ---------------------
        #  build Discriminator
        # ---------------------

        self.dis = self.build_discriminator()
        self.dis.compile(loss=self.wasserstein_loss,
        optimizer=optimizer,
        metrics=['accuracy'])


        # ---------------------
        #  build Generator
        # ---------------------

        self.gen = self.build_generator()
        
        
        z = Input(shape=(self.z_dim,))
        img = self.gen(z)
        self.dis.trainable = False
        valid = self.dis(img)

        # ---------------------
        #  The combined Model
        # ---------------------

        self.combined = Model(z,valid)   #zを入力にした時にvalidを計算する際に必要な層を含む
        self.combined.compile(loss=self.wasserstein_loss,
        optimizer=optimizer,
        metrics=['accuracy'])



    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def build_discriminator(self):

            model = Sequential()

            model.add(Conv2D(64, kernel_size=4, strides=2,
                             input_shape=self.img_shape, padding="same",
                             use_bias=False, kernel_initializer=self.conv_init))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(128, kernel_size=4, strides=2, padding="same",
                             use_bias=False, kernel_initializer=self.conv_init))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(256, kernel_size=4, strides=2, padding="same",
                             use_bias=False, kernel_initializer=self.conv_init))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(512, kernel_size=4, strides=2, padding="same",
                             use_bias=False, kernel_initializer=self.conv_init))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(1, kernel_size=4, strides=1,
                             use_bias=False, kernel_initializer=self.conv_init))
            model.add(Flatten())

            model.summary()

            img = Input(shape=self.img_shape)  # inputs
            validity = model(img)  # outputs

            return Model(img, validity)

    def build_generator(self):

            model = Sequential()

            model.add(Dense(4 * 4 * 512, activation="relu",
                            input_dim=self.z_dim))
            model.add(Reshape((4, 4,512)))
            model.add(Conv2DTranspose(256, 4, strides=2,
                                      padding="same",
                                      use_bias=False, kernel_initializer=self.conv_init))  # 8x8x256
            model.add(Activation("relu"))
            model.add(Conv2DTranspose(128, 4, strides=2,
                                      padding="same",
                                      use_bias=False, kernel_initializer=self.conv_init))  # 16x16x128
            model.add(Activation("relu"))
            model.add(Conv2DTranspose(64, 4,  strides=2,
                                      padding="same",
                                      use_bias=False, kernel_initializer=self.conv_init))  # 32x32x64
            model.add(Activation("relu"))
            model.add(Conv2DTranspose(self.channels, 4,  strides=2,
                                      padding="same",
                                      use_bias=False, kernel_initializer=self.conv_init))  # 64x64x3
            model.add(Activation("tanh"))

            model.summary()  # modelの要約を出力

            noise = Input(shape=(self.z_dim,))  # inputs
            img = model(noise)  # outputs

            return Model(noise, img)

    def train(self, epochs, batch_size=128, save_interval=50,model_interval=100):
        X_train, X_test=np.load("Vtuber.npy")
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[4])

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1)) #(128,1)の全て-1の配列
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            print("Epoch is", epoch)
            #WGANではdiscriminatorをncritic回更新してからgeneratorを更新する
            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                idx = np.random.randint(0, X_train.shape[0], batch_size)
                image_batch = X_train[idx]
                noise = np.random.normal(0,1, (batch_size,self.z_dim))
                generate_images   = self.gen.predict(noise)

                d_loss_real = self.dis.train_on_batch(image_batch,valid)
                d_loss_fake = self.dis.train_on_batch(generate_images,fake)
                d_loss = 0.5 * np.add(d_loss_fake,d_loss_real)

                # ---------------------
                #  リプシッツ連続を保証するためのweight clipping
                # ---------------------

                for l in self.dis.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip,
                                       self.clip) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            print("%d [D loss: %f] [G loss: %f]" %
                  (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            if epoch % sava_interval == 0:
                    self.save_imgs(epoch)
                    if epoch % model_interval == 0:
                        self.generator.save(
                            "DCGAN_model/dcgan-{}-epoch.h5".format(epoch))
       
    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r*c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("result_img/Vtuber_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=20000, batch_size=16,
                save_interval=100, model_interval=5000)



