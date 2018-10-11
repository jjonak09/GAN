
from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D,Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import sys
import numpy as np
import matplotlib.pyplot as plt


class DCGAN():
    def __init__(self):
        self.image_size = 64
        self.channels = 3
        self.img_shape = (self.image_size, self.image_size, self.channels)
        self.z_dim = 100
        self.conv_init = RandomNormal(0, 0.02) #ネットワーク重み平均0,標準偏差0.02

        optimizer_dis = Adam(0.0001, 0.5)
        optimizer_gen = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer_dis,
                                   metrics=['accuracy'])


        self.generator = self.build_generator()
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        self.combined = Model(z, valid) #zを入力してvaildを得るまでのそう全てを含んでいる
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer_gen)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512 * 4 * 4, activation="relu",
                        input_dim=self.z_dim, use_bias=False, kernel_initializer=self.conv_init))
        model.add(Reshape((4, 4, 512)))
        model.add(Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False,
                                  kernel_initializer = self.conv_init)) #8x8x256
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False,
                                  kernel_initializer=self.conv_init,))  # 16x16x128
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(64, 4,  strides=2, padding="same", use_bias=False,
                                  kernel_initializer=self.conv_init))  # 32x32x64
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(self.channels, 4,  strides=2,
                                  padding="same", use_bias=False,
                                  kernel_initializer=self.conv_init))  # 64x64x3
        model.add(Activation("tanh"))

        model.summary() #modelの要約を出力

        noise = Input(shape=(self.z_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64,4, strides=2,
                         input_shape=self.img_shape, padding="same", use_bias=False,
                         kernel_initializer=self.conv_init))  # 32x32x64
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, 4, strides=2, padding="same",use_bias = False,
                         kernel_initializer=self.conv_init))  # 16x16x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, 4, strides=2, padding="same", use_bias=False,
                         kernel_initializer=self.conv_init))  # 8x8x256
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, 4, strides=2, padding="same", use_bias=False,
                         kernel_initializer=self.conv_init))  # 4x4x512
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(1, 4, strides=2, use_bias=False,
                         kernel_initializer=self.conv_init))# 4x4x512
        model.add(GlobalAveragePooling2D())
        model.add(Activation("sigmoid"))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=32, sava_interval=50,model_interval = 1000):
        X_train, X_test = np.load('./Vtuber.npy')
        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3) #?
        X_train = X_train.reshape(
            X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[4])

        valid = np.ones((batch_size, 1))  #real
        fake = np.zeros((batch_size, 1))  #fake

        for epoch in range(epochs):
            print("Epoch is", epoch)
            for index in range(int(X_train.shape[0]/batch_size)):

                idx = np.random.randint(0, X_train.shape[0], batch_size)
                image_batch = X_train[idx]

                # -----------------
                # Training Generator
                # -----------------

                noise = np.random.normal(0, 1, (batch_size, self.z_dim)) #batch_size x z_dimのノイズを発生させる
                g_loss = self.combined.train_on_batch(noise, valid)# realのラベルで扱うことによってdiscriminatorを騙す

                # ------------------
                # Training Discriminator
                # -----------------
                generated_images = self.generator.predict(noise)

                d_loss_real = self.discriminator.train_on_batch(image_batch, valid)
                d_loss_fake = self.discriminator.train_on_batch(
                    generated_images, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            

                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                    (epoch, d_loss[0], 100*d_loss[1], g_loss))

                if epoch % sava_interval == 0:
                    self.save_imgs(epoch)
                    if epoch % model_interval == 0:
                        self.generator.save(
                            "DCGAN_model/dcgan-{}-epoch.h5".format(epoch))


    def save_imgs(self, epoch):
        r, c = 6, 6
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
    dcgan = DCGAN()
    dcgan.train(epochs=20000, batch_size=16, sava_interval=100,model_interval = 500)
