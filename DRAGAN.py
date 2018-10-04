from __future__ import print_function, division
import keras.backend as K
K.set_image_data_format('channels_first')
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Cropping2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import sys,os
import numpy as np
import matplotlib.pyplot as plt


class DRAGAN():
    def __init__(self):

         # -----------------
         # Parameters
         # -----------------

        self.image_size = 64
        self.channels = 3
        self.img_shape = (self.channels,self.image_size, self.image_size)
        self.z_dim = 100
        self.Diters = 5
        self.λ = 10
        #learning rate
        self.lrD = 1e-4
        self.lrG = 1e-4
    
        self.conv_init = RandomNormal(0, 0.02)
        self.gamma_init = RandomNormal(1., 0.02)

        self.dis = self.build_discriminator()
        self.gen = self.build_generator()

        # -----------------
        # compute Wasserstein loss and gradient penalty
        # -----------------

        self.dis_real_input = Input(shape=(self.channels,self.image_size, self.image_size))
        self.noise = Input(shape=(self.z_dim,))
        self.dis_fake_input = self.gen(self.noise)

        self.ϵ_input = K.placeholder(shape=(None,self.channels,self.image_size, self.image_size))
        self.dis_mixed_input = Input(shape=(self.channels, self.image_size, self.image_size), tensor=self.dis_real_input + self.ϵ_input)
        
        self.loss_real = K.mean(self.dis(self.dis_real_input))
        self.loss_fake = K.mean(self.dis(self.dis_fake_input))

        self.grad_mixed = K.gradients(self.dis(self.dis_mixed_input),[self.dis_mixed_input])[0]
        self.norm_grad_mixed = K.sqrt(K.sum(K.square(self.grad_mixed),axis=[1,2,3]))
        self.grad_penalty = K.mean(K.square(self.norm_grad_mixed -1))

        self.loss = self.loss_fake - self.loss_real + self.λ * self.grad_penalty

        # -----------------
        # loss for dis
        # -----------------
        self.dis.trainable_weights
        self.training_updates = Adam(lr=self.lrD).get_updates(
            self.dis.trainable_weights, [], self.loss)

        self.dis_train = K.function(
            [self.dis_real_input, self.noise, self.ϵ_input],
            [self.loss_real,self.loss_fake],
            self.training_updates)

        # -----------------
        # loss for gen
        # -----------------
        self.loss = -self.loss_fake
        self.training_updates = Adam(lr=self.lrG).get_updates(
            self.gen.trainable_weights,[],self.loss)

        self.gen_train = K.function(
            [self.noise],
            [self.loss],
            self.training_updates)




    def build_discriminator(self):

            model = Sequential()

            model.add(Conv2D(64, kernel_size=4, strides=2,
                            input_shape=self.img_shape, padding="same",
                            use_bias=False,kernel_initializer=self.conv_init))  
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

            img = Input(shape=self.img_shape) #inputs
            validity = model(img)  #outputs

            return Model(img, validity)


    def build_generator(self):

            model = Sequential()

            model.add(Dense(4 * 4 * 512, activation="relu",
                            input_dim=self.z_dim))
            model.add(Reshape((512,4, 4)))
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

            noise = Input(shape=(self.z_dim,)) #inputs
            img = model(noise)  #outputs

            return Model(noise, img)


    def train(self, epochs, batch_size=32, sava_interval=50):
        print("a")


if __name__ == '__main__':
    dragan = DRAGAN()
    dragan.train(epochs=4000, batch_size=32, sava_interval=50)
