from __future__ import print_function, division
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation,Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid,relu,softplus
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import argparse

conv_init = RandomNormal(0, 0.02)



def SubpixelConv2D(input_shape, scale=2):

    # Copyright (c) 2017 Jerry Liu
    # Released under the MIT license
    # https://github.com/twairball/keras-subpixel-conv/blob/master/LICENSE

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)


# -------------------------------------------
# Use pixel shuffler instead of Conv2DTranspose
# -------------------------------------------

def generator(z_dim):

    input = Input(shape=(z_dim,))
    h = input
    h = Dense(512 * 4 * 4,kernel_initializer=conv_init)(h) #4x4x512
    h = Activation("relu")(h)
    h = Reshape((4, 4, 512))(h)
    h = Conv2D(1024, 1, strides=1, padding="same",kernel_initializer=conv_init)(h)
    h = SubpixelConv2D(h)(h)                             #8x8x256
    h = Activation("relu")(h)
    h = Conv2D(512, 1, strides=1, padding="same",kernel_initializer=conv_init)(h)
    h = SubpixelConv2D(h)(h)                              #16x16x128
    h = Activation("relu")(h)
    h = Conv2D(256, 1, strides=1, padding="same",kernel_initializer=conv_init)(h)
    h = SubpixelConv2D(h)(h)                              #32x32x64
    h = Activation("relu")(h)
    h = Conv2D(12, 1, strides=1, padding="same",kernel_initializer=conv_init)(h)
    h = SubpixelConv2D(h)(h)                               #64x64x3
    h = Activation("tanh")(h)

    model = Model(inputs=input, outputs=h)
    return model



# def generator(z_dim):

#     model = Sequential()

#     model.add(Dense(512 * 4 * 4, activation="relu",
#                     input_dim=z_dim
#                     ,  kernel_initializer=conv_init
#                     ))
#     model.add(Activation("relu"))
#     model.add(Reshape((4, 4, 512)))
#     model.add(Conv2DTranspose(256, 4, strides=2, padding="same"
#     ,kernel_initializer=conv_init
#     ))  # 8x8x256
#     model.add(Activation("relu"))
#     model.add(Conv2DTranspose(128, 4, strides=2, padding="same"
#     ,kernel_initializer=conv_init
#     ))  # 16x16x128
#     model.add(Activation("relu"))
#     model.add(Conv2DTranspose(64, 4,  strides=2, padding="same"
#     ,kernel_initializer=conv_init
#     ))  # 32x32x64
#     model.add(Activation("relu"))
#     model.add(Conv2DTranspose(3, 4,  strides=2,
#                               padding="same"
#                               ,kernel_initializer=conv_init
#                               ))  # 64x64x3
#     model.add(Activation("tanh"))

#     noise = Input(shape=(z_dim,))
#     img = model(noise)
#     return Model(noise, img)


def discriminator(img_shape):

    model = Sequential()

    model.add(Conv2D(64, 4, strides=2,
                     input_shape=img_shape, padding="same"
                     ,kernel_initializer=conv_init
                     ))  # 32x32x64
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, 4, strides=2, padding="same"
    ,kernel_initializer=conv_init
    ))  # 16x16x128
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, 4, strides=2, padding="same"
    ,kernel_initializer=conv_init
    ))  # 8x8x256
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(512, 4, strides=2, padding="same"
    ,kernel_initializer=conv_init
    ))  # 4x4x512
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


# -----------------
# parameters
# -----------------

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--epoch", default=10000, type=int,
                    help="the number of epochs")
parser.add_argument("--save_interval", default=10, type=int,
                    help="the interval of snapshot")
parser.add_argument("--model_interval", default=100, type=int,
                    help="the interval of savemodel")
parser.add_argument("--batchsize", default=128, type=int, help="batch size")
parser.add_argument("--lam", default=10.0, type=float,
                    help="the weight of regularizer")

args = parser.parse_args()
epochs = args.epoch
save_interval = args.save_interval
model_interval = args.model_interval
batch_size = args.batchsize
_lambda = args.lam

z_dim = 100
img_shape = (64, 64, 3)
image_size = 64
channels = 3
lr_D = 1e-4
lr_G = 1e-4

gen = generator(z_dim)
dis = discriminator(img_shape)

gen.summary()
# dis.summary()

# -----------------
# load dataset
# -----------------

X_train = np.load('./64x64.npy')
X_train = np.float32(X_train)
X_train = X_train/127.5 - 1
X_train = np.expand_dims(X_train, axis=3)
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[4])


# -------------------------
# compute grandient penalty
# -------------------------

dis_real = Input(shape=(image_size, image_size, channels))
noisev = Input(shape=(z_dim,))
dis_fake = gen(noisev)

delta_input = K.placeholder(shape=(None, image_size, image_size, channels))
alpha = K.random_uniform(
    shape=[batch_size, 1, 1, 1],
    minval=0.,
    maxval=1.
)

dis_mixed = Input(shape=(image_size, image_size, channels),
                  tensor=dis_real + delta_input)

loss_real = K.sum(softplus(-dis(dis_real))) / batch_size
loss_fake = K.sum(softplus(dis(dis_fake))) / batch_size

dis_mixed_real = alpha * dis_real + ((1 - alpha) * dis_mixed)

grad_mixed = K.gradients(dis(dis_mixed_real), [dis_mixed_real])[0]
norm = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
grad_penalty = K.mean(K.square(norm - 1))
loss_dis = loss_fake + loss_real + _lambda * grad_penalty

# ---------------------
# loss for discriminator
# ---------------------

training_updates = Adam(lr=lr_D, beta_1=0.5).get_updates(
    dis.trainable_weights, [], loss_dis)
dis_train = K.function([dis_real, noisev, delta_input],
                       [loss_real, loss_fake],
                       training_updates)

# -----------------
# loss for generator
# -----------------

loss_gen = K.sum(softplus(-dis(dis_fake))) / batch_size

training_updates = Adam(lr=lr_G, beta_1=0.5).get_updates(
    gen.trainable_weights, [], loss_gen)
gen_train = K.function([noisev],
                       [loss_gen],
                       training_updates)

# -----------------
# Training session
# -----------------

fixed_noise = np.random.normal(size=(25, z_dim))
batch = X_train.shape[0] // batch_size

for epoch in range(epochs):
    print("Epoch is", epoch)
    for index in range(batch):

        idx = np.random.randint(0, X_train.shape[0], batch_size)
        image_batch = X_train[idx]
        noise = np.random.normal(size=(batch_size, z_dim))
        delta = 0.5 * image_batch.std() * np.random.random(size=image_batch.shape)
        delta *= np.random.uniform(size=(batch_size, 1, 1, 1))
        errD_real, errD_fake = dis_train([image_batch, noise, delta])
        errD = errD_real - errD_fake

        errG, = gen_train([noise])
        print('%d/%d  Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (index, batch, errD, errG, errD_real, errD_fake))

        if epoch % save_interval == 0 and index == 0:
            gen_imgs = gen.predict(fixed_noise)
            r, c = 5, 5
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("result/%d.png" % epoch)
            plt.close()
            if epoch % model_interval == 0 and index == 0:
                gen.save("DRAGAN_model/model-{}-epoch.h5".format(epoch))
