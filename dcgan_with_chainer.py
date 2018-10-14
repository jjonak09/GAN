import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, cuda, serializers, Variable, initializers, Chain
import os
import cv2 as cv
import pylab
from model import discriminator, generator
import argparse
import math

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha, beta):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
    return optimizer

def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))

parser = argparse.ArgumentParser(description = "DCGAN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 64, type = int, help = "batch size")
parser.add_argument("--interval", default = 10, type = int, help = "the interval of snapshot")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
wid = int(math.sqrt(batchsize))

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

x_train = np.load('./Vtuber.npy').astype(np.float32)

print(x_train.shape)
Ntrain, channels, width, height = x_train.shape

gen_model = generator()
dis_model = discriminator()

gen_model.to_gpu()
dis_model.to_gpu()

gen_opt = set_optimizer(gen_model, 1e-4, 0.5)
dis_opt = set_optimizer(dis_model, 1e-4, 0.5)

zvis = xp.random.uniform(-1,1,(batchsize,100),dtype=np.float32)

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        x_dis = np.zeros((batchsize,channels,width,height), dtype=np.float32)
        for j in range(batchsize):
            rnd = np.random.randint(Ntrain)
            x_dis[j,:,:,:] = x_train[rnd]

        z = Variable(xp.random.uniform(-1,1,(batchsize,100), dtype = xp.float32))
        x = gen_model(z)
        y = dis_model(x)

        gen_loss = F.softmax_cross_entropy(y, Variable(xp.zeros(batchsize,dtype=xp.int32)))
        dis_loss = F.softmax_cross_entropy(y, Variable(xp.ones(batchsize,dtype=xp.int32)))

        x_dis = Variable(cuda.to_gpu(x_dis))
        y_dis = dis_model(x_dis)
        dis_loss += F.softmax_cross_entropy(y_dis, Variable(xp.zeros(batchsize,dtype=xp.int32)))
        
        gen_model.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        
        dis_model.cleargrads()
        dis_loss.backward()
        dis_opt.update()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if epoch%interval==0 and batch ==0:
            serializers.save_npz('discriminator.model',dis_model)
            serializers.save_npz('generator.model',gen_model)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            z = zvis
            z = Variable(z)
            with chainer.using_config('train',False):
                x = gen_model(z)
            x = x.data.get()
            for i_ in range(batchsize):
                tmp = ((np.vectorize(clip_img)(x[i_,:,:,:])+1)/2).transpose(1,2,0)
                pylab.subplot(wid,wid,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(image_out_dir, epoch))

    print('epoch : {} dis_loss : {} gen_loss : {}'.format(epoch,sum_dis_loss/Ntrain,sum_gen_loss/Ntrain))