from PIL import Image
import os,glob
import numpy as np 
from sklearn import model_selection
import random

image_size = 64

data_dir = "./dataset_96x96"

X_train = []

files = glob.glob(data_dir + "/*.jpg")
random.shuffle(files)
for i,file in enumerate(files):
    if i >= 16188: break #datasetの画像数によって変える
    image = Image.open(file)
    image = image.convert("RGB")
    image = image.resize((image_size,image_size))
    data = np.asarray(image).transpose(2,0,1).astype(np.float32)/127.5 -1.
    X_train.append(data)

X_train = np.array(X_train)
np.save("./Vtuber.npy",X_train)