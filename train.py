# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 22:51:53 2020

@author: otz55
"""
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import PIL
from PIL import ImageEnhance
import json
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from keras.losses import mse
import keras
from keras import layers
from keras.models import Model
from keras import optimizers
from keras.callbacks import LambdaCallback
data_dir = os.getcwd() + '\\data'
img_w = 224
img_h = 224

#read data
y_data = np.load(data_dir + '\\y_data.npy')
y_index = json.loads(open(data_dir + '\\y_index.json').readline())
y_neg_list = [i for i in os.listdir(data_dir + '\\img_neg')]

def img_show(img_arr, axis):
    fig, ax = plt.subplots(1)
    ax.imshow(img_arr)
    axis_2d = axis.reshape(-1, 2)
    
    for i in range(194):
        c = Circle((axis_2d[i, 0] * img_arr.shape[0], axis_2d[i, 1] * img_arr.shape[1]), 0.5, color = 'red')
        ax.add_patch(c)


def rotate_axis(d, angle):
    rad = ((-angle) * math.pi)/180
    d2d = d.reshape(-1, 2)
    
    xs = d2d[:, 0] - 0.5
    ys = d2d[:, 1] - 0.5
    hyp = np.power(np.power(xs, 2) + np.power(ys, 2), 0.5)
    rads = np.arctan2(ys, xs)
    rads += rad
    
    xs[:] = (np.cos(rads) * hyp) + 0.5
    ys[:] = (np.sin(rads) * hyp) + 0.5
    
    d1 = np.zeros(d2d.shape)
    d1[:, 0] = xs
    d1[:, 1] = ys
    return np.squeeze(d1.reshape(1, -1), 0)
    
#get annotation bounding box
def get_bounding_box(axis):
    np_x = axis.reshape(194, 2)[:, 0]
    np_y = axis.reshape(194, 2)[:, 1]
    
    min_x = np_x[np.argmin(np_x)]
    max_x = np_x[np.argmax(np_x)]
    min_y = np_y[np.argmin(np_y)]
    max_y = np_y[np.argmax(np_y)]
    
    return (max_x, min_x, max_y, min_y)
    

#input data generator
def data_generator(batch_size, w, h, rotate = False, crop = True, enhance_range = 0.3):
    
    #return data
    Y = np.zeros((batch_size * 2, 389))
    X = np.zeros((batch_size * 2, w, h, 3))
    feature = np.zeros((388,))
    counter = 0
    
    while True:
        #generate batch
        for i in range(batch_size):           
            # randomly read correct/incorrect samples
            pos_img = PIL.Image.open(data_dir + '\\img_pos\\' + y_index[str(counter)]).convert('RGB')
            neg_img = PIL.Image.open(data_dir + '\\img_neg\\' + y_neg_list[counter]).convert('RGB')
            feature[:] = y_data[counter]      
            #crop
            if crop == True:
                width, height = pos_img.size
                max_x, min_x, max_y, min_y = get_bounding_box(feature)
                
                if min_x >= 0 and min_y >= 0 and max_x <= 1 and max_y <= 1: 
                    left = np.random.uniform(0, min_x)
                    right = np.random.uniform(max_x, 1)
                    top = np.random.uniform(0, min_y)
                    bottom = np.random.uniform(max_y, 1)
                    
                    feature = feature.reshape(-1, 2)
                    feature[:, 0] = (feature[:, 0] - left) * (1/(right - left))
                    feature[:, 1] = (feature[:, 1] - top) * (1/(bottom - top))
                    feature = np.squeeze(feature.reshape(1, -1), 0)
                    pos_img = pos_img.crop((left * width, top * height, right * width, bottom * height))
                    
                    width, height = neg_img.size
                    neg_img = neg_img.crop((np.random.uniform(0, 0.25) * width, 
                                            np.random.uniform(0, 0.25) * height, 
                                            np.random.uniform(0.75, 1) * width, 
                                            np.random.uniform(0.75, 1) * height))
                    
                    
            #resize
            pos_img = pos_img.resize((w, h))
            neg_img = neg_img.resize((w, h))
            
            #rotate
            if rotate == True:
                angle = (np.random.randint(0,3) * 90) - 90
                feature = rotate_axis(feature, angle)
                pos_img = pos_img.rotate(angle)
                angle = (np.random.randint(0,3) * 90) - 90
                neg_img = neg_img.rotate(angle) 
            
            #random brightness, contrast and color
            pos_img = ImageEnhance.Brightness(pos_img).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            pos_img = ImageEnhance.Color(pos_img).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            pos_img = ImageEnhance.Contrast(pos_img).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            
            neg_img = ImageEnhance.Brightness(neg_img).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            neg_img = ImageEnhance.Color(neg_img).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            neg_img = ImageEnhance.Contrast(neg_img).enhance(1 + np.random.uniform(0, enhance_range) - (enhance_range/2))
            
            #generate train data
            X[i] = np.array(pos_img)/255
            X[batch_size + i] = np.array(neg_img)/255
            
            #it's not a background therefore 0 position is 1
            Y[i, 0] = 1 
            Y[i, 1: ] = feature
            
            counter +=1
            counter %= len(y_index)
            
        #shuffle data
        index = np.arange(0, batch_size*2)  
        np.random.shuffle(index)
        yield (X[index], Y[index])

def custom_loss(y_actual, y_pred):
   for i in range(y_actual.shape[0]):
       if y_actual[i, 0] == 0:
           y_pred[i, 1 :] = np.zeros((388,))
  
   return mse(y_actual, y_pred)           

#generate model
def get_model():
    
    input_layer = layers.Input(shape = (img_w, img_h, 3))
    #use VGG16 model
    conv_model = keras.applications.VGG16(include_top = False, weights = 'imagenet')
    for i in conv_model.layers:
        i.trainable = False
    
    conv_model.layers[-1].trainable = True
    conv_model.layers[-2].trainable = True
    conv_model.layers[-3].trainable = True
    conv_model.layers[-4].trainable = True
    
    for i in conv_model.layers:
        print(i ,i.trainable)
    
    x = layers.Conv2D(2048, (7, 7))(conv_model(input_layer))
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(389, (1, 1))(x)
    x = layers.Flatten()(x)
    x = layers.Activation('sigmoid')(x)   
    
    model = Model(input_layer, x)
    model.summary()
    model.compile(optimizer = optimizers.Adam(lr = 0.0000002), loss = custom_loss)
    return model


def auto_save(epoch, logs):
        model.save_weights(data_dir + '\\model.h5')

#try to load weights
model = get_model()
try:
    model.load_weights(data_dir + '\\model.h5')
except:
    None

#train model
model.fit_generator(data_generator(10, img_w, img_h),
                    steps_per_epoch = 30,
                    epochs = 2000, 
                    callbacks=[LambdaCallback(on_epoch_end = auto_save)])
auto_save(10, None)


'''
a, b = next(data_generator((0, 2330), 1, img_w, img_h))
img_show(a[0], b[0,1 :])
'''