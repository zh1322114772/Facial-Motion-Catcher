# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:19:49 2020

@author: otz55
"""
import os


os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras import layers
from keras.models import Model
from keras import optimizers
from keras.losses import mse


'''
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.losses import mse
'''

import numpy as np
import cv2
from timeit import default_timer as timer

def draw_line(color, line, image, h, w):
    for i in range(line.shape[0] - 1):
        cv2.line(image, (int(line[i, 0] * w), int(line[i, 1] * h)), (int(line[i + 1, 0] * w), int(line[i + 1, 1] * h)), color, 2)

def highlight_face(res_1, chin, nose, mouth_in, mouth_out, r_eye, l_eye, r_eye_b, l_eye_b, image):

    
    chin[:] = np.mean(res_1[0 : 40].reshape(-1, 2, 2), axis = 1)
    nose[:] = np.mean(res_1[41 : 57].reshape(-1, 2, 2), axis = 1)
    mouth_in[0: -1] = np.mean(res_1[58 : 86].reshape(-1, 4, 2), axis = 1)
    mouth_out[0: -1] = np.mean(res_1[86 : 114].reshape(-1, 4, 2), axis = 1)
    r_eye[0: -1] = np.mean(res_1[114 : 134].reshape(-1, 4, 2), axis = 1)
    l_eye[0: -1] = np.mean(res_1[134 : 154].reshape(-1, 4, 2), axis = 1)
    r_eye_b[0: -1] = np.mean(res_1[154 : 174].reshape(-1, 4, 2), axis = 1)
    l_eye_b[0: -1] = np.mean(res_1[174 : 194].reshape(-1, 4, 2), axis = 1)

    #make circle
    mouth_in[-1] = mouth_in[0] 
    mouth_out[-1] = mouth_out[0] 
    r_eye[-1] = r_eye[0] 
    l_eye[-1] = l_eye[0] 
    r_eye_b[-1] = r_eye_b[0] 
    l_eye_b[-1] = l_eye_b[0] 


img_w = 224
img_h = 224
data_dir = os.getcwd() + '\\data'

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
    model.compile(optimizer = optimizers.Adam(lr = 0.0000003), loss = custom_loss)
    return model

model = get_model()
#try to load weights
model = get_model()
try:
    model.load_weights(data_dir + '\\model.h5')
except:
    print('Unable to load weights')
    exit()
    
#cv2
capture = cv2.VideoCapture(0)
axis = np.zeros((194,2))

#axis
chin_axis = np.zeros((20, 2))
nose_axis = np.zeros((8, 2))
mouth_inner_axis = np.zeros((8, 2))
mouth_outter_axis = np.zeros((8, 2))
r_eye_axis = np.zeros((6, 2))
l_eye_axis = np.zeros((6, 2))
r_eyebrow_axis = np.zeros((6, 2))
l_eyebrow_axis = np.zeros((6, 2))

white_board = np.zeros((800, 800), np.uint8)

while True:
    isTrue, frame = capture.read()
    frame1 = cv2.resize(frame, (224, 224))/255
    start = timer()
    res_1 = np.squeeze(model.predict(np.expand_dims(frame1, 0)), 0)
    
    #correctness
    if(res_1[0]>0.99):
        res_2d = res_1[1:].reshape(-1, 2)
        highlight_face(res_2d, chin_axis, nose_axis, mouth_inner_axis, mouth_outter_axis, r_eye_axis, l_eye_axis, r_eyebrow_axis, l_eyebrow_axis, frame)
        white_board[:] = np.zeros((800, 800), np.uint8)
        draw_line((255, 255, 255), chin_axis, white_board, 800, 800)
        draw_line((255, 255, 255), nose_axis, white_board, 800, 800)
        draw_line((255, 255, 255), mouth_inner_axis, white_board, 800, 800)
        draw_line((255, 255, 255), mouth_outter_axis, white_board, 800, 800)
        draw_line((255, 255, 255), r_eye_axis, white_board, 800, 800)
        draw_line((255, 255, 255), l_eye_axis, white_board, 800, 800)
        draw_line((255, 255, 255), r_eyebrow_axis, white_board, 800, 800)
        draw_line((255, 255, 255), l_eyebrow_axis, white_board, 800, 800)
        height, width, channels = frame.shape
        
        for i in range(0, 194):
            frame = cv2.circle(frame, (int(res_2d[i, 0] * width), int(res_2d[i, 1] * height)), 2, (255, 0, 0), 2)
    
    tick = (timer() - start) * 1000
    
    frame = cv2.putText(frame, 'FPS: ' + str(round(tick, 2)) + '  is a face: ' + str(round(res_1[0] * 100, 2)) + '%', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0) , 2)
    
    
    cv2.imshow('WINDOW', frame)
    cv2.imshow('Face', white_board)
    key = cv2.waitKey(1)
    if key == 27: # exit on ESC
        break



capture.release()
cv2.destroyAllWindows()
    