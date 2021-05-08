# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 22:30:58 2020

@author: otz55
"""

import os
import numpy as np
import json
import PIL

#data directory
data_dir = os.getcwd() + '\\data'
file_names = [i for i in os.listdir(data_dir + '\\annotation')]
index = {}
y_data = np.zeros((len(file_names), 388), dtype = np.float32)


#annotation process
for i, n in enumerate(file_names):
    print('processing...', i)
    
    #read annotation text
    txt = [j.replace('\n', '') for j in open(data_dir + '\\annotation\\' + n).readlines()]
    
    #get corresponding image size
    img = PIL.Image.open(data_dir + '\\img_pos\\' + txt[0] + '.jpg')
    img_w, img_h = img.size
    
    #re-scale annotation to 0 - 1
    for l in range(194):
        ax = txt[l + 1].split()
        y_data[i, l * 2] = float(ax[0])/img_w
        y_data[i, (l * 2) + 1] = float(ax[2])/img_h
        
    #set index mapping
    index[i] = txt[0] + '.jpg'
    
#save data
np.save(data_dir + '\\y_data', y_data)
writer = open(data_dir + '\\y_index.json', mode = 'w')
writer.write(json.dumps(index))
writer.close()
print('down...')