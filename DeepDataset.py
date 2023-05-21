"""Input Datasets"""
import os
import numpy as np
from PIL import Image

def read_img(img_path, label, img_size):
    
    img_files = os.listdir(img_path)
    data_list = []
    
    for imgf in img_files:
        imgp = os.path.join(img_path, imgf)
        img = Image.open(imgp)
        imgr = img.resize((img_size,img_size))
        imga = np.array(imgr).reshape((1,img_size,img_size))
        data_list.append(imga)
    data_arrays = np.array(data_list)
    labels = np.repeat(label, len(img_files))
    return data_arrays, labels
