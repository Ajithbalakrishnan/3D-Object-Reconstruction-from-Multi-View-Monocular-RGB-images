import tensorflow as tf
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




if __name__ == '__main__':
    print ('enterd')    
    obj_rgbs_folder ='./Data_sample/amazon_real_rgbs/edited/'
    rgbs = []
    rgbs_views = sorted(os.listdir(obj_rgbs_folder))
    for v in rgbs_views:
        if not v.endswith('png'): continue
        print('start')
        im = Image.open(obj_rgbs_folder+v)
        print (im)



#resize_img = tf.reshape(image.img, [-1, 127, 127, 3])
