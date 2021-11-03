import cv2
import numpy as np
from itertools import product 
import random
import math
import os

target_sizes=[(256,256),(150,150),(140,140),(128,128),(64,64),(50,50),(32,32)]
for i in target_sizes:
    path = os.path.dirname(os.path.abspath(__file__))
    secret = os.path.join(path, 'data_test', 'cover','4.1.06.tiff')
    target_size = i
    target_path = os.path.join(path, 'data_test', 'payload-img','grey_'+str(target_size[0])+'x'+str(target_size[1])+'_4.1.06.tiff')
    imgGray = cv2.imread(secret,0)
    print(target_size)
    resized = cv2.resize(imgGray, target_size)
    cv2.imwrite(target_path, resized)