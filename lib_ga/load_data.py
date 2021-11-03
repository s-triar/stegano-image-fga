import cv2
import numpy as np
from itertools import product 
import random
import math

def load_image(path):
    imgGray = cv2.imread(path,0)
    # print(imgGray.shape)
    # cv2.imshow('payload', imgGray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return imgGray

def save_img(img,path):
    cv2.imwrite(path, img)

def findFactor(sec):
    t=sec[0]
    if(len(sec)==2):
        t = sec[0]*sec[1]*8
    pairs=[0,0]
    res = 9999999999
    for i in range(1,t+1):
        if t % i == 0:
            x=i
            y=int(t/i)
            res_temp = abs(x-y)
            if(res_temp<res):
                res=res_temp
                pairs[0]=x
                pairs[1]=y
    return pairs
def img_to_bin(img):
    a = np.empty(shape=img.shape, dtype=int)
    for ix in range(0, img.shape[0]):
        for iy in range(0, img.shape[1]):
            a[ix,iy] = int(bin(img[ix,iy])[-1])
    return a

def imgs_diff(img1,img2):
    a = np.empty(shape=img1.shape, dtype=np.uint8)
    for ix in range(0, img1.shape[0]):
        for iy in range(0, img1.shape[1]):
            a[ix,iy] = 255 if img1[ix,iy] == img2[ix,iy] else 0
    return a

def img_as_bin_img(img):
    a = np.empty(shape=img.shape, dtype=np.uint8)
    for ix in range(0, img.shape[0]):
        for iy in range(0, img.shape[1]):
            a[ix,iy] = 255 if int(bin(img[ix,iy])[-1]) == 1 else 0
    return a

def binary_to_image(secret, shape):
    a = np.empty(shape=shape, dtype=np.uint8)
    idx =0
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            t = ''
            for it in secret[idx:(idx+8)]:
                t=t+str(it)
            a[i,j]=int(t,2)
            idx = idx+8
    print(a)
    print(a.shape)
    # cv2.imshow('payload', a)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return a
        
def binary_to_text(secret):
    idx =0
    chars = ''
    for i in range(0, len(secret),8):
        t = ''
        for it in secret[i:(i+8)]:
            t=t+str(it)
        # print(i,secret[i:(i+8)])
        # print(int(t,2))
        chars = chars + chr(int(t,2))
    print(chars)

def secret_to_bin(sec):
    a = np.empty(shape=sec.shape[0]*sec.shape[1]*8, dtype=int)
    indexA = 0
    sec_flat = sec.flatten()
    for pix in sec_flat:
        temp = [0,0,0,0,0,0,0,0]
        p_bin = bin(pix)[2:]
        for i in temp[:len(temp)-len(p_bin)]:
            a[indexA]=i
            indexA=indexA+1
        for i in p_bin:
            a[indexA]=int(i)
            indexA=indexA+1
    return a

def one_d_to_two_d(data):
    newSise = findFactor(data.shape)
    return np.reshape(data,(newSise[0],newSise[1]))


def load_text(path):
    f = open(path, "r")
    c = f.read()
    a_byte_array = bytearray(c, "utf8")
    biner = []
    for i in a_byte_array:
        b = bin(i)
        for i in range(0,8-len(b[2:])):
            biner.append(0)
        for ii in b[2:]:
            biner.append(int(ii))
    return np.array(biner)

def load_binary_in_text(path):
    f = open(path, "r")
    c = f.read()
    biner = []
    for i in c:
        try:
            biner.append(int(i))
        except:
            pass
    return np.array(biner)
