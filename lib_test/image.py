import numpy as np
from itertools import product 
import random
import math

def load_image(x,y):
    sz=[x,y]
    a = np.empty(shape=(sz[0],sz[1]), dtype=int)
    for ix in range(0, sz[0]):
        for iy in range(0, sz[1]):
            a[ix,iy] = random.randint(0,255)
    return a

def load_secret(x,y):
    sz=[x,y] # max 511 x 512
    a = np.empty(shape=(sz[0],sz[1]), dtype=int)
    for ix in range(0, sz[0]):
        for iy in range(0, sz[1]):
            a[ix,iy] = random.randint(0,255)
    return a


def img2bin(img):
    a = np.empty(shape=img.shape, dtype=int)
    for ix in range(0, img.shape[0]):
        for iy in range(0, img.shape[1]):
            a[ix,iy] = bin(img[ix,iy])[-1]
    return a

def findFactor(sec):
    t = sec.shape[0]*sec.shape[1]*8
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

    

def secret2bin(sec):
    newSise = findFactor(sec)
    print(newSise)
    a = np.empty(shape=sec.shape[0]*sec.shape[1]*8, dtype=int)
    indexA = 0
    sec_flat = sec.flatten()
    print(len(sec_flat), len(a))
    for pix in sec_flat:
        temp = [0,0,0,0,0,0,0,0]
        p_bin = bin(pix)[2:]
        for i in temp[:len(temp)-len(p_bin)]:
            a[indexA]=i
            indexA=indexA+1
        for i in p_bin:
            a[indexA]=i
            indexA=indexA+1
    return np.reshape(a,(newSise[0],newSise[1]))