import cv2
import numpy as np
from itertools import product 
import random
import math

const_path = "D:\\NyeMan\\KULIAH S2\\Semester 1\\Topik Dalam Jaringan Multimedia - A\\image stegano with fga\\data_test/"
const_path_hasil = "D:\\NyeMan\\KULIAH S2\\Semester 1\\Topik Dalam Jaringan Multimedia - A\\image stegano with fga\\data_hasil/"

def load_image(path):
    imgGray = cv2.imread(const_path+path,0)
    # print(imgGray)
    return imgGray
def load_secret(path):
    imgGray = cv2.imread(const_path+path,0)
    # print(imgGray)
    return imgGray
def load_image_hasil(path):
    imgGray = cv2.imread(const_path_hasil+path,0)
    # print(imgGray)
    return imgGray

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
    # print(newSise)
    a = np.empty(shape=sec.shape[0]*sec.shape[1]*8, dtype=int)
    indexA = 0
    sec_flat = sec.flatten()
    # print(len(sec_flat), len(a))
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

def embedSecret(img, newBin, fileName):
    a = np.empty(shape=img.shape, dtype=int)

    for ix in range(0, img.shape[0]):
        for iy in range(0, img.shape[1]):
            a[ix,iy] = int(bin(img[ix,iy])[:-1]+str(int(newBin[ix*img.shape[0]+iy])),2)
    cv2.imwrite(const_path_hasil+fileName, a)

def saveGrey(img,filename):
    cv2.imwrite(const_path_hasil+filename, img)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr