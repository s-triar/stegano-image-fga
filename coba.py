import lib.stegonize as steg
import lib.load_data as ld
import os
import cv2
import numpy as np


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    payloadType = 'image' # image, text, binary_in_text
    file_payload = 'grey_150x150_4.1.06.tiff'
    file_stego = 'grey_st_pt_stgo_grey_150x150_4106_4.2.07.tiff'
    file_cover = '4.2.07.tiff'
    path_stego = os.path.join(path, 'data_hasil_strt_pt','img', file_stego)
    path_payload = os.path.join(path, 'data_test','payload-img', file_payload)
    path_cover = os.path.join(path, 'data_test','cover-min', file_cover)
    stegoImage = ld.load_image(path_stego)
    stegoBin = ld.img_as_bin_img(stegoImage)
    stegoBinCp = (stegoBin.copy()).flatten()
    stegoBinCp0=len(stegoBinCp[stegoBinCp==0])
    stegoBinCp1=len(stegoBinCp[stegoBinCp==255])
    print("stego bin", stegoBinCp0,stegoBinCp1)
    payloadImage = ld.load_image(path_payload)
    payloadBin = ld.img_as_bin_img(payloadImage)
    coverImage = ld.load_image(path_cover)
    coverBin = ld.img_as_bin_img(coverImage)
    imgDiff = ld.imgs_diff(coverBin,stegoBin)
    cv2.imshow('payload', payloadBin)
    cv2.imshow('cover', coverBin)
    cv2.imshow('stego', stegoBin)
    cv2.imshow('stego | cover diff', imgDiff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
