import numpy as np
import math

def embedSecret(img, newBin):
    a = np.empty(shape=img.shape, dtype=int)
    for ix in range(0, img.shape[0]):
        for iy in range(0, img.shape[1]):
            a[ix,iy] = int(bin(img[ix,iy])[:-1]+str(int(newBin[ix*img.shape[0]+iy])),2)
    return a

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def shifting(secret, offset):
    x = np.roll(secret, offset, axis=0)
    return np.roll(x, offset, axis=1)

def deShifting(secret, offset):
    x = np.roll(secret, -offset, axis=1)
    return np.roll(x, -offset, axis=0)

def devide(flat):
    n_sisa = len(flat)%4
    arr_sisa = flat[-n_sisa:] if n_sisa >0 else np.array([])
    return arr_sisa, np.split(flat[:-n_sisa] if n_sisa>0 else flat[:],4)

def inverse_arr(arr):
    temp =[]
    for i in arr:
        temp.append('0' if i=='1' or i ==1 else '1')
    return temp

def swapping(secret, n_member, start_flag, direction, data_polarity):
    flat_secret = secret.flatten()
    arr_sisa, flat = devide(flat_secret)
    if(start_flag == 0):
        pivot = flat[3][:n_member].copy() if data_polarity[3] == '0' else inverse_arr(flat[3][:n_member].copy())
        pivot = pivot if direction =='0' else np.flip(pivot)
        t3 = flat[2][:n_member] if data_polarity[2] == '0' else inverse_arr(flat[2][:n_member])
        t3 = t3 if direction =='0' else np.flip(t3)
        t2 = flat[1][:n_member] if data_polarity[1] == '0' else inverse_arr(flat[1][:n_member])
        t2 = t2 if direction =='0' else np.flip(t2)
        t1 = flat[0][:n_member] if data_polarity[0] == '0' else inverse_arr(flat[0][:n_member])
        t1 = t1 if direction =='0' else np.flip(t1)
        t0 = pivot
        flat[3][:n_member] = t3
        flat[2][:n_member] = t2
        flat[1][:n_member] = t1
        flat[0][:n_member] = t0
    else:
        pivot = flat[3][-n_member:].copy() if data_polarity[3] == '0' else inverse_arr(flat[3][-n_member:].copy())
        pivot = pivot if direction =='0' else np.flip(pivot)
        t3 = flat[2][-n_member:] if data_polarity[2] == '0' else inverse_arr(flat[2][-n_member:])
        t3 = t3 if direction =='0' else np.flip(t3)
        t2 = flat[1][-n_member:] if data_polarity[1] == '0' else inverse_arr(flat[1][-n_member:])
        t2 = t2 if direction =='0' else np.flip(t2)
        t1 = flat[0][-n_member:] if data_polarity[0] == '0' else inverse_arr(flat[0][-n_member:])
        t1 = t1 if direction =='0' else np.flip(t1)
        t0 = pivot
        flat[3][-n_member:] = t3
        flat[2][-n_member:] = t2
        flat[1][-n_member:] = t1
        flat[0][-n_member:] = t0
    return np.concatenate((np.array(flat).flatten(),arr_sisa))

def deSwapping(shape, flat_secret, n_member, start_flag, direction, data_polarity):
    arr_sisa, flat = devide(flat_secret)
    if(start_flag == 0):
        pivot = flat[0][:n_member].copy() if data_polarity[0] == '0' else inverse_arr(flat[0][:n_member].copy())
        pivot = pivot if direction =='0' else np.flip(pivot)
        t0 = flat[1][:n_member] if data_polarity[1] == '0' else inverse_arr(flat[1][:n_member])
        t0 = t0 if direction =='0' else np.flip(t0)
        t1 = flat[2][:n_member] if data_polarity[2] == '0' else inverse_arr(flat[2][:n_member])
        t1 = t1 if direction =='0' else np.flip(t1)
        t2 = flat[3][:n_member] if data_polarity[3] == '0' else inverse_arr(flat[3][:n_member])
        t2 = t2 if direction =='0' else np.flip(t2)
        t3 = pivot
        flat[0][:n_member] = t0
        flat[1][:n_member] = t1
        flat[2][:n_member] = t2
        flat[3][:n_member] = t3
    else:
        pivot = flat[0][-n_member:].copy() if data_polarity[0] == '0' else inverse_arr(flat[0][-n_member:].copy())
        pivot = pivot if direction =='0' else np.flip(pivot)
        t0 = flat[1][-n_member:] if data_polarity[1] == '0' else inverse_arr(flat[1][-n_member:])
        t0 = t0 if direction =='0' else np.flip(t0)
        t1 = flat[2][-n_member:] if data_polarity[2] == '0' else inverse_arr(flat[2][-n_member:])
        t1 = t1 if direction =='0' else np.flip(t1)
        t2 = flat[3][-n_member:] if data_polarity[3] == '0' else inverse_arr(flat[3][-n_member:])
        t2 = t2 if direction =='0' else np.flip(t2)
        t3 = pivot
        flat[0][-n_member:] = t0
        flat[1][-n_member:] = t1
        flat[2][-n_member:] = t2
        flat[3][-n_member:] = t3
    temp = np.concatenate((np.array(flat).flatten(),arr_sisa))
    return np.reshape(temp,shape)

