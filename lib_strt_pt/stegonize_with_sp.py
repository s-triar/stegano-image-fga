import numpy as np
import math

def embedSecret(img, newBin):
    a = np.empty(shape=img.shape, dtype=np.uint8)
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

def startPointCover(coor,type, cover):
    # type = 0 => normal => ke kanan
    # type = 1 => transpose => ke bawah
    # type = 2 => reverse => ke kiri
    # type = 3 => transpose + reverse => ke atas
    n_roll = (coor[0]*cover.shape[0]) + coor[1]
    if(type==1):
        a = np.transpose(cover)
        a = a.flatten()
        a = np.roll(a,n_roll)
        return a
    elif(type==2):
        a = np.flip(cover)
        a = a.flatten()
        a = np.roll(a,n_roll)
        return a
    elif(type==3):
        a = np.transpose(cover)
        a = np.flip(a)
        a = a.flatten()
        a = np.roll(a,n_roll)
        return a
    else:
        a = cover.flatten()
        a = np.roll(a,n_roll)
        return a

def deStartPointCover(coor, type, cover, shape):
    # type = 0 => normal => ke kanan
    # type = 1 => transpose => ke bawah
    # type = 2 => reverse => ke kiri
    # type = 3 => transpose + reverse => ke atas
    n_roll = (coor[0]*shape[0]) + coor[1]
    if(type==1):
        a = np.roll(cover,-n_roll)
        a = np.reshape(a,(shape[1],shape[0]))
        a = np.transpose(a)
        return a
    elif(type==2):
        a = np.roll(cover,-n_roll)
        a = np.reshape(a,shape)
        a = np.flip(a)
        return a
    elif(type==3):
        a = np.roll(cover,-n_roll)
        a = np.reshape(a,(shape[1],shape[0]))
        a = np.flip(a)
        a = np.transpose(a)
        return a
    else:
        a = np.roll(cover,-n_roll)
        a = np.reshape(a,shape)
        return a


def devide(flat):
    n_sisa = len(flat)%4
    arr_sisa = flat[-n_sisa:] if n_sisa >0 else np.array([])
    return arr_sisa, np.split(flat[:-n_sisa] if n_sisa>0 else flat[:],4)

def inverse_arr(arr):
    temp =[]
    for i in arr:
        temp.append(0 if i ==1 else 1)
    return temp

def swapping(secret, n_member, start_flag, direction, data_polarity):
    flat_secret = secret.flatten()
    arr_sisa, flat = devide(flat_secret)
    if(start_flag == 0):
        pivot = flat[3][:n_member].copy() if data_polarity[3] == '0' else inverse_arr(flat[3][:n_member].copy())
        pivot = pivot if direction ==0 else np.flip(pivot)
        t3 = flat[2][:n_member].copy() if data_polarity[2] == '0' else inverse_arr(flat[2][:n_member].copy())
        t3 = t3 if direction ==0 else np.flip(t3)
        t2 = flat[1][:n_member].copy() if data_polarity[1] == '0' else inverse_arr(flat[1][:n_member].copy())
        t2 = t2 if direction ==0 else np.flip(t2)
        t1 = flat[0][:n_member].copy() if data_polarity[0] == '0' else inverse_arr(flat[0][:n_member].copy())
        t1 = t1 if direction ==0 else np.flip(t1)
        t0 = pivot
        flat[3][:n_member] = t3
        flat[2][:n_member] = t2
        flat[1][:n_member] = t1
        flat[0][:n_member] = t0
    else:
        pivot = flat[3][-n_member:].copy() if data_polarity[3] == '0' else inverse_arr(flat[3][-n_member:].copy())
        pivot = pivot if direction ==0 else np.flip(pivot)
        t3 = flat[2][-n_member:].copy() if data_polarity[2] == '0' else inverse_arr(flat[2][-n_member:].copy())
        t3 = t3 if direction ==0 else np.flip(t3)
        t2 = flat[1][-n_member:].copy() if data_polarity[1] == '0' else inverse_arr(flat[1][-n_member:].copy())
        t2 = t2 if direction ==0 else np.flip(t2)
        t1 = flat[0][-n_member:].copy() if data_polarity[0] == '0' else inverse_arr(flat[0][-n_member:].copy())
        t1 = t1 if direction ==0 else np.flip(t1)
        t0 = pivot
        flat[3][-n_member:] = t3
        flat[2][-n_member:] = t2
        flat[1][-n_member:] = t1
        flat[0][-n_member:] = t0
    return np.concatenate((np.array(flat).flatten(),arr_sisa))

def deSwapping(shape, flat_secret, n_member, start_flag, direction, data_polarity):
    arr_sisa, flat = devide(flat_secret)
    if(start_flag == 0):
        pivot = flat[0][:n_member].copy() if data_polarity[3] == '0' else inverse_arr(flat[0][:n_member].copy())
        pivot = pivot if direction ==0 else np.flip(pivot)
        t0 = flat[1][:n_member].copy() if data_polarity[0] == '0' else inverse_arr(flat[1][:n_member].copy())
        t0 = t0 if direction ==0 else np.flip(t0)
        t1 = flat[2][:n_member].copy() if data_polarity[1] == '0' else inverse_arr(flat[2][:n_member].copy())
        t1 = t1 if direction ==0 else np.flip(t1)
        t2 = flat[3][:n_member].copy() if data_polarity[2] == '0' else inverse_arr(flat[3][:n_member].copy())
        t2 = t2 if direction ==0 else np.flip(t2)
        t3 = pivot
        flat[0][:n_member] = t0
        flat[1][:n_member] = t1
        flat[2][:n_member] = t2
        flat[3][:n_member] = t3
    else:
        pivot = flat[0][-n_member:].copy() if data_polarity[3] == '0' else inverse_arr(flat[0][-n_member:].copy())
        pivot = pivot if direction ==0 else np.flip(pivot)
        t0 = flat[1][-n_member:].copy() if data_polarity[0] == '0' else inverse_arr(flat[1][-n_member:].copy())
        t0 = t0 if direction ==0 else np.flip(t0)
        t1 = flat[2][-n_member:].copy() if data_polarity[1] == '0' else inverse_arr(flat[2][-n_member:].copy())
        t1 = t1 if direction ==0 else np.flip(t1)
        t2 = flat[3][-n_member:].copy() if data_polarity[2] == '0' else inverse_arr(flat[3][-n_member:].copy())
        t2 = t2 if direction ==0 else np.flip(t2)
        t3 = pivot
        flat[0][-n_member:] = t0
        flat[1][-n_member:] = t1
        flat[2][-n_member:] = t2
        flat[3][-n_member:] = t3
    temp = np.concatenate((np.array(flat).flatten(),arr_sisa))
    return np.reshape(temp,shape)


def generateKromosom(x,y, cover_shape):
    start_point_x = len(bin(cover_shape[0]-1)[2:])
    start_point_y = len(bin(cover_shape[1]-1)[2:])
    scan_dir = 2
    shiftingSecretData = len(bin(min(x,y)-1)[2:])
    repeatShifting = len(bin(min(x,y)-1)[2:])
    swapping = len(bin(int((x*y)/4)-1)[2:])
    swappingStartPoint = 1
    swappingDirection = 1
    dataPolarity = 4
    nKromosom = start_point_x+start_point_y+scan_dir+ shiftingSecretData+repeatShifting+swapping+swappingStartPoint+swappingDirection+dataPolarity
    return nKromosom,  start_point_x, start_point_y, scan_dir, shiftingSecretData, repeatShifting,swapping,swappingStartPoint,swappingDirection,dataPolarity

def extractKromosom(bins, individu):
        # bins = [self.shiftingSecretData, self.repeatShifting,self.swapping,self.swappingStartPoint,self.swappingDirection,self.dataPolarity]
        startInd = 0
        intChr =[]
        # print(intChr)
        for i in range(0, len(bins)-1):
            r = ""
            startIndCP =startInd
            for ichr in range(startIndCP, bins[i]+startIndCP):
                r = r + str(individu[ichr])
                startInd=ichr+1
            # print(r)
            intChr.append(int(r,2))
            # print(intChr)
        t=""
        for i in individu[-bins[-1]:]: 
            t=t+str(i)
        intChr.append(t)
        
        return intChr
    
def doStegano(secret, intChro):
    secretCopy = secret
    offset = intChro[3]
    repeatShift = intChro[4]
    for i in range(0, repeatShift):
        secretCopy = shifting(secretCopy,offset)
    # secretCopy = secretCopy.flatten()
    n_member, start_flag, direction, data_polarity = intChro[5], intChro[6], intChro[7], intChro[8]
    secretCopy = swapping(secretCopy,n_member, start_flag, direction, data_polarity)
    return secretCopy

def doReverseStego(intChro, stego, shape):
    secretCopy = stego.copy()
    n_member, start_flag, direction, data_polarity = intChro[5], intChro[6], intChro[7], intChro[8]
    secretCopy = deSwapping(shape, secretCopy, n_member, start_flag, direction, data_polarity)
    # secretCopy = np.reshape(secretCopy,shape)
    offset = intChro[3]
    repeatShift = intChro[4]
    for i in range(0, repeatShift):
        secretCopy = deShifting(secretCopy,offset)
    return np.array(secretCopy,dtype=np.int8)

def compareSecret(before,after):
    c = np.logical_xor(before,after)
    return np.sum(c==True)


