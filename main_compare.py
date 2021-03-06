import lib.stegonize as steg
import lib.load_data as ld
import os


def binArrayToInt(ar):
    a =''
    for i in ar:
        a+=str(i)
    return int(a,2)

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    
    payloadType = 'image' # image, text, binary_in_text
    file_payload = 'grey_150x150_4.1.06.tiff'
    file_stego = 'grey_stgo_grey_150x150_4106_boat.512.tiff'
    path_stego = os.path.join(path, 'data_hasil','img', file_stego)
    path_payload = os.path.join(path, 'data_test','payload-img', file_payload)

    payloadBiner = None

    stegoImage = ld.load_image(path_stego)
    stegoBin = ld.img_to_bin(stegoImage)
    flat_stego_bin = stegoBin.flatten()
    print(flat_stego_bin[-70:])
    x_stego_bin = bin(stegoImage.shape[0]-1)[2:]
    y_stego_bin = bin(stegoImage.shape[1]-1)[2:]
    shapeKromosom = None
    print(stegoImage.shape)
    print(x_stego_bin, y_stego_bin)
    payloadType = flat_stego_bin[-1] # image 1, text 0, binary_in_text 0
    genSizeHSecret=0
    genSizeWSecret=0
    wsecret_asli=0
    hsecret_asli=0
    wh=0
    wsecret=0
    hsecret=0
    print("payload type",payloadType)
    if(payloadType==1):
        shapeKromosom = len(x_stego_bin)+len(y_stego_bin)
        genSizeHSecret = flat_stego_bin[-(len(x_stego_bin)+1):-1]
        genSizeWSecret = flat_stego_bin[-(len(x_stego_bin)+len(y_stego_bin)+1):-(len(x_stego_bin)+1)]
        print("genSize",genSizeWSecret, genSizeHSecret)
        wsecret_asli = binArrayToInt(genSizeWSecret)
        hsecret_asli = binArrayToInt(genSizeHSecret)
        print("size secret asli",wsecret_asli,hsecret_asli)
        wh = ld.findFactor((wsecret_asli,hsecret_asli))
        print(wh)
        wsecret = wh[0]
        hsecret = wh[1]
        print("size secret", wsecret,hsecret)
    else:
        shapeKromosom = len(bin((stegoImage.shape[0]*stegoImage.shape[1])-1)[2:])

        genSizeWSecret = flat_stego_bin[-(shapeKromosom+1):-(1)]
        wsecret_asli = binArrayToInt(genSizeWSecret)
        wh = ld.findFactor([wsecret_asli])
        print("wh",wh)
        wsecret = wh[0]
        hsecret = wh[1]
        print("size secret",wsecret,hsecret)
    nKromosom, shiftingSecretData, repeatShifting,swapping,swappingStartPoint,swappingDirection,dataPolarity = steg.generateKromosom(wsecret,hsecret)

    kromosomBin = flat_stego_bin[-(shapeKromosom+nKromosom+1):-(shapeKromosom+1)]
    print(kromosomBin)
    bins = [shiftingSecretData, repeatShifting,swapping,swappingStartPoint,swappingDirection,dataPolarity]
    extractedKromosom = steg.extractKromosom(bins, kromosomBin)
    print(extractedKromosom)
    rearrangeSecretFromStego = flat_stego_bin[:wsecret*hsecret]
    secret = steg.doReverseStego(extractedKromosom,rearrangeSecretFromStego, (wsecret,hsecret))
    secret_flat = secret.flatten()
    print(secret_flat)
    payloadBiner = None

    if(payloadType == 1):
        payloadData = ld.load_image(path_payload)
        payloadBiner = ld.secret_to_bin(payloadData)
        # print(payloadBiner)
    # elif(payloadType == 0):
    #     payloadBiner = ld.load_text(path_payload)
    #     # print(payloadBiner)
    else:
        payloadBiner = ld.load_binary_in_text(path_payload)
        # payloadBiner = payloadBiner[:5000]
        # print((payloadBiner))
    
    c= steg.compareSecret(payloadBiner,secret_flat)
    print(c)
    # print("PSNR: ",psnr)


    # gene
    # 1 shifting secret *
    # 2 repeat shifting *
    # 3 swapping *
    # 4 swapping starting point 1 bit
    # 5 swapping direction 1 bit
    # 6 data polarity
    # 7 n binary = len of biner of width of image => width of secret
    # 8 n binary = len of biner of height of image => heigth of secret