from lib.fga import Fga as fga
import lib.stegonize as steg
import lib.load_data as ld
import os
if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    
    payloadType = 'binary_in_text' # image, text, binary_in_text

    file_payload = 'random-binary_10Kb.txt'
    file_cover = 'boat.512.tiff'
    path_cover = os.path.join(path, 'data_test', file_cover)
    path_payload = os.path.join(path, 'data_test','payload', file_payload)
    path_hasil = os.path.join(path, 'data_hasil', payloadType,"test.txt")

    payloadBiner = None

    if(payloadType == 'image'):
        payloadData = ld.load_image(path_payload)
        path_save_grey = os.path.join(path, 'data_hasil', 'grey','grey_payload_'+file_payload)
        ld.save_img(payloadData,path_save_grey)
        payloadBiner = ld.secret_to_bin(payloadData)
        # print(payloadBiner)
    elif(payloadType == 'text'):
        payloadBiner = ld.load_text(path_payload)
        # print(payloadBiner)
    else:
        payloadBiner = ld.load_binary_in_text(path_payload)
        payloadBiner = payloadBiner[:1000]
        print((payloadBiner.shape))

    twoDPayload = ld.one_d_to_two_d(payloadBiner)
    # print(twoDPayload)
    # print(twoDPayload.shape)

    # load cover image
    coverImage = ld.load_image(path_cover)
    path_save_grey_cover = os.path.join(path, 'data_hasil', 'grey','grey_cover_'+file_cover)
    ld.save_img(coverImage,path_save_grey_cover)
    coverBin = ld.img_to_bin(coverImage)
    obFga = fga(0.3,0.3,4,100,20,coverBin,twoDPayload)
    newBinImg = obFga.Run()
    path_save_stego = os.path.join(path, 'data_hasil', 'grey','grey_stgo_'+file_cover)
    stegoImg = steg.embedSecret(coverImage,newBinImg)
    ld.save_img(stegoImg,path_save_stego)
    psnr = steg.PSNR(ld.load_image(path_save_grey_cover),ld.load_image(path_save_stego))
    print("PSNR: ",psnr)