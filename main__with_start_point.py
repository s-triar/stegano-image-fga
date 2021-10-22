from lib_strt_pt.fga_with_sp import Fga as fga
import lib_strt_pt.stegonize_with_sp as steg
import lib_strt_pt.load_data_with_sp as ld
import os
import lib_strt_pt.logger_with_sp as lg
import glob

def DoRun(path, payload_type, payload_file_name, cover_file_name, path_payload_, path_cover_):
    payloadType = payload_type # image, text, binary_in_text

    file_payload = payload_file_name
    file_cover = cover_file_name
    path_cover = path_cover_ #os.path.join(path, 'data_test', file_cover)
    path_payload = path_payload_ #os.path.join(path, 'data_test', file_payload)
    path_hasil = os.path.join(path,'exp_with_sp_0-3_0-3_1000_20.csv')

    payloadBiner = None
    payloadShape = None
    payloadT = 0
    if(payloadType == 'image'):
        payloadData = ld.load_image(path_payload)
        payloadShape=payloadData.shape
        path_save_grey = os.path.join(path, 'data_hasil_strt_pt', 'grey','grey_st_pt_payload_'+file_payload)
        ld.save_img(payloadData,path_save_grey)
        payloadBiner = ld.secret_to_bin(payloadData)
        payloadT = 1
        # print(payloadBiner)
    elif(payloadType == 'text'):
        payloadBiner = ld.load_text(path_payload)
        payloadShape=payloadBiner.shape
        # print(payloadBiner)
    else:
        payloadBiner = ld.load_binary_in_text(path_payload)
        payloadShape=payloadBiner.shape
        # payloadBiner = payloadBiner[:5000]
        # print((payloadBiner))

    twoDPayload = ld.one_d_to_two_d(payloadBiner)
    # print(twoDPayload)
    # print(twoDPayload.shape)
    # load cover image
    coverImage = ld.load_image(path_cover)
    path_save_grey_cover = os.path.join(path, 'data_hasil_strt_pt', 'grey','grey_st_pt_cover_'+"".join(file_payload.split(".")[:-1])+"_"+file_cover)
    ld.save_img(coverImage,path_save_grey_cover)
    coverBin = ld.img_to_bin(coverImage)
    obFga = fga(0.3,0.3,1000,20,coverBin,twoDPayload, payloadShape, payloadT, os.path.join(path,'logs_strt_pt',"".join(file_payload.split(".")[:-1])+'-_-'+"".join(file_cover.split(".")[:-1])+'.csv'))
    newBinImg, bestAt, best = obFga.Run()
    path_save_stego = os.path.join(path, 'data_hasil_strt_pt', 'grey','grey_st_pt_stgo_'+"".join(file_payload.split(".")[:-1])+"_"+file_cover)
    # print("newBinImg",newBinImg, newBinImg.shape)
    stegoImg = steg.embedSecret(coverImage,newBinImg)
    # print("stegoImg",stegoImg, stegoImg.shape)
    ld.save_img(stegoImg,path_save_stego)
    psnr = steg.PSNR(ld.load_image(path_save_grey_cover),ld.load_image(path_save_stego))
    print("PSNR: ",psnr)
    best_str = "".join([str(yy) for yy in best])
    lg.writeResult(path_hasil,
        file_cover, file_payload, path_cover, path_payload, psnr, payloadType, payloadT, 'fga first concept', bestAt, best_str,path_save_stego)

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    list_cover = os.path.join(path, 'data_test', 'cover-min')+'/*.tiff'
    # file_cover = 'boat.512.tiff'
    # file_payload = 'random-binary_1Kb.txt'
    # path_cover = os.path.join(path, 'data_test', file_cover)
    # path_payload = os.path.join(path, 'data_test', 'payload', file_payload)
    payloadType = 'binary_in_text' # image, text, binary_in_text
    raw = os.path.join(path, 'data_test', 'payload')+'/*.txt'
    for cr in glob.glob(list_cover):
        file_cover = cr.split('\\')[-1]
        for f in glob.glob(raw):
            print(f.split('\\')[-1], f)
            try:
                DoRun(path, payloadType, f.split('\\')[-1], file_cover, f, cr)
            except Exception as ex:
                print(ex)
    