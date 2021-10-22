from lib_strt_pt.fga_with_sp import Fga as fga
import lib_strt_pt.stegonize_with_sp as steg
# import lib_strt_pt.image_with_sp as t_img
import numpy as np
import lib_strt_pt.load_data_with_sp as ld
import os

if __name__ == "__main__":
    asli = [[1,1,0,1,1,1,0],
            [1,1,1,1,0,1,0],
            [0,0,0,0,1,0,1],
            [0,0,1,0,1,0,1],
            [0,0,1,0,1,0,1],
            [0,0,1,0,1,1,1],
            [0,0,1,0,1,1,1],
            [0,0,1,1,1,1,0]]
    secret_asli = [[0,1,1,1,0],[1,1,0,1,0]]
    a = np.array(asli, dtype=np.int8)
    secret = np.array(secret_asli, dtype=np.int8)
    print('asli')
    print(a)
    extractKromosom = [3,2,2,3,3,3,4,0,'1010']
    secretRearranged = steg.doStegano(secret,extractKromosom)
    img_bin = steg.startPointCover([extractKromosom[0],extractKromosom[1]],extractKromosom[2],a.copy()[:-1,:])
    emb = np.concatenate((secretRearranged, img_bin.copy()[len(secretRearranged):]))
    emb = steg.deStartPointCover([extractKromosom[0],extractKromosom[1]],extractKromosom[2],emb, a.copy()[:-1,:].shape)
    emb = emb.flatten()
    emb = np.concatenate((emb, (a.copy()[-1,:]).flatten()))
    emb = np.reshape(emb, a.shape)
    print("emb")
    print(emb)

    flat_stego_bin = steg.startPointCover([extractKromosom[0],extractKromosom[1]],extractKromosom[2],emb.copy()[:-1,:])
    rearrangeSecretFromStego = flat_stego_bin[:secret.shape[0]*secret.shape[1]]
    secret_ex = steg.doReverseStego(extractKromosom,rearrangeSecretFromStego, (secret.shape[0],secret.shape[1]))
    print("secret_ex")
    print(secret_ex)