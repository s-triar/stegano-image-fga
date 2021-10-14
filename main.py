from lib.fga import Fga as fga
import lib.stegonize as steg
import lib.image as t_img

if __name__ == "__main__":
    formatfilen = '.tiff'
    filen = "4.2.03"
    img = t_img.load_image(filen+formatfilen)
    t_img.saveGrey(img,filen+'_gray'+formatfilen)
    sec = t_img.load_secret('xxx2.jpg')
    img_bin = t_img.img2bin(img)
    print("img bin", img_bin.shape)
    sec_bin = t_img.secret2bin(sec)
    print("sec_bin",sec_bin.shape)
    obFga = fga(0.1,0.1,4,10,20,img_bin,sec_bin)
    newBinImg = obFga.Run()
    t_img.embedSecret(img,newBinImg,filen+'_stego'+formatfilen)
    psnr = t_img.PSNR(t_img.load_image_hasil(filen+'_gray'+formatfilen),t_img.load_image_hasil(filen+'_stego'+formatfilen))
    print("PSNR: ",psnr)