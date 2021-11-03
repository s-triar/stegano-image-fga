import csv
def writeResult(pathFile, cover, secret, path_cover, path_secret, psnr, payload_type, payload_type_int, algo, best_at, best, path_stego):
    with open(pathFile, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        data = [cover, secret, path_cover, path_secret, psnr, payload_type, payload_type_int, algo,best_at, best, path_stego]
        writer.writerow(data)
    