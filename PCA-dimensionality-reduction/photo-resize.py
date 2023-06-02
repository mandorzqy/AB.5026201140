import os
import os.path
from PIL import Image, ImageEnhance

#nama folder gambar original
folders = ["B1972VBA", "BG1468AX", "L1412EY", "L1559AAT", "W1025ED", "W1046BM", "resized"]

#path dataset gambar
path = 'E:/Projects/analitika-bisnis/dataset-membersihkan/'

#membuka masing-masing folder
for folder in folders[:6]:
    x = 0
    for filename in os.listdir(path + folder):
        x += 1
        img_ori = Image.open(path + folder + '/'+ filename)
        img_resize1 = img_ori.resize((250, 100)) #resize ukuran menjadi 356px x 200px
        enhancer2 = ImageEnhance.Sharpness(img_resize1)       
        img_output = enhancer2.enhance(1.4)
        img_output.save(path + folders[6] + '/' + folder + '_'+ str(x) + '.jpg') #save ke folder baru
