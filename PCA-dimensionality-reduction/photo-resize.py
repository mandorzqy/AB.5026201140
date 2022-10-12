import os
import os.path
from PIL import Image

#nama folder gambar original
folders = ['<nama folder yang berisi gambar>']

#path dataset gambar
path = '<path dari folders>'

#membuka masing-masing folder
for folder in folders[:6]:
    x = 0
    for filename in os.listdir(path + folder):
        x += 1
        img_ori = Image.open(path + folder + '/'+ filename)
        img_resize1 = img_ori.resize((356, 200)) #resize ukuran menjadi 356px x 200px
        img_resize1.save(path + folders[6] + '/' + folder + '_'+ str(x) + '.jpg') #save ke folder baru
