import numpy as np
import cv2
import sys
from deepgaze.color_detection import BackProjectionColorDetector
from matplotlib import pyplot as plt
from PIL import Image


image_file = sys.argv[1]
imageName = sys.argv[2]
isPorn = sys.argv[3]

text_file = open("Output.csv", "a+")
# Cargamos la imagen
original = cv2.imread(image_file)

# Imagen HSV
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

# Imagen YCrCb
YCrCb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)

# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)

# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150) 
 
# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
imgCont=cv2.drawContours(original,contornos,-1,(0,0,255), 2)


# Imagen Histograma 
color = ('b','g','r')
hist,bins = np.histogram(original.ravel(),256,[0,256])

#text_file.write(' '.join(str(e) for e in hist))
#text_file.write(',')
#text_file.write(' '.join(str(e) for e in bins))


# Skin Detector

def get_skin_rgb(im):
    im = im.crop((int(im.size[0]*0.2), int(im.size[1]*0.2), im.size[0]-int(im.size[0]*0.2), im.size[1]-int(im.size[1]*0.2)))
    skin = sum([count for count, rgb in im.getcolors(im.size[0]*im.size[1]) if rgb[0]>60 and rgb[1]<(rgb[0]*0.85) and rgb[2]<(rgb[0]*0.7) and rgb[1]>(rgb[0]*0.4) and rgb[2]>(rgb[0]*0.2)])
    return float(skin)/float(im.size[0]*im.size[1])

im = Image.open(image_file)
skin_percent = get_skin_rgb(im) * 100
out = '%.1f' % (skin_percent)
text_file.write(out)
text_file.write(',')

#Skin detecto hsv

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    skin = False
    
    if(h >=0 and h<= 50 and s>0.23 and s<=0.68) :
            skin = True

    return skin

pixels = list(im.getdata())
perskin = 0
skin = 0
i = 0

for pix in pixels:
       i=i+1
       if(rgb2hsv(pix[0],pix[1],pix[2])):
             skin=skin+1


x = (skin * 100) / i

out2 = '%.1f' % (x)


text_file.write(out2)

text_file.write(',')
text_file.write(isPorn)
text_file.write('\r\n')

text_file.close()
