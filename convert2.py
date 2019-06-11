import numpy as np
import cv2
import sys

from matplotlib import pyplot as plt
from PIL import Image


image_file = sys.argv[1]

# Cargamos la imagen
original = cv2.imread(image_file)
cv2.imshow("original", original)

# Imagen HSV
hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

cv2.imshow("imagen hsv", hsv)

# Imagen YCrCb
YCrCb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)

cv2.imshow("imagen YCrCb", YCrCb)


# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)
 
cv2.imshow("imagen suavizado", gauss)
 
# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)
 
cv2.imshow("imagen canny", canny)
 
# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
 
imgCont=cv2.drawContours(original,contornos,-1,(0,0,255), 2)
cv2.imshow("imagen contornos", original)

cv2.imwrite('temp/temp.png',imgCont)

# Imagen Histograma 
color = ('b','g','r')
hist,bins = np.histogram(original.ravel(),256,[0,256])


plt.figure(1)
for i,col in enumerate(color):
    histr = cv2.calcHist([original],[i],None,[100],[0,100])
    plt.plot(histr,color = col)
    plt.xlim([0,100])
plt.title('Histograma')

# Skin Detector

def get_skin_rgb(im):
    im = im.crop((int(im.size[0]*0.2), int(im.size[1]*0.2), im.size[0]-int(im.size[0]*0.2), im.size[1]-int(im.size[1]*0.2)))
    skin = sum([count for count, rgb in im.getcolors(im.size[0]*im.size[1]) if rgb[0]>60 and rgb[1]<(rgb[0]*0.85) and rgb[2]<(rgb[0]*0.7) and rgb[1]>(rgb[0]*0.4) and rgb[2]>(rgb[0]*0.2)])
    return float(skin)/float(im.size[0]*im.size[1])

im = Image.open('temp/temp.png')
skin_percent = get_skin_rgb(im) * 100
out = '%.1f' % (skin_percent)


plt.figure(2)
plt.subplot(331),plt.text(0, 0.5, skin_percent, fontsize=12),plt.title('% De Piel RGB')
plt.xticks([]), plt.yticks([])

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



plt.subplot(332),plt.text(0, 0.5, x, fontsize=12),plt.title('% De Piel hsv')
plt.xticks([]), plt.yticks([])
plt.show()