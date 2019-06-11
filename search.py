import os
import sys
import numpy as np
from os import listdir
from os.path import isfile, isdir, join

# Recorremos los archivos
def listdir_recurd(files_list, root, folder, checked_folders):

    if (folder != root):
        checked_folders.append(folder)

    for f in listdir(folder):
        d = join(folder, f)       

        if isdir(d) and d not in checked_folders:
            listdir_recurd(files_list, root, d, checked_folders)
        else:
            if isfile(d):  # si no hago esto, inserta en la lista el nombre de las carpetas ignoradas
                files_list.append(join(folder, f))

    return files_list

filesn = listdir_recurd([], 'test/n', 'test/n', []) # Este es el path o ruta de la carpeta deberia ser (dataset/0)
filesy = listdir_recurd([], 'test/y', 'test/y', []) # Y este deberia ser (dataset/1)

for archivo in filesn:
    x = archivo.split("/")
    n = x[len(x)-1]
    myCmd = 'python3 convert.py ' + archivo + ' '+ n + ' 0'
    os.system(myCmd)

for archivo in filesy:
    x = archivo.split("/")
    y = x[len(x)-1]
    myCmd = 'python3 convert.py ' + archivo + ' '+ y + ' 1'
    os.system(myCmd)