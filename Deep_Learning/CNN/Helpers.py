import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

def printbasesimple(base):
        print(tabulate(base.reshape((28, 28))))

def printgray(base, titre="", dims=(28, 28)):
    img = base.reshape(dims)
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title(titre)
    plt.colorbar(label='Value')
    plt.show()

def printimage(base, titre=""):
    img = base.transpose(1,2,0)
    plt.imshow(img)
    plt.axis("off")
    plt.title(titre)
    plt.show()

def converttogreyscale(rgbimage):
        return np.tensordot(rgbimage,np.array([0.299, 0.587, 0.114]), (1, 2))

def flatening(image):
        return image.reshape((-1,1))

def paddington(image, padavant, padapres): #padavant ce qu'on ajoute a la ligne et l'autre est evident
        return np.pad(image, ((0,0), (padavant, padapres), (padavant, padapres))) # padding

# EL RESTO ES SOLO PARA SABER TIEMPO DE CADA FUNCION

# from statistics import mean
# from functools import wraps
# from collections import defaultdict
# import atexit
# import time
#
#
# execution_times = defaultdict(list)
#
# def timed(method):
#     @wraps(method)
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = method(*args, **kwargs)
#         end = time.time()
#         execution_times[method.__name__].append(end - start)
#         return result
#     return wrapper
#
# def maxdico(d):
#     m = (None, 0)
#     for i in d.keys():
#         t = mean(d[i])
#         if t > m[1]:
#             m = (i, t)
#     return m
#
# @atexit.register
# def print_avg_times():
#     print("\n--- Temps moyens d'exécution ---")
#     for name, times in execution_times.items():
#         avg = mean(times)
#         print("__________________________________________________________________________________")
#         print(f"{name}: {avg} s    (appelée {len(times)} fois)")
#
#     print("______________________________________________________________________________________")
#     a = maxdico(execution_times)
#     print(f"Le maximum de temps est de {a[1]} par {a[0]}")
