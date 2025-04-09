import pickle
import numpy as np

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

def takeinputs(basededonnees):
    if basededonnees == "mnist":
        # MNIST
        with open('Datas/Mnist/valeursentraine', 'rb') as f:
            valeurs = np.array(pickle.load(f))

        with open('Datas/Mnist/pixelsentraine', 'rb') as f:
            pixels = np.array(pickle.load(f)).T

        with open('Datas/Mnist/testval', 'rb') as f:
            qcmval = pickle.load(f)

        with open('Datas/Mnist/testpix', 'rb') as f:
            qcmpix = np.array(pickle.load(f)).T

        perm = np.random.permutation(pixels.shape[1])

        pixmelange = pixels[:, perm]
        valmelange = valeurs[perm]

        labels_mnist = {str(i):i for i in range(10)}

        return valmelange, pixmelange, qcmval, qcmpix, labels_mnist

    elif basededonnees == "fashion":
        # MNIST_FASHION
        with open('Datas/fashion-mnist/pixentraine_fashion', 'rb') as f:
            fashion_pix = np.array(pickle.load(f)).reshape(10000, 784).T
        with open('Datas/fashion-mnist/valentraine_fashion', 'rb') as f:
            fashion_val = np.array(pickle.load(f))

        perm2 = np.random.permutation(fashion_pix.shape[1])

        fashion_pix_melange = fashion_pix[:, perm2]
        fashion_val_melange = fashion_val[perm2]

        fashion_pix_entraine = fashion_pix_melange[:, :8000]
        fashion_val_entraine = fashion_val_melange[:8000]

        fashion_pix_test = fashion_pix_melange[:, 8000:]
        fashion_val_test = fashion_val_melange[8000:]

        labels_fashion = {0: "T-shirt/Top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt",
                          7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

        return fashion_val_entraine, fashion_pix_entraine, fashion_val_test, fashion_pix_test, labels_fashion

    elif basededonnees == "ciphar-10":
        raise ValueError("Pasencoreprêt")
        with open('Datas/Ciphar-10/pixentraine_cifar10', 'rb') as f:
            ciphar_pixentraine = pickle.load(f)
        with open("Datas/Ciphar-10/valentraine_cifar10", "rb") as f:
            ciphar_valentraine = pickle.load(f)
        with open("Datas/Ciphar-10/pixtest_cifar10", "rb") as f:
            ciphar_pixtest = pickle.load(f)
        with open("Datas/Ciphar-10/valtest_cifar10", "rb") as f:
            ciphar_valtest = pickle.load(f)

        labels_ciphar = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

        return ciphar_valentraine, ciphar_pixentraine, ciphar_valtest, ciphar_pixtest, labels_ciphar

    else:
        raise ValueError("Pas de base de données dispo!")

class Draw:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Paint")

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.pack()

        self.dessine = False

        self.posx, self.posy = None,None

        self.butons = []

        self.image = Image.new("L", (280, 280), 255)
        self.drawing = ImageDraw.Draw(self.image)

        self.pixels = None

        self.creerboutons()

        self.root.mainloop()

    def creerboutons(self):
        imprimer = tk.Button(self.root, text="Print", command=self.imprime)
        imprimer.pack(side=tk.LEFT)

        fermer = tk.Button(self.root, text="Fermer", command=self.root.destroy)
        fermer.pack(side=tk.LEFT)

        self.butons.append(imprimer)
        self.butons.append(fermer)


        self.canvas.bind("<Button-1>", self.commence)
        self.canvas.bind("<ButtonRelease-1>", self.arret)
        self.canvas.bind("<B1-Motion>", self.draw)

    def commence(self, event):
        self.dessine = True
        self.posx, self.posy = event.x, event.y

    def arret(self, event):
        self.dessine = False

    def draw(self, event):
        if self.dessine:
            x, y = event.x, event.y

            self.canvas.create_line((self.posx, self.posy, x, y), fill="white", width=10)

            self.drawing.line([self.posx, self.posy, x, y], fill=0, width=8)

            self.posx, self.posy = x, y


    def imprime(self):
        im = self.image.resize((28, 28), Image.Resampling.LANCZOS).convert("L")

        im = ImageOps.invert(im)

        self.pixels = np.array(im.getdata()).reshape(1, 28, 28)


        self.root.destroy()


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
