"""This CNN is not very well coded i am conscious.
It is a work in progress, it was more to try to code it without insight just trying to make it work.
I am recoding it more clearly and well structured.

Things to improve for next time:
    - improve how the parameters are passed (arguments are too difficult)
    - make batch for CNN
    - look up if it is better with numpy or scipy (i think numpy)"""

#GENERAL
import pickle
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
# CONV
from scipy.signal import correlate2d
from skimage.measure import block_reduce

# ORGANIZACION
from Deep_Learning.CNN.CNN1.Import_data import processdata
from Parameters import Parametros
from Drawing import Draw
from Helpers import printbasesimple, printgray, printimage, flatening, paddington
from Functions import geterrorfunc, getfct

# PARA EL FUTURO:
# - batch para convolucion
# - Dropout layer
# - try to draw continuo que no se cierre la pestaña
# - import directamente en la clase para que sea mas facil
# - mejorar estructura
# - guardar mejor modelo directamente la instance con pickle
# - opti learning rate
# - verifier que cost diminue

class CNN:
    def __init__(self, par = Parametros):

        self.iter = par.iterations  # nombre iteration entrainement
        self.nblay = len(par.infolay) # nombre de layers
        self.lenbatch = par.batch #longueur du batch, implémenté seulement sans convolution

        self.base = par.base

        # INITIALISATION VARIABLES
        self.cvcoef = par.coefcv #learning rate

        # POUR CNN
        self.nbconv = len(par.infoconvlay)
        self.lenkernel = par.kernel #lado filtro
        self.padding = 0 #par.padding #espacio con bordes
        self.stride = 1 #par.stride #de cuanto se mueve el filtro
        self.poolstride = par.kernelpool
        self.lenkernelpool = par.kernelpool

        self.convdims = [] # (dimconv, dimpool, nbfiltresentree, nbfiltressortie)
        self.dimkernels = [] #dimensions des filtres
        self.dimconvbiais = [] #dimensions biais convolution

        # QUELLES FCTS UTILISER
        if par.poolnp:
            self.pooling = self.poolingnp
        else:
            self.pooling = self.poolingskim

        if par.convnp:
            self.convolution = self.convolutionnp
        else:
            self.convolution = self.convolutionscp

        if par.backconvnp:
            self.backconvolution = self.backconvolutionnp
        else:
            self.backconvolution = self.backconvolutionscp

        par.infoconvlay = [(1 if self.base=="mnist" or self.base=="fashion" else 3, "input")] + par.infoconvlay

        d = 28 if self.base=="mnist" or self.base=="fashion" else par.pix[0].shape[1]

        for i in range(self.nbconv): # CRÉER LES DIMENSIONS D'OUTPUT CONV LAYERS
            dim = int(((d + 2 * self.padding - self.lenkernel) / self.stride) + 1) #dimension apres convolution
            if par.infoconvlay[i+1][2]:
                dimpool = int(((dim - self.lenkernelpool) / self.poolstride) + 1)  #dimension apres pooling layer si dimensions paires
            else:
                dimpool = dim

            self.convdims.append((dim, dimpool))
            d = dimpool

        if self.nbconv > 0: # DIMENSIONS WEIGHT PREMIER LAYER MLP
            par.infolay = [(self.convdims[-1][1], "input")] + par.infolay
        else:
            par.infolay = [(par.pix.shape[0], "input")] + par.infolay

        # INPUTS POUR ENTRAINEMENT
        # self.pix = self.processdata(par.pix, par.pix.shape[0]==3072, False, self.nbconv>0) #pix de train
        self.pix = processdata(self.base, par.pix, False, self.nbconv > 0)
        self.vales = par.vales #val de train
        self.labels = par.labels


        # BASE DE DONNÉES POUR LES TESTS
        # self.qcmpix = self.processdata(par.qcmpix, par.pix.shape[0]==3072, True, self.nbconv>0)
        self.qcmpix = processdata(self.base, par.qcmpix,True, self.nbconv > 0)
        self.qcmval = par.qcmval

        self.dimweights = []  # dimensiones pesos para backprop
        self.dimbiais = []  # dimensions des biais du fully connected

        self.parameters = self.params(par.infolay, par.infoconvlay) #creer les parametres dans un dico/ infolay doit avoir tout au debut la longueur de l'input

        self.errorfunc = geterrorfunc(par.errorfunc) #choisir la fonction d'erreur

        self.aprentissagedynamique = par.apprentissagedynamique # estce qu'il apprned au fur et à mesure qu'il passe le qcm
        self.tauxfiniter = par.tauxfiniter # faire ou non a la fin de chaque itération un calcul de taux d'erreur

        # PARA LEARNING RATE ADAPTATIF (basé sur RMSprop) (ne fonctionne pas totalement encore)
        self.RMSprop = par.RMSprop  # si l'autre est activé on le désactive
        self.beta = par.beta # decay rate
        #POUR GARDER LES MOYENNES DES GRADIENTS DES PARAMETRES
        self.moyencl = [np.zeros(i) for i in self.dimkernels]
        self.moyencb = [np.zeros(i) for i in self.dimconvbiais]
        self.moyenw = [np.zeros(i) for i in self.dimweights]
        self.moyenb = [np.zeros(i) for i in self.dimbiais]

        self.convlay = par.infoconvlay
        self.lay = par.infolay

        self.tauxinitial = self.tauxerreur()

    def params(self, infolay, infoconvlay): #infolay liste avec un tuple avec (nbneurons, fctactivation) / infoconvlay (nbfiltres, fct)
        param = {}

        for c in range(1, len(infoconvlay)):
            param["cl" + str(c-1)] = np.random.uniform(-1, 1, size= (infoconvlay[c][0], infoconvlay[c-1][0], self.lenkernel, self.lenkernel)) # kernel: (nb canaux sortie, nb canaux entree, hauteur filtre, largeur filtre)
            self.dimkernels.append((infoconvlay[c][0], infoconvlay[c-1][0], self.lenkernel, self.lenkernel))

            param["cb" + str(c-1)] = np.zeros((infoconvlay[c][0], self.convdims[c-1][0], self.convdims[c-1][0])) # biais: (canaux sortie, hauteur output, largeur output)
            self.dimconvbiais.append((infoconvlay[c][0], self.convdims[c-1][0], self.convdims[c-1][0]))

            param["fctcl" + str(c-1)] = getfct(infoconvlay[c][1], self.cvcoef)
            param["pool" + str(c-1)] = infoconvlay[c][2]

            self.convdims[c-1] = (self.convdims[c-1][0], self.convdims[c-1][1], infoconvlay[c-1][0], infoconvlay[c][0]) #añadir el numero filtros entrada y salida

        if self.nbconv > 0:
            infolay[0] = (infolay[0][0]*infolay[0][0]*self.convdims[self.nbconv-1][3], "input") #ajustar para que primer peso tenga buenas dim

        for l in range(1, len(infolay)):
            param["w" + str(l-1)] = np.random.uniform(-1, 1, (infolay[l][0], infolay[l-1][0])) #nbneurons * nbinput
            self.dimweights.append((infolay[l][0], infolay[l-1][0]))
            param["b" + str(l-1)] = np.zeros((infolay[l][0], 1))
            self.dimbiais.append((infolay[l][0], 1))

            param["fct" + str(l-1)] = getfct(infolay[l][1], self.cvcoef)[0]
            param["diff" + str(l-1)] = getfct(infolay[l][1], self.cvcoef)[1]

        return param

    def convolutionnp(self, image, kernel, *, mode="valid", reverse=False):  # 2 casos dependiendo de shape kernel y imagen
            lenkernel = kernel.shape  # Csortie, Centree, H,L

            if mode == "full":
                newimage = paddington(image, lenkernel[2]-1, lenkernel[3]-1)
            elif mode == "valid":
                newimage = image
            else:
                raise ValueError("mode must be 'full' or 'valid'")

            if len(lenkernel) == 4:

                mapa = np.lib.stride_tricks.sliding_window_view(newimage, (lenkernel[2], lenkernel[3]), axis=(1, 2)) #CREER VISION EN WINDOWS IMAGE

                if not reverse: #forward prop
                    output = np.tensordot(mapa, kernel, axes=([0, 3, 4], [1, 2, 3])).transpose((2, 0, 1))
                else: #backprop
                    output = np.tensordot(mapa, kernel, axes=([0, 3, 4], [0, 2, 3])).transpose((2, 0, 1))

            elif len(lenkernel) == 3: #cas ou le kernel a que 3 dimensions au lieu de 4

                mapa = np.lib.stride_tricks.sliding_window_view(newimage, (lenkernel[1], lenkernel[2]), axis=(1, 2))

                output = np.tensordot(mapa, kernel, axes=([3, 4], [1, 2])).transpose(3,0,1,2)

            else:
                raise ValueError("Problem with the shapes they are not good")

            return output

    def convolutionscp(self, image, kernel, *, dimout=None, mode=None, reverse=None):

        lenkernel = kernel.shape  # (sortie, entree, hauteur,largeur)

        if dimout is None:  # calcul dim sortie
            dimout = (lenkernel[0], int((image.shape[1] - self.lenkernel) / self.stride) + 1, int((image.shape[2] - self.lenkernel) / self.stride) + 1)

        output = np.zeros(dimout)

        for d in range(dimout[0]): #parcours
            for ce in range(image.shape[0]):
                output[d] += correlate2d(image[ce], kernel[d,ce], mode="valid")[::self.stride,::self.stride]

        return output

    def poolingnp(self, image):
        division = np.lib.stride_tricks.sliding_window_view(image, (self.lenkernelpool, self.lenkernelpool), axis=(1, 2))[:, ::self.lenkernelpool, ::self.lenkernelpool]
        return np.average(division, axis=(3, 4))

    def poolingskim(self, image):
        d, h, l = image.shape

        if h % self.lenkernelpool == 0:
            newdims = (d, int((h - self.lenkernelpool) / self.lenkernelpool) + 1, int((l - self.lenkernelpool) / self.lenkernelpool) + 1)

            output = np.zeros(newdims)

            for c in range(newdims[0]):
                output[c] += block_reduce(image[c], (self.lenkernelpool, self.lenkernelpool), func=np.mean)

        else:
            newdims = (d, int((h - self.lenkernelpool) / self.lenkernelpool) + 1, int((l - self.lenkernelpool) / self.lenkernelpool) + 1)

            output = np.zeros(newdims)

            for c in range(d):
                output[c] = block_reduce(image[c], (self.lenkernelpool, self.lenkernelpool), func=np.mean)[:newdims[1], :newdims[2]]

        return output

    def forwardprop(self, input): #forward all the layers until output
        outlast = input

        activationsconv = [input] #garder activees pour backprop des convolution
        activationslay = [] #garder activees pour la backprop les variables des layers

        zslay = [] # avant activation
        zsconv = []

        for c in range(self.nbconv): #parcours layers convolution
            kernel = self.parameters["cl" + str(c)]
            biais = self.parameters["cb" + str(c)]

            if self.padding > 0:
                paded = paddington(outlast, self.padding, self.padding)
            else:
                paded = outlast

            conv = self.convolution(paded, kernel) + biais

            if self.parameters["pool" + str(c)]:
                pool = self.pooling(conv)
            else:
                pool = conv

            if c == self.nbconv - 1: #si arrives a la fin flattening layer
                outlast = flatening(pool)
                zsconv.append(outlast)
            else: #sinon continue
                outlast = pool
                zsconv.append(outlast)

            outlast = self.parameters["fctcl" + str(c)][0](outlast)
            activationsconv.append(outlast)

        activationslay.append(activationsconv[-1])

        for l in range(self.nblay):
            w = self.parameters["w" + str(l)]
            b = self.parameters["b" + str(l)]
            z = np.dot(w, outlast) + b

            a = self.parameters["fct" + str(l)](z)

            zslay.append(z)
            activationslay.append(a)
            outlast = a

        return outlast, zslay, zsconv, activationslay, activationsconv #out last c'est la prediction et vieux c'est pour backprop

    def backpoolnp(self, dapres, dimsortie): # REVENIR AUX MEMES DIMENSIONS QU'AVANT POOLING
        moyenne = dapres / (self.lenkernelpool * self.lenkernelpool)

        if dimsortie[1] % self.lenkernelpool == 0: #si pile
            output = np.zeros(dimsortie)

            for d in range(dapres.shape[0]):
                output[d] = np.repeat(np.repeat(moyenne[d], self.lenkernelpool, axis=0), self.lenkernelpool, axis=1) #on recree un kernel avec les dimensions

        else:
            c, h, l = dimsortie

            dif = h % self.lenkernelpool, l % self.lenkernelpool #si pas pile

            newh, newl = h - (dif[0]), l - (dif[1])

            output = np.zeros(dimsortie)

            for d in range(dapres.shape[0]):
                output[d, :newh, :newl] = np.repeat(np.repeat(moyenne[d], self.lenkernelpool, axis=0), self.lenkernelpool, axis=1)

        return output

    def backconvolutionscp(self, activation, dapres, filtre):
        gradc = np.zeros(filtre.shape)

        newdelta = np.zeros(activation.shape)

        for d in range(gradc.shape[0]):
            for c in range(activation.shape[0]):
                gradc[d, c] += correlate2d(activation[c, ::self.stride, ::self.stride], dapres[d], mode="valid")
                newdelta[c] += convolve2d(dapres[d], filtre[d,c], mode="full")

        return gradc, newdelta

    def backconvolutionnp(self, activation, dapres, filtre): # FAIRE LA BACKPROP DE CONVOLUTION
        #pad image pour delta
        #convolution comme avant mais en inversant kernel

        gradc = self.convolution(activation, dapres)

        newdelta = self.convolution(dapres, np.flip(filtre, axis=(2,3)), mode="full", reverse=True)

        return gradc, newdelta

    def backprop(self, expected, zslay, zsconv, activationslay, activationsconv, nbinp, premier):
        C = self.errorfunc[0](activationslay[-1], expected, nbinp) #Calcular error

        #crear los outputs
        dw = [np.zeros(self.dimweights[i]) for i in range(self.nblay)]
        db = [np.zeros((self.dimweights[i][0], 1)) for i in range(self.nblay)]
        dc = [np.zeros((self.convdims[i][3], self.convdims[i][2], self.lenkernel, self.lenkernel)) for i in range(self.nbconv)]
        dcb = [np.zeros(self.parameters["cb" + str(i)].shape) for i in range(self.nbconv)]

        delta = self.errorfunc[1](activationslay[-1], expected, nbinp) #error output layer

        dw[-1] += np.dot(delta, activationslay[-2].T) #dC/dpesos antes de salida
        db[-1] += np.sum(delta, axis=1, keepdims=True) #dC/dbias antes de salida

        for l in range(self.nblay - 2, -1, -1): #parcours layers à l'envers

            w = self.parameters["w" + str(l + 1)]
            dif = self.parameters["diff" + str(l)](zslay[l])

            delta = np.dot(w.T, delta) * dif #update error con error siguiente layer

            dwl = np.dot(delta, activationslay[l].T)
            dbl = np.sum(delta, axis=1, keepdims=True)

            dw[l] += dwl
            db[l] += dbl

        if self.nbconv>0:
            # Calcular ultimo delta para el conv layer
            ultimoweight = self.parameters["w0"]
            ultimadif = self.parameters["fctcl" + str(self.nbconv-1)][1](zsconv[-1])

            s = self.convdims[-1] #dimensiones ultimo conv

            delta = (np.dot(ultimoweight.T, delta) * ultimadif).reshape(s[3], s[1],s[1]) #calcular ultimo error de nn

            if self.parameters["pool" + str(self.nbconv-1)]:
                delta = self.backpoolnp(delta, (s[3], s[0], s[0])) #recuperar misma talla que input de pooling

            gradc, newdelta = self.backconvolution(activationsconv[self.nbconv - 1], delta, self.parameters["cl" + str(self.nbconv - 1)])

            dc[-1] += gradc
            dcb[-1] += delta

            for c in range(self.nbconv - 2, -1, -1):
                diff = self.parameters["fctcl" + str(c)][1](zsconv[c])

                delta = newdelta * diff

                s = self.convdims[c]

                #backpool
                if self.parameters["pool" + str(c)]:
                    delta = self.backpoolnp(delta, (s[3], s[0], s[0]))  # recuperar misma talla que input de pooling

                gradc, newdelta = self.backconvolution(activationsconv[c], delta, self.parameters["cl" + str(c)])

                dc[c] += gradc
                dcb[c] += delta

        return dw, db, C, dc, dcb

    def actualiseweights(self, dw, db, nbinput, dc=None, dcb=None):
        coef = self.cvcoef / nbinput
        eps = 1e-6
        if not self.RMSprop:
            for w in range(max(self.nblay,self.nbconv)):
                if w < self.nblay:
                    self.parameters["w" + str(w)] -= coef * dw[w]
                    self.parameters["b" + str(w)] -= coef * db[w]
                if w < self.nbconv:
                    self.parameters["cl" + str(w)] -= coef * dc[w]
                    self.parameters["cb" + str(w)] -= coef * dcb[w]
        else: #RMSprop
            for w in range(max(self.nblay, self.nbconv)):
                if w < self.nblay:

                    self.moyenw[w] = self.moyennemobile(self.moyenw[w], dw[w])
                    self.parameters["w" + str(w)] -= (coef * dw[w])/ np.sqrt(self.moyenw[w] + eps)

                    self.moyenb[w] = self.moyennemobile(self.moyenb[w], db[w])
                    self.parameters["b" + str(w)] -= (coef * db[w])/ np.sqrt(self.moyenb[w] + eps)

                if w < self.nbconv:

                    self.moyencl[w] = self.moyennemobile(self.moyencl[w], dc[w])
                    self.parameters["cl" + str(w)] -= (coef * dc[w])/ np.sqrt(self.moyencl[w] + eps)

                    self.moyencb[w] = self.moyennemobile(self.moyencb[w], dcb[w])
                    self.parameters["cb" + str(w)] -= (coef * dcb[w])/ np.sqrt(self.moyencb[w] + eps)

        return

    def moyennemobile(self, moyenne, grad): #calculer moyenne mobile pour le learning rate adaptatif
        return self.beta * moyenne + (1 - self.beta) * np.square(np.clip(grad, 1e-150, None))

    def choix(self, y):
        return np.argmax(y,axis=0)

    def vecteur(self, val):
        if self.lenbatch == 1:
            newval = [val]
        else:
            newval = val
        return np.eye(10)[newval].T

    def train(self):
        if self.lenbatch > 1:
            self.trainbatch()
        elif self.lenbatch == 1:
            self.trainsimple()

        if self.tauxerreur()>self.tauxinitial:
            self.exportmodel("BestModels/bestmodel" + self.base)
        return

    def trainsimple(self):
        if self.nbconv == 0:
            C = []
            for _ in range(self.iter):
                L = []
                for p in tqdm(range(self.pix.shape[1])):
                    forw = self.forwardprop(self.pix[:,p].reshape(-1,1))

                    dw, db, loss, dc, dcb = self.backprop(self.vecteur(self.vales[p]), forw[1], forw[2], forw[3], forw[4], 1, p==0)

                    self.actualiseweights(dw, db, 1, dc, dcb)

                    L.append(loss)

                C.append(np.average(L))

        else:
            ecart = 5000 if len(self.pix) < 10000 else 15000
            C = []
            for i in range(self.iter):
                L = []
                for p in tqdm(range(len(self.pix))):
                    forw = self.forwardprop(self.pix[p]) #canaux, h,l

                    dw, db, loss, dc, dcb = self.backprop(self.vecteur(self.vales[p]), forw[1], forw[2], forw[3], forw[4], 1, p == 0)

                    self.actualiseweights(dw, db, 1, dc, dcb)

                    L.append(loss)
                C.append(np.average(L))

                if self.tauxfiniter:
                    print("___________________________________________________________________________________________________________")
                    print(f"Le taux à l'itération {i} est de {self.tauxlent()}")

        return

    def trainbatch(self):
        if self.nbconv == 0:
            for _ in range(self.iter):
                nbbatch = self.pix.shape[1] // self.lenbatch
                for bat in range(nbbatch):
                    matrice = self.pix[:, bat*self.lenbatch:(bat+1)*self.lenbatch].reshape(-1, self.lenbatch)

                    forw = self.forwardprop(matrice)

                    dw, db, loss, dc, dcb = self.backprop(self.vecteur(self.vales[bat*self.lenbatch:(bat+1)*self.lenbatch]), forw[1], forw[2], forw[3], forw[4], self.lenbatch, bat==0)

                    self.actualiseweights(dw, db, self.lenbatch)

        else:
            print("EN TRAVAUX")

        return

    def tauxlent(self): #go in all the test and see accuracy (used for CNN)
        nbbien = 0
        for image in range(len(self.qcmpix)):
            forw = self.forwardprop(self.qcmpix[image])

            observed = self.choix(forw[0])

            if observed == self.qcmval[image]:
                nbbien += 1

        return nbbien * 100 / len(self.qcmpix)

    def tauxrapide(self): #used only for NN for now
        if self.nbconv == 0:
            forw = self.forwardprop(self.qcmpix.reshape(784,-1))

            observed = self.choix(forw[0])

            difference = observed - self.qcmval

            nbbien = np.count_nonzero(difference==0)

            return nbbien*100 / self.qcmpix.shape[1]
        else:
            print("EN TRAVAUX")
            return

    def tauxerreur(self):
       if self.nbconv == 0:
           t = self.tauxrapide()
           print(t)
       else:
           t = self.tauxlent()
           print(t)
       return t

    def prediction(self, image):
        printgray(image, "")
        forw = self.forwardprop(image)
        decision = self.choix(forw[0])

        print(f"Je crois bien que cela est un {self.labels[decision[0]]}")

        try:
            verdadero = input("Que es realmente? : ")
        except:
            print("NO has puesto el buen type")
            verdadero = None

        if verdadero is not None and verdadero != decision:
            if self.base == "mnist":
                print("Vale intentare mejorar para la proxima vez")
                dw, db, _, dc, dcb = self.backprop(self.vecteur(int(verdadero)), forw[1], forw[2], forw[3], forw[4], 1, False)

                self.actualiseweights(dw, db, 1, dc, dcb)
            else:
                pass

        return

    def importmodel(self, namefile):
        with open(namefile, "rb") as file:
            dico = pickle.load(file)

        self.nbconv = dico["nbconv"]
        self.nblay = dico["nblay"]

        self.parameters = dico["parameters"]

        for c in range(self.nbconv):
            self.parameters["fctcl" + str(c)] = getfct(dico["convlay"][c+1][1], self.cvcoef)
        for l in range(self.nblay):
            self.parameters["fct" + str(l)] = getfct(dico["lay"][l+1][1], self.cvcoef)[0]
            self.parameters["diff" + str(l)] = getfct(dico["lay"][l+1][1], self.cvcoef)[1]

        self.convdims = dico["convdims"]
        self.dimweights = dico["dimweights"]
        self.lenkernel = dico["lenkernel"]
        self.padding = dico["padding"]
        self.stride = dico["stride"]
        self.poolstride = dico["poolstride"]
        self.lenkernelpool = dico["lenkernelpool"]
        self.lenbatch = dico["lenbatch"]
        self.pix = dico["pix"]
        self.vales = dico["vales"]
        self.qcmpix = dico["qcmpix"]
        self.qcmval = dico["qcmval"]
        self.dimbiais = dico["dimbiais"]
        self.convlay = dico["convlay"]
        self.lay = dico["lay"]
        self.base = dico["base"]
        # self.tauxinitial = dico["tauxinitial"]

    def exportmodel(self, namefile):
        # POUR ENREGISTRER LES PARAMETRES
        dico = self.parameters.copy()

        for c in range(self.nbconv):
            dico["fctcl" + str(c)] = self.convlay[c+1][1]
        for l in range(self.nblay):
            dico["fct" + str(l)] = self.lay[l][1]
            dico["diff" + str(l)] = self.lay[l][1]

        tab = {"parameters": dico, "nbconv": self.nbconv, "nblay": self.nblay, "convdims": self.convdims, "dimweights": self.dimweights, "lenkernel": self.lenkernel,
               "padding": self.padding, "stride": self.stride, "poolstride": self.poolstride, "lenkernelpool": self.lenkernelpool,
               "lenbatch": self.lenbatch, "pix": self.pix, "vales": self.vales, "qcmpix": self.qcmpix, "qcmval": self.qcmval, "dimbiais": self.dimbiais,
               "convlay": self.convlay, "lay": self.lay, "base": self.base, "tauxinitial": self.tauxinitial}

        with open(namefile, "wb") as file:
            pickle.dump(tab, file)

    def TryToDraw(self):
        cnv = Draw()

        px = cnv.pixels

        if self.nbconv > 0:
            self.prediction(px)
        else:
            self.prediction(px.reshape(-1, 1))

    def graphisme(self):
        fct = []
        for _ in range(self.iter):
            self.trainsimple()
            a = self.tauxlent()
            fct.append(a)
        plt.plot([i for i in range(len(fct))], fct)
        plt.xlabel('Iteration')
        plt.ylabel('Taux erreur')
        plt.title('Fonction de Erreur')
        plt.show()