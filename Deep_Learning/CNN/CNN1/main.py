from Deep_Learning.CNN.CNN1.Import_data import takeinputs
from Parameters import Parametros
from CNN import CNN
import time

base = "mnist"
inputs = takeinputs(base) #"mnist" #"fashion" #ciphar-10

val, pix, qcmval, qcmpix, labels = inputs

convlay = [(10, "relu", True)]

lay = [(32, "sigmoid"), (10, "softmax")]

parametros = Parametros(pix=pix, vales=val, qcmpix=qcmpix, qcmval=qcmval, labels=labels,
                        infolay=lay, infoconvlay=convlay, iterations=1, coefcv=0.01, base=base)

g = CNN(parametros)

# g.train()
#
# printgray(g.pix[10])
#
# g.tauxerreur()

# MODEL ENTRAINÉ

# g.importmodel("BestModels/bestmodelmnist")
#
# t0 = g.tauxerreur()
#
# print(g.base)
#
# for i in range(10):
#     g.TryToDraw()
#
# t = g.tauxerreur()
#
# if t >= t0:
#     print("ME HE SUPERADO MUCHO!!!!")
#     g.exportmodel("BestModels/bestmodelmnist")


# MODELE A ENTRAINÉ

print("je commence a mentrainer")
t = time.time()

g.train()

print("jai fini en :", time.time()-t)
g.tauxerreur()

