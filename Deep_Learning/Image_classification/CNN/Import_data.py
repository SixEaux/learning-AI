import pickle
import numpy as np
from Helpers import converttogreyscale

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

        labels_mnist = {i:i for i in range(10)}

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
        with open('Datas/Ciphar-10/pixentraine_ciphar', 'rb') as f:
            ciphar_pixentraine = pickle.load(f)
        with open("Datas/Ciphar-10/valentraine_ciphar", "rb") as f:
            ciphar_valentraine = pickle.load(f)
        with open("Datas/Ciphar-10/pixtest_ciphar", "rb") as f:
            ciphar_pixtest = pickle.load(f)
        with open("Datas/Ciphar-10/valtest_ciphar", "rb") as f:
            ciphar_valtest = pickle.load(f)

        labels_ciphar = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

        return ciphar_valentraine, ciphar_pixentraine, ciphar_valtest, ciphar_pixtest, labels_ciphar

    else:
        raise ValueError("Pas de base de données dispo!")


def processdata(base, pix, qcm, conv): 
    """transform data to a good form for the NN depending on if it is for test, which data used and if there is convolution or not
    (i need this due to a not so good transformation of the data when imported needs fixing)"""
    if conv:
        if base == "mnist":
            if qcm:
                datamod = [pix[:, a].reshape(1, 28, 28) for a in range(pix.shape[1])]
            else:
                datamod = [pix[:, a].reshape(1, 28, 28) / 255 for a in range(pix.shape[1])]

        elif base == "fashion":
            if qcm:
                datamod = [pix[:, a].reshape(1, 28, 28) for a in range(pix.shape[1])]
            else:
                datamod = [pix[:, a].reshape(1, 28, 28) / 255 for a in range(pix.shape[1])]

        elif base == "ciphar-10":
            datamod = pix

        else:
            print("Je n'ai pas encore fait cela") #IL FAUT ICI AJOUTER POUR DETECTER TYPE INPUT
            raise TypeError

    else:
        if base == "mnist":
            if qcm:
                datamod = pix
            else:
                datamod = pix / 255

        elif base == "fashion":
            if qcm:
                datamod = pix
            else:
                datamod = pix / 255

        elif base == "ciphar-10":
            datamod = [converttogreyscale(i).reshape(-1, 1) for i in pix] #NO ESTA BIEN AUN ESTE CREO
            final = np.array(datamod[0])
            for i in range(1, len(datamod)):
                final = np.concatenate([final, datamod[i]], axes=1)

            datamod = final

        else:
            print("Je n'ai pas encore fait cela")  # IL FAUT ICI AJOUTER POUR DETECTER TYPE INPUT
            raise TypeError

    return datamod
