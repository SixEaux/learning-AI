import numpy as np
from scipy.special import expit

def geterrorfunc(errorfunc): 
    """get error function with EQM and CEL to choose from"""
    if errorfunc == "EQM": #quadratic mean error
        def eqm(obs, exp, nbinput):
            return (np.sum((obs - exp) ** 2, axis=0))/ (2 * nbinput)
        def eqmdif(obs, expected, nbinput):
            return  (obs - expected)/nbinput
        return [eqm, eqmdif]

    elif errorfunc == "CEL": #Categorical Cross-Entropy (better for classification)
        def CEL(obs, exp, nbinput):
            return -np.sum(exp * np.log(np.clip(obs, 1e-9, 1 - 1e-9)), axis=0) / nbinput
        def CELdif(obs, exp, nbinput):
            return (obs - exp) / nbinput
        return [CEL, CELdif]

    else:
        raise ValueError("errorfunc must be specified")

def getfct(acti, cvcoef):
    """get activation function from choice"""
    if acti == 'sigmoid':
        def sigmoid(x):
            return expit(x)
        def sigmoiddif(x):
            return (expit(x)) * (1 - expit(x))
        return [sigmoid, sigmoiddif]

    elif acti == 'relu':
        def relu(x):
            return np.maximum(x, 0)
        def reludif(x):
            return np.where(x >= 0, 1, 0)
        return [relu, reludif]

    elif acti == 'tanh':
        def tan(x):
            return np.tanh(x)
        def tandiff(x):
            return 1 - np.square(np.tanh(x))
        return [tan, tandiff]

    elif acti == 'softmaxaprox':
        def softmaxaprox(x):
            x = x - np.max(x, axis=0, keepdims=True)
            return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

        def softmaxaproxdif(output):
            return output * (1 - output)

        return [softmaxaprox, softmaxaproxdif]

    elif acti == 'softmax':
        def softmax(x):
            x = x - np.max(x, axis=0, keepdims=True)
            return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

        def softmaxdif(output):
            n = output.shape[0]
            jacob = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        jacob[i, j] = output[i] * (1 - output[i])
                    else:
                        jacob[i, j] = -output[i] * output[j]

            return jacob

        return [softmax, softmaxdif]

    elif acti == "leakyrelu":
        def leakyrelu(x):
            return np.maximum(cvcoef * x, 0)

        def leakyreludif(x):
            return np.where(x > 0, cvcoef, 0)

        return [leakyrelu, leakyreludif]

    else:
        raise "You forgot to specify the activation function"