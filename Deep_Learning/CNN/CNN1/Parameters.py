import numpy as np
from dataclasses import dataclass


@dataclass
class Parametros:
    # DATASET
    base: str
    pix : list or np.ndarray # type: ignore
    vales : np.ndarray
    qcmpix: list or np.ndarray # type: ignore
    qcmval: np.ndarray
    labels: dict

    # LISTES DES COUCHES
    infolay: list #LISTE AVEC (NOMBRE DE NEURONES, FCT ACTIVATION)
    infoconvlay: list # (NOMBRE DE FILTRES, FCT ACTIVATION, BOOL TRUE SI POOLING APRES CONVOLUTION)

    # PRENDRE LES PARAMETRES D'UN MODELE DEJA ENTRAINE
    modeldejaentraine: bool = False

    iterations: int = 5 # NOMBRE ITERATIONS SUR LES DONNEES DE TRAIN
    coefcv: float = 0.1 # LEARNING RATE
    batch: int = 1 # LONGUEUR BATCH (SEULEMENT POUR MULTICOUCHE POUR L'INSTANT
    errorfunc: str = "CEL" # FCT DE COUT

    apprentissagedynamique: bool = False # ESTCE QU'IL APPREND SUR LES TESTS PENDANT LE TEST
    tauxfiniter: bool = False # ESTCE QU'IL FAIT LE QCM A LA FIN DE CHAQUE ITERATION

    #CNN
    kernel: int = 3 # DIMENSION KERNEL
    kernelpool: int = 2 # DIMENSION FILTRE DE POOLING
    padding: int = 0 # RESTE A ZERO POUR L'INSTANT
    stride: int = 1 # RESTE A 1 POUR L'INSTANT

    # LAISSER A TRUE CEST LES FONCTIONS PLUS OPTIMISÉES
    poolnp: bool = True
    convnp: bool = True
    backconvnp: bool = True

    # ADAPTATIVE AVANCÉ
    RMSprop: bool = False # BASÉ SUR LES MOYENNE MOBILE DES COUTS MAIS NE MARCHE PAS ENCORE
    beta: float = 0.9