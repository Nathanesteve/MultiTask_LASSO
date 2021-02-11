import numpy as np
from random import random
import matplotlib.pyplot as plt

n_samples = 100  #Taille éechatillon

def Beta_estimation( X , Y ): #Fonction calculant l'OLS.
    
    beta_hat = np.dot( np.dot( np.linalg.inv( np.dot(X.T , X) ) , X.T ) , Y )
    return beta_hat 


#Simulation                                        

mu1 , sigma1 = 175 , 10                   
mu2 , sigma2 = 2000 , 1000   
mu3 , sigma3 = 10 , 4

taille = mu1 + sigma1 * np.random.randn(n_samples)
salary  = mu2 + sigma2 * np.random.randn(n_samples)
note = mu3 + sigma3 * np.random.randn(n_samples)

eps = np.random.randn(n_samples)  #Bruit

X = np.c_[ taille , salary , note ] #Variables explicatives

beta = np.array([1,20,3])  #Vrai beta

Y =  np.dot(X,beta) + eps  #Vrai modèle (variable à expliquer) 

print(Beta_estimation(X,Y)) #Estimateur OLS