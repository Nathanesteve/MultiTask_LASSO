import numpy as np
from random import random
import matplotlib.pyplot as plt


n_samples = 100  #Taille échantillon
np.random.seed(12312312)

def Beta_estimation( X , y ): #Fonction calculant l'OLS.
    
    beta_hat = np.dot( np.dot( np.linalg.inv( np.dot(X.T , X) ) , X.T ) , y )  #Utiliser @ pour le produit
    return beta_hat 



#Simulation                                        

mu1 , sigma1 = 175 , 10                   
mu2 , sigma2 = 2000 , 1000   
mu3 , sigma3 = 10 , 4

taille = mu1 + sigma1 * np.random.randn(n_samples)
salary  = mu2 + sigma2 * np.random.randn(n_samples)
note = mu3 + sigma3 * np.random.randn(n_samples)

eps = np.random.randn(n_samples)  #Bruit

X = np.c_[ taille , salary , note ] #Variables explicatives  #hstack stack vstack

beta = np.array([1,20,3])  #Vrai beta

y =  np.dot(X,beta) + eps  #Vrai modèle (variable à expliquer) 

#print(Beta_estimation(X,y)) #Estimateur OLS

#----------------------------------------------------

#CD moindres carrés version 1 (non fonctionnelle car divergence des parametres à estimer)


#Initialisation

nb_iter = 100

B = np.zeros((3,nb_iter))  #Beta_hat
r = np.zeros((n_samples,nb_iter)) #Reste 
r[:,0] = y # k = 0
r_int = np.zeros((n_samples,)) #
r_int[:] = y # k = 0


def MCO(X,y):

    for k in range(1,nb_iter): # 100 itérations
        
        for j in range(0,3):  # 3 variables explicatives

            r_int[:]  = r[:,k-1] + X[:,j]*B[j,k-1]
            B[j,k] = (X[:,j].T@r_int)/(np.linalg.norm(X[:,j])**2)
            r[:,k] = r_int - (X[:,j]*B[j,k])
    
        r[:,k] = y - X@B[:,k] #On stocke le reste: r = y - X*Beta_hat


    return B


MCO(X,y)
print(B[:,nb_iter - 1])  # Affiche les parametres à estimer (Ne soyez pas effrayé par le résultat...)


#----------------------------------------

#CD moindres carrés version 2 (fonctionnelle)


B1 = np.zeros((3,)) #Beta_hat
r1 = np.zeros((n_samples,))  #Reste 
r1_int = np.zeros((n_samples,))

def MCO1(X,y):

    r1 = y - X@B1[:]
    
    for k in range(0,nb_iter): # 100 itérations

        for j in range(0,3):  # 3 variables explicatives

            r1_int[:]  = r1[:] + X[:,j]*B1[j]
            B1[j] = (X[:,j].T@r1_int)/(np.linalg.norm(X[:,j])**2)
            r1[:] = r1_int - (X[:,j]*B1[j])
            
        r1 = y - X@B1[:] #On stocke le reste: r = y - X*Beta_hat

    return B1

print(MCO1(X,y)) #Affiche parametres à estimer (On trouve bien une approximation du vecteur [1,20,3])

#Reamarque: Ici p = 3 mais nous pouvons facilement le généraliser à p quelconque