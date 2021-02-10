import numpy as np
from random import random
import matplotlib.pyplot as plt
n = 100

def Betas_estimation( X , Y ):
    Betas = np.dot( np.dot( np.linalg.inv( np.dot(X.T , X) ) , X.T ) , Y )
    l = len(Betas)
    list = []
    for i in range(0,l):
        list.append(Betas[i,i])

    return(list)


mu1, sigma1 = 175 , 10
mu2, sigma2 = 2000 , 1000
mu3, sigma3 = 10 , 4
mu4, sigma4 = 35 , 20
taille = mu1 + sigma1 * np.random.randn(n)
salary  = mu2 + sigma2 * np.random.randn(n)
note = mu3 + sigma3 * np.random.randn(n)
X = np.c_[ taille , salary , note ]
beta_diag = np.array([1,20,3]) 
beta = np.diag(beta_diag)
Y = np.dot(X,beta)
print(Betas_estimation(X,Y))
print(len(X))