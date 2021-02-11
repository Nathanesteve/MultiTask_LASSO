from random import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123456789)
n = 100
z = 123 #add this line

def uniforme( a , b ): # function uniforme sur [a,b]
    return( a + (b -a)*random())


k = np.zeros(n)
u = np.zeros(n)

for i in range(0,n): # Init 1/n colum vector 
    k[i] = 1/n
for i in range(0,n): # x_i
    u[i] = uniforme(-2,2)

Beta1 = -2
Beta2 = 3
mu, sigma = 0 , 4
eps = mu + sigma * np.random.randn(n) # generation of n gaussian(0,1)

y = Beta1 + Beta2 * u + eps


plt.plot(u, y,'o')
plt.plot(u,Beta1+Beta2*u,label="True regression")

u_barre = np.dot(u,k)
y_barre = np.dot(y,k)

sto1 = np.zeros(n)
sto2 = np.zeros(n)

for i in range(0,n):
    sto1[i] = (u[i]-u_barre)*(y[i]-y_barre)
    sto2[i] = (u[i]-u_barre)*(u[i]-u_barre)

Beta2_hat = sum(sto1)/sum(sto2)
Beta1_hat = y_barre - Beta2_hat*u_barre
print(Beta2_hat) 
print(Beta1_hat)
plt.plot(u,Beta1_hat+Beta2_hat*u,'b',label="estimation")
plt.legend()
plt.show()
