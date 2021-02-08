import numpy as np
import matplotlib.pyplot as plt
from math import factorial

c = 0
for i in range(100):
    c = c + 1/(factorial(i))+10

print(c)
