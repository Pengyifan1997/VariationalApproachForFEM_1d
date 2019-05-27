import numpy as np
import random as rd

# right side data
# the number of the test functions
M = np.array((2000))

# the max demension of the test function
D = np.array((80))

# generate the test functions
# here we use Fourier series
# generate their demensions
d = np.zeros((M,2))
for i in range(0,M):
    # demension of sin mod
    d[i,0] = rd.randint(1,D)
    # demension of cos mod
    d[i,1] = rd.randint(1,D)

# generate their coefficients
c = np.zeros((M,2*D + 1))
for i in range(0,M):
    # coefficients of sin mod
    for j in range(0,D):
        if j < d[i,0]:
            c[i,j] = 2*rd.random() - 1
    # coefficients of cos mod
    for j in range(D,2*D):
        if j - 10 < d[i,1]:
            c[i,j] = 2*rd.random() - 1
    # constant coefficient
    c[i,2*D] = 2*rd.random() - 1

np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/testfunction_%d_%d.txt'%(M,D,M,D),c)