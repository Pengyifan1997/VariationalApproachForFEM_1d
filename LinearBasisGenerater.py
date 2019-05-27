import numpy as np 

# right side data
# the number of the test functions
M = np.array((2000))

# the max demension of the test function
D = np.array((40))

# solve the equation
# the coarse grid
Nc = np.array((63))

# refinement times
K = np.array((4))

# the fine grid
Nf = K*(Nc + 1) - 1

# to calculate the i-th basis function on coarse grid
T = np.zeros((Nf,Nc))
for i in range(0,Nc):
    for j in range(0,K - 1):
        T[i*K + j,i] = (j + 1)/K
        T[(i + 2)*K - j - 2,i] = (j + 1)/K
    T[(i + 1)*K - 1,i] = 1

np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/LinearBasis/LinearBasis.txt'%(M,D,Nc,K),T)