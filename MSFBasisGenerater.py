import numpy as np 
import sympy as syp
import scipy.linalg as spl 


x = syp.symbols("x")

# the equation is: - divergence(a*gradient(u)) = 0
# for each u : u(xi) = 1 , u(xj) = 0 for j doesn't equal to i
# left side assumption
epsilon = 0.0013
a = 2 + syp.cos(x/epsilon)

# right side data
# the number of the test functions
M = np.array((2000))

# the max demension of the test function
D = np.array((40))

# the coarse grid
Nc = np.array((63))

# refinement times
# one coarse grid is divided into K fine grids
K = np.array((4))

# the fine grid
Nf = K*(Nc + 1) - 1

# the basis on the fine grid are simple piece-wise linear functions
def basis(number,interval):
    for i in range(0,Nf):
        if number == i:
            if interval == i:
                f = (Nf + 1)*x - i 
            elif interval == i + 1:
                f = -(Nf + 1)*x + (i + 2)
            else:
                f = 0
    return f

# to calculate the i-th basis function on coarse grid
T = np.zeros((Nf,Nc))
for i in range(0,Nc):
    L = np.zeros((K - 1,K - 1))
    l = np.zeros((K - 1,1))
    for j in range(0,K - 1):
        mid1 = a*syp.diff(basis(i*K + j,(i + 1)*K - 1),x,1)*syp.diff(basis((i + 1)*K - 1,(i + 1)*K - 1),x,1)
        l[j,0] = 0 - syp.integrate(mid1,(x,((i + 1)*K - 1)/(Nf + 1),(i + 1)*K/(Nf + 1)))
        for k in range(0,K - 1):
            mid2 = a*syp.diff(basis(i*K + j,i*K + j),x,1)*syp.diff(basis(i*K + k,i*K + j),x,1)
            mid3 = a*syp.diff(basis(i*K + j,i*K + j + 1),x,1)*syp.diff(basis(i*K + k,i*K + j + 1),x,1)
            L[j,k] = syp.integrate(mid2,(x,(i*K + j - 1)/(Nf + 1),(i*K + j)/(Nf + 1))) + syp.integrate(mid3,(x,(i*K + j)/(Nf + 1),(i*K + j + 1)/(Nf + 1)))
    R = np.zeros((K - 1,K - 1))
    r = np.zeros((K - 1,1))
    for j in range(0,K - 1):
        mid1 = a*syp.diff(basis((i + 1)*K + j,(i + 1)*K),x,1)*syp.diff(basis((i + 1)*K - 1,(i + 1)*K),x,1)
        r[j,0] = 0 - syp.integrate(mid1,(x,(i + 1)*K/(Nf + 1),((i + 1)*K + 1)/(Nf + 1)))
        for k in range(0,K - 1):
            mid2 = a*syp.diff(basis((i + 1)*K + j,(i + 1)*K + j),x,1)*syp.diff(basis((i + 1)*K + k,(i + 1)*K + j),x,1)
            mid3 = a*syp.diff(basis((i + 1)*K + j,(i + 1)*K + j + 1),x,1)*syp.diff(basis((i + 1)*K + k,(i + 1)*K + j + 1),x,1)
            R[j,k] = syp.integrate(mid2,(x,((i + 1)*K + j - 1)/(Nf + 1),((i + 1)*K + j)/(Nf + 1))) + syp.integrate(mid3,(x,((i + 1)*K + j)/(Nf + 1),((i + 1)*K + j + 1)/(Nf + 1)))
    Left = spl.solve(L,l)
    Right = spl.solve(R,r)
    for j in range(0,K - 1):
        T[i*K + j,i] = Left[j,0]
    T[(i + 1)*K - 1,i] = 1
    for j in range(0,K - 1):
        T[(i + 1)*K + j,i] = Right[j,0]

np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/MSFBasis/MSFBasis.txt'%(M,D,Nc,K),T)