import sympy as syp 
import numpy as np 
import scipy.linalg as spl 

x = syp.symbols("x")

def faster_simpson(f, a, b, steps):
   h = (b-a)/float(steps)
   a1 = a+h/2
   s1 = sum( f.evalf(subs={x:a1+i*h}) for i in range(0,steps))
   s2 = sum( f.evalf(subs={x:a+i*h}) for i in range(1,steps))
   return (h/6.0)*(f.evalf(subs={x:a})+f.evalf(subs={x:b})+4.0*s1+2.0*s2)

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

# solve the equation
# the coarse grid
Nc = np.array((63))

# refinement times
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

# stiffness matrix on the fine grid
L = np.zeros((Nf,Nf))
for i in range(0,Nf):
    for j in range(0,Nf):
        mid1 = np.zeros(1)
        for k in range(i,i + 2):
            mid2 = a*syp.diff(basis(i,k),x,1)*syp.diff(basis(j,k),x,1)
            mid3 = faster_simpson(mid2,k/(Nf + 1),(k + 1)/(Nf + 1),2)
            mid1 = mid3 + mid1
        L[i,j] = mid1

# the coefficinet matrix of right side
F = np.zeros((Nf,2*D + 1))
for i in range(0,Nf):
    for j in range(0,D):
        mid4 = np.zeros(1)
        for k in range(i,i + 2):
            mid5 = basis(i,k)*syp.sin((j + 1)*np.pi*x)
            mid6 = faster_simpson(mid5,k/(Nf + 1),(k + 1)/(Nf + 1),2)
            mid4 = mid6 + mid4
        F[i,j] = mid4
    for j in range(D,2*D):
        mid4 = np.zeros(1)
        for k in range(i,i + 2):
            mid5 = basis(i,k)*syp.cos((j - 9)*np.pi*x)
            mid6 = faster_simpson(mid5,k/(Nf + 1),(k + 1)/(Nf + 1),2)
            mid4 = mid6 + mid4
        F[i,j] = mid4
    mid4 = np.zeros(1)
    for k in range(i,i + 2):
        mid5 = basis(i,k)
        mid6 = faster_simpson(mid5,k/(Nf + 1),(k + 1)/(Nf + 1),2)
        mid4 = mid6 + mid4
    F[i,2*D] = mid4

c = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/testfunction_%d_%d.txt'%(M,D,M,D))

# the numerical solution on the fine grid
f = np.dot(F,np.transpose(c))
xf = np.transpose(spl.solve(L,f))

np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/stiffnessmatrix_%d_%d.txt'%(M,D,Nc,K,Nc,K),L)
np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/coefficinetmatrix_%d_%d.txt'%(M,D,Nc,K,Nc,K),F)
np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/numsolution_%d_%d.txt'%(M,D,Nc,K,Nc,K),xf)
