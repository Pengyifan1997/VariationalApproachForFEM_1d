import numpy as np 
import sympy as syp
import scipy.linalg as spl 
import matplotlib.pyplot as plt 
import pdb

x = syp.symbols("x")

def faster_simpson(f, a, b, steps):
   h = (b-a)/float(steps)
   a1 = a+h/2
   s1 = sum( f.evalf(subs={x:a1+i*h}) for i in range(0,steps))
   s2 = sum( f.evalf(subs={x:a+i*h}) for i in range(1,steps))
   return (h/6.0)*(f.evalf(subs={x:a})+f.evalf(subs={x:b})+4.0*s1+2.0*s2)

# the equation is: - divergence(a*gradient(u)) = f
# left side assumption
epsilon = 0.0013
a = 2 + syp.cos(x/epsilon)

# right side data
# the number of the test functions
M = np.array((500))

# the max demension of the test function
D = np.array((40))

N = np.array((5))

# generate their coefficients
u = syp.sin(7*np.pi*x)

f1 = a*syp.diff(u,x,1)
f2 = - syp.diff(f1,x,1)

loss = np.zeros((N,2))

for i in range(0,N):

    Nc = np.array((np.power(2,i + 2) - 1))
    K = np.array((4))
    Nf = K*(Nc + 1) - 1
    I = 512/(Nf + 1)

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
    L = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/stiffnessmatrix_%d_%d.txt'%(M,D,Nc,K,Nc,K))

    # transform matrix
    T = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/LinearBasis/transformmatrix.txt'%(M,D,Nc,K))

    # the coefficinet matrix of right side
    F = np.zeros((Nf,1))
    for j in range(0,Nf):
        mid1 = np.zeros(1)
        for k in range(j,j + 2):
            mid2 = basis(j,k)*f2
            mid3 = faster_simpson(mid2,k/(Nf + 1),(k + 1)/(Nf + 1),I.astype(int))
            mid1 = mid3 + mid1
        F[j,0] = mid1
    
    xc = spl.solve(np.dot(np.dot(np.transpose(T),L),T),np.dot(np.transpose(T),F))
    model = np.transpose(np.dot(T,xc))

    mid4 = u - (Nf + 1)*model[0,0]*x
    loss[i,0] = faster_simpson(mid4*mid4,0,1/(Nf + 1),I.astype(int))
    for j in range(1,Nf):
        mid4 = u - (Nf + 1)*(model[0,j] - model[0,j - 1])*(x - j/(Nf + 1)) - model[0,j - 1]
        loss[i,0] = loss[i,0] + faster_simpson(mid4*mid4,j/(Nf + 1),(j + 1)/(Nf + 1),I.astype(int))
    mid4 = u + (Nf + 1)*model[0,Nf - 1]*(x - Nf/(Nf + 1)) - model[0,Nf - 1]
    loss[i,0] = loss[i,0] + faster_simpson(mid4*mid4,Nf/(Nf + 1),1,I.astype(int))
    loss[i,0] = np.sqrt(loss[i,0])

    def g(x):
        if 0 <= x < 1/(Nf + 1):
            z = model[0,0]*(Nf + 1)*x
        elif 1/(Nf + 1) <= x < Nf/(Nf + 1):
            for k in range(1,Nf):
                if k/(Nf + 1) <= x < (k + 1)/(Nf + 1):
                    z = model[0,k - 1]*(-(Nf + 1)*x + (k + 1)) + model[0,k]*((Nf + 1)*x - k)
        elif Nf/(Nf + 1) <= x < 1:
            z = model[0,Nf - 1]*(-(Nf + 1)*x + (Nf + 1))
        else:
            z = 0
        return z

    xx1=[]
    yy1=[]

    a=np.linspace(0,1,10*(Nf + 1) - 1)
    for jj in a:
        xx1.append(jj)
        yy1.append(g(jj))

    # transform matrix
    T = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/LinearBasis/LinearBasis.txt'%(M,D,Nc,K))
    
    xc = spl.solve(np.dot(np.dot(np.transpose(T),L),T),np.dot(np.transpose(T),F))
    model = np.transpose(np.dot(T,xc))

    mid4 = u - (Nf + 1)*model[0,0]*x
    loss[i,1] = faster_simpson(mid4*mid4,0,1/(Nf + 1),I.astype(int))
    for j in range(1,Nf):
        mid4 = u - (Nf + 1)*(model[0,j] - model[0,j - 1])*(x - j/(Nf + 1)) - model[0,j - 1]
        loss[i,1] = loss[i,1] + faster_simpson(mid4*mid4,j/(Nf + 1),(j + 1)/(Nf + 1),I.astype(int))
    mid4 = u + (Nf + 1)*model[0,Nf - 1]*(x - Nf/(Nf + 1)) - model[0,Nf - 1]
    loss[i,1] = loss[i,1] + faster_simpson(mid4*mid4,Nf/(Nf + 1),1,I.astype(int))
    loss[i,1] = np.sqrt(loss[i,1])

    def g(x):
        if 0 <= x < 1/(Nf + 1):
            z = model[0,0]*(Nf + 1)*x
        elif 1/(Nf + 1) <= x < Nf/(Nf + 1):
            for k in range(1,Nf):
                if k/(Nf + 1) <= x < (k + 1)/(Nf + 1):
                    z = model[0,k - 1]*(-(Nf + 1)*x + (k + 1)) + model[0,k]*((Nf + 1)*x - k)
        elif Nf/(Nf + 1) <= x < 1:
            z = model[0,Nf - 1]*(-(Nf + 1)*x + (Nf + 1))
        else:
            z = 0
        return z

    xx2=[]
    yy2=[]

    a=np.linspace(0,1,10*(Nf + 1) - 1)
    for jj in a:
        xx2.append(jj)
        yy2.append(g(jj))

    xx3=[]
    yy3=[]

    a=np.linspace(0,1,10*(Nf + 1) - 1)
    for jj in a:
        xx3.append(jj)
        yy3.append(u.evalf(subs={x:jj}))

    plt.figure(i)

    plt.title('Result Analysis')
    plt.plot(xx1,yy1, color='green', label='function by result basis')
    plt.plot(xx2,yy2, color='red', label='function by original basis')
    plt.plot(xx3,yy3, color='blue', label='original function')
    plt.legend()

    plt.savefig('/Users/pengyifan/Desktop/1d_vaob/functionimage/analysis_%d.png'%(i))

plt.show()



np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/Linearloss.txt'%(M,D),loss)