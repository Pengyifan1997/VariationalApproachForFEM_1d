import numpy as np 
import matplotlib.pyplot as plt 

# right side data
# the number of the test functions
M = np.array((1000))

# the max demension of the test function
D = np.array((40))

# solve the equation
# the coarse grid
Nc = np.array((7))

# refinement times
K = np.array((4))

# the fine grid
Nf = K*(Nc + 1) - 1

#curr_T = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/LinearBasis/LinearBasis.txt'%(M,D,Nc,K))
curr_T1 = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_10/grid_%d_%d/LinearBasis/transformmatrix.txt'%(M,Nc,K))
curr_T2 = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_20/grid_%d_%d/LinearBasis/transformmatrix.txt'%(M,Nc,K))
curr_T3 = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_40/grid_%d_%d/LinearBasis/transformmatrix.txt'%(M,Nc,K))
#curr_T = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/VariationalBasis/VariationalBasis.txt'%(M,D,Nc,K))


for i in range(1,Nc + 1):
    def g(x):
        if 0 <= x < 1/(Nf + 1):
            z = curr_T1[0,i - 1]*(Nf + 1)*x
        elif 1/(Nf + 1) <= x < Nf/(Nf + 1):
            for k in range(1,Nf):
                if k/(Nf + 1) <= x < (k + 1)/(Nf + 1):
                    z = curr_T1[k - 1,i - 1]*(-(Nf + 1)*x + (k + 1)) + curr_T1[k,i - 1]*((Nf + 1)*x - k)
        elif Nf/(Nf + 1) <= x < 1:
            z = curr_T1[Nf - 1,i - 1]*(-(Nf + 1)*x + (Nf + 1))
        else:
            z = 0
        return z

    x1=[]
    y1=[]

    a=np.linspace(0,1,10*(Nf + 1) - 1)
    for j in a:
        x1.append(j)
        y1.append(g(j))

    def g(x):
        if 0 <= x < 1/(Nf + 1):
            z = curr_T2[0,i - 1]*(Nf + 1)*x
        elif 1/(Nf + 1) <= x < Nf/(Nf + 1):
            for k in range(1,Nf):
                if k/(Nf + 1) <= x < (k + 1)/(Nf + 1):
                    z = curr_T2[k - 1,i - 1]*(-(Nf + 1)*x + (k + 1)) + curr_T2[k,i - 1]*((Nf + 1)*x - k)
        elif Nf/(Nf + 1) <= x < 1:
            z = curr_T2[Nf - 1,i - 1]*(-(Nf + 1)*x + (Nf + 1))
        else:
            z = 0
        return z

    x2=[]
    y2=[]

    a=np.linspace(0,1,10*(Nf + 1) - 1)
    for j in a:
        x2.append(j)
        y2.append(g(j))

    def g(x):
        if 0 <= x < 1/(Nf + 1):
            z = curr_T3[0,i - 1]*(Nf + 1)*x
        elif 1/(Nf + 1) <= x < Nf/(Nf + 1):
            for k in range(1,Nf):
                if k/(Nf + 1) <= x < (k + 1)/(Nf + 1):
                    z = curr_T3[k - 1,i - 1]*(-(Nf + 1)*x + (k + 1)) + curr_T3[k,i - 1]*((Nf + 1)*x - k)
        elif Nf/(Nf + 1) <= x < 1:
            z = curr_T3[Nf - 1,i - 1]*(-(Nf + 1)*x + (Nf + 1))
        else:
            z = 0
        return z

    x3=[]
    y3=[]

    a=np.linspace(0,1,10*(Nf + 1) - 1)
    for j in a:
        x3.append(j)
        y3.append(g(j))

    plt.figure(i)

    plt.title('dimension different')
    plt.plot(x1,y1, color='green', label='dimension of 10')
    plt.plot(x2,y2, color='red', label='dimension of 20')
    plt.plot(x3,y3, color='blue', label='dimension of 40')
    plt.legend()

    plt.savefig('/Users/pengyifan/Desktop/1d_vaob/functionimage/figure_%d.png'%(i))

plt.show()
