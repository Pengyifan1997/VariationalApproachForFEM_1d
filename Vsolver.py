import numpy as np 
import sympy as syp
import scipy.linalg as spl 
import torch 
import matplotlib.pyplot as plt 

x = syp.symbols("x")

# the equation is: - divergence(a*gradient(u)) = 0
# for each u : u(xi) = 1 , u(xj) = 0 for j doesn't equal to i
# left side assumption
epsilon = 0.0013
a = 2 + syp.cos(x/epsilon)

# right side data
# the number of the test functions
M = np.array((1000))

# the max demension of the test function
D = np.array((40))

# generate the test functions
# here we use Fourier series

# generate their coefficients
c = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/testfunction_%d_%d.txt'%(M,D,M,D))

# solve the equation
# the coarse grid
Nc = np.array((31))

# refinement times
K = np.array((4))

# the fine grid
Nf = K*(Nc + 1) - 1

# stiffness matrix on the fine grid
L = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/stiffnessmatrix_%d_%d.txt'%(M,D,Nc,K,Nc,K))

# the coefficinet matrix of right side
F = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/coefficinetmatrix_%d_%d.txt'%(M,D,Nc,K,Nc,K))

# the numerical solution on the fine grid
xf = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/numsolution_%d_%d.txt'%(M,D,Nc,K,Nc,K))

# set initial value for the target basis
# get the initial value matrix
ivmatrix = np.loadtxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/VariationalBasis/VariationalBasis.txt'%(M,D,Nc,K))

c = c.astype(np.float32)
c = torch.from_numpy(c)
xf = xf.astype(np.float32)
xf = torch.from_numpy(xf)
F = F.astype(np.float32)
F = torch.from_numpy(F)
L = L.astype(np.float32)
L = torch.from_numpy(L)

# define the variables and set their initial value
ivmatrix = ivmatrix.astype(np.float32)
T = torch.from_numpy(ivmatrix)
T.requires_grad = True

# use pytorch to solve the maximunm problem

xx=[]
yy=[]

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.Adam([T],lr = learning_rate)
Lossa = np.ones((1,1))
for t in range(1):
    
    f = torch.mm(F,torch.transpose(c,0,1))
    xc,LU = torch.gesv(torch.mm(torch.transpose(T,0,1),f),torch.mm(torch.mm(torch.transpose(T,0,1),L),T))
    model = torch.transpose(torch.mm(T,xc),0,1)

    

    loss = loss_fn(model,xf)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    curr_loss = np.array(loss.item())
    Lossp = np.ones((1,1))
    Lossp = np.dot(Lossp,curr_loss)

    if Lossp < Lossa :
        Lossa = Lossp
        Ta = T
    
    xx.append(np.log10(t))
    yy.append(np.log10(Lossp[0,0]))

plt.figure(0)
plt.plot(xx,yy)
plt.savefig('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/VariationalBasis/lossdecay.png'%(M,D,Nc,K))

print(t,Lossa)

T.requires_grad = False
curr_T = T.numpy()

np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/MSFBasis/transformmatrix.txt'%(M,D,Nc,K),curr_T)

for i in range(1,Nc + 1):
    def g(x):
        if 0 <= x < 1/(Nf + 1):
            z = curr_T[0,i - 1]*(Nf + 1)*x
        elif 1/(Nf + 1) <= x < Nf/(Nf + 1):
            for k in range(1,Nf):
                if k/(Nf + 1) <= x < (k + 1)/(Nf + 1):
                    z = curr_T[k - 1,i - 1]*(-(Nf + 1)*x + (k + 1)) + curr_T[k,i - 1]*((Nf + 1)*x - k)
        elif Nf/(Nf + 1) <= x < 1:
            z = curr_T[Nf - 1,i - 1]*(-(Nf + 1)*x + (Nf + 1))
        else:
            z = 0
        return z

    plt.figure(i)

    x=[]
    y=[]

    a=np.linspace(0,1,10*(Nf + 1) - 1)
    for j in a:
        x.append(j)
        y.append(g(j))

    plt.plot(x,y)

    plt.savefig('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/VariationalBasis/figure_%d.png'%(M,D,Nc,K,i))

Loss = np.sqrt(295.6079/(Nf + 1)/(M))

print(Loss)

np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/VariationalBasis/loss.txt'%(M,D,Nc,K),Loss)