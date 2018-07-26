import numpy as np 
import sympy as syp
from numpy import poly1d 
import random as rd 
import scipy.linalg as spl 
import tensorflow as tf 
import pdb 
import matplotlib.pyplot as plt 

x = syp.symbols("x")

# the equation is: - divergence(a*gradient(u)) = f
# left side assumption
a = 1

# right side data
# the number of the test functions
M = np.array((500))

# generate the test functions
# here we use Fourier series
# generate their demensions
d = np.zeros((M,2))
for i in range(0,M):
    # demension of sin mod
    d[i,0] = rd.randint(1,10)
    # demension of cos mod
    d[i,1] = rd.randint(1,10)

# generate their coefficients
c = np.zeros((M,21))
for i in range(0,M):
    # coefficients of sin mod
    for j in range(0,10):
        if j < d[i,0]:
            c[i,j] = 2*rd.random() - 1
    # coefficients of cos mod
    for j in range(10,20):
        if j - 10 < d[i,1]:
            c[i,j] = 2*rd.random() - 1
    # constant coefficient
    c[i,20] = 2*rd.random() - 1

# solve the equation
# the coarse grid
Nc = np.array((10))

# refinement times
K = np.array((4))

# the fine grid
Nf = K*(Nc + 1) - 1

# the basis on the fine grid are simple piece-wise linear functions
def basis(number,interval):
    for i in range(0,Nf):
        if number == i:
            if interval == i:
                f = (Nf + 1)*x - (i + 1) 
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
        for k in range(0,Nf + 1):
            mid2 = a*syp.diff(basis(i,k),x,1)*syp.diff(basis(j,k),x,1)
            mid3 = syp.integrate(mid2,(x,(k + 1)/(Nf + 1),k/(Nf + 1)))
            mid1 = mid3 + mid1
        L[i,j] = mid1

# the coefficinet matrix of right side
F = np.zeros((Nf,21))
for i in range(0,Nf):
    for j in range(0,10):
        mid4 = np.zeros(1)
        for k in range(0,Nf + 1):
            mid5 = basis(i,k)*syp.sin((k + 1)*np.pi*x)
            mid6 = syp.integrate(mid5,(x,(k + 1)/(Nf + 1),k/(Nf + 1)))
            mid4 = mid6 + mid4
        F[i,j] = mid4
    for j in range(10,20):
        mid4 = np.zeros(1)
        for k in range(0,Nf + 1):
            mid5 = basis(i,k)*syp.cos((k - 9)*np.pi*x)
            mid6 = syp.integrate(mid5,(x,(k + 1)/(Nf + 1),k/(Nf + 1)))
            mid4 = mid6 + mid4
        F[i,j] = mid4
    mid4 = np.zeros(1)
    for k in range(0,Nf + 1):
        mid5 = basis(i,k)
        mid6 = syp.integrate(mid5,(x,(k + 1)/(Nf + 1),k/(Nf + 1)))
        mid4 = mid6 + mid4
    F[i,20] = mid4

# the numerical solution on the fine grid
xf = np.zeros((1,Nf))
for k in range(0,M):
    f = np.dot(F,np.transpose(c[k,0:21]))
    xf1 = spl.solve(L,f)
    xf = np.row_stack((xf,np.transpose(xf1)))
xf = np.delete(xf,0,0)

# set initial value for the target basis
# the major matrix
iv1 = np.zeros((2*K - 2,Nc))
for i in range(0,Nc):
    for j in range(0,K):
        iv1[j,i] = (j + 1)/K
    for j in range(K,2*K - 2):
        iv1[j,i] = (2*K - 1 - j)/K

# the rest matrix
iv2 = np.zeros((K*(Nc - 1) - 1,Nc))

# use tensorflow to solve the maximunm problem
# define the variables and set their initial value
T1 = tf.Variable(iv1,dtype = tf.float64)
T2 = tf.Variable(iv2,dtype = tf.float64)

# fill the last blank of the major matrix
Tp1 = np.zeros((2*K - 1,Nc))
for i in range(0,Nc):
    Tp1[2*K - 2,i] = K
left1 = np.zeros((2*K - 1,2*K - 2))
for i in range(0,2*K - 2):
    left1[2*K - 2,i] = -1
Tp1 = tf.add(Tp1,tf.matmul(left1,T1))
for i in range(0,2*K - 2):
    left1 = np.zeros((2*K - 1,2*K - 2))
    left1[i,i] = 1
    Tp1 = tf.add(Tp1,tf.matmul(left1,T1))

# fill the last blank of the rest matrix
Tp2 = np.zeros((K*(Nc - 1),Nc))
left2 = np.zeros((K*(Nc - 1),K*(Nc - 1) - 1))
for i in range(0,K*(Nc - 1) - 1):
    left2[K*(Nc - 1) - 1,i] = -1
Tp2 = tf.add(Tp2,tf.matmul(left2,T2))
for i in range(0,K*(Nc - 1) - 1):
    left2 = np.zeros((K*(Nc - 1),K*(Nc - 1) - 1))
    left2[i,i] = 1
    Tp2 = tf.add(Tp2,tf.matmul(left2,T2))

T = np.zeros((Nf,Nc))
# reform the major matrix
for i in range(0,Nc):
    Left3 = np.zeros((Nf,2*K - 1))
    for j in range(0,2*K - 1):
        Left3[i*K + j,j] = 1
    Right3 = np.zeros((Nc,Nc))
    Right3[i,i] = 1
    T = tf.add(tf.matmul(Left3,tf.matmul(Tp1,Right3)),T)

# reform the rest matrix
for i in range(0,Nc):
    Left4 =  np.zeros((Nf,(Nc - 1)*K))
    for j in range(0,i*K):
        Left4[j,j] = 1
    for j in range(i*K,(Nc - 1)*K):
        Left4[j + 2*K - 1,j] = 1
    Right4 = np.zeros((Nc,Nc))
    Right4[i,i] = 1
    T = tf.add(tf.matmul(Left4,tf.matmul(Tp2,Right4)),T)

x = tf.placeholder(tf.float64,(M,21))

F = tf.convert_to_tensor(F)
f = tf.matmul(F,tf.transpose(x))
xc = tf.matrix_solve(tf.matmul(tf.matmul(tf.transpose(T),L),T),tf.matmul(tf.transpose(T),f))

model = tf.transpose(tf.matmul(T,xc))

y = tf.placeholder(tf.float64,(M,Nf))

xd = tf.subtract(y,model)
nm = np.zeros((Nf,Nf))
for i in range(0,Nf):
    nm[i,i] = 2/3/(Nf + 1)
    if i > 0:
        nm[i,i - 1] = 1/6/(Nf + 1)
    if i < Nf - 1:
        nm[i,i + 1] = 1/6/(Nf + 1)
xnm = tf.matmul(xd,tf.matmul(nm,tf.transpose(xd)))
xtr = np.zeros((1,1))
for i in range(0,M):
    L3 = np.zeros((1,M))
    L3[0,i] = 1
    R3 = np.zeros((M,1))
    R3[i,0] = 1
    xtr = tf.add(tf.matmul(L3,tf.matmul(xnm,R3)),xtr)

loss = xtr[0,0]

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

x_train = c
y_train = xf

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    sess.run(train,{x: x_train, y: y_train})
    curr_T1, curr_T2, curr_loss = sess.run([T1,T2,loss],{x: x_train, y: y_train})
print("loss: %s"%(curr_loss))

# fill the last blank of the major matrix
Ta1 = np.zeros((2*K - 1,Nc))
for i in range(0,Nc):
    Ta1[2*K - 2,i] = K
left1 = np.zeros((2*K - 1,2*K - 2))
for i in range(0,2*K - 2):
    left1[2*K - 2,i] = -1
Ta1 = Ta1 + np.dot(left1,curr_T1)
for i in range(0,2*K - 2):
    left1 = np.zeros((2*K - 1,2*K - 2))
    left1[i,i] = 1
    Ta1 = Ta1 + np.dot(left1,curr_T1)

# fill the last blank of the rest matrix
Ta2 = np.zeros((K*(Nc - 1),Nc))
left2 = np.zeros((K*(Nc - 1),K*(Nc - 1) - 1))
for i in range(0,K*(Nc - 1) - 1):
    left2[K*(Nc - 1) - 1,i] = -1
Ta2 = Ta2 + np.dot(left2,curr_T2)
for i in range(0,K*(Nc - 1) - 1):
    left2 = np.zeros((K*(Nc - 1),K*(Nc - 1) - 1))
    left2[i,i] = 1
    Ta2 = Ta2 + np.dot(left2,curr_T2)

curr_T = np.zeros((Nf,Nc))
# reform the major matrix
for i in range(0,Nc):
    Left3 = np.zeros((Nf,2*K - 1))
    for j in range(0,2*K - 1):
        Left3[i*K + j,j] = 1
    Right3 = np.zeros((Nc,Nc))
    Right3[i,i] = 1
    curr_T = curr_T + np.dot(Left3,np.dot(Ta1,Right3))

# reform the rest matrix
for i in range(0,Nc):
    Left4 =  np.zeros((Nf,(Nc - 1)*K))
    for j in range(0,i*K):
        Left4[j,j] = 1
    for j in range(i*K,(Nc - 1)*K):
        Left4[j + 2*K - 1,j] = 1
    Right4 = np.zeros((Nc,Nc))
    Right4[i,i] = 1
    curr_T = curr_T + np.dot(Left4,np.dot(Ta2,Right4))

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

    plt.show()