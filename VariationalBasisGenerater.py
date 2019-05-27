import numpy as np 
import sympy as syp
import scipy.linalg as spl 
import tensorflow as tf 
import matplotlib.pyplot as plt 

x = syp.symbols("x")

# the equation is: - divergence(a*gradient(u)) = 0
# left side assumption
epsilon = 0.0013
a = 2 + syp.cos(x/epsilon)

# right side data
# the number of the test functions
M = np.array((500))

# the max demension of the test function
D = np.array((20))

# the coarse grid
Nc = np.array((63))

# refinement times
# one coarse grid is divided into K fine grids
K = np.array((4))

# the fine grid
Nf = K*(Nc + 1) - 1

# the coefficinet matrix to simply the calculation
C = np.zeros((1,Nf + 1))
for i in range(0,Nf + 1):
    A = a*a
    C[0,i] = (Nf + 1)*(Nf + 1)*syp.integrate(A,(x,i/(Nf + 1),(i + 1)/(Nf + 1)))

# set the initial value of target functions
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

Tpp1 = np.zeros((Nf + 1,Nc))
Left5 = np.zeros((Nf + 1,Nf))
for i in range(0,Nf):
    Left5[i,i] = 1
Tpp1 = tf.add(tf.matmul(Left5,T),Tpp1)
Tpp2 = np.zeros((Nf + 1,Nc))
Left6 = np.zeros((Nf + 1,Nf))
for i in range(0,Nf):
    Left6[i + 1,i] = - 1
Tpp2 = tf.add(tf.matmul(Left6,T),Tpp2)
Tpp3 = tf.add(Tpp1,Tpp2)
Tpp4 = tf.square(Tpp3)

x = tf.placeholder(tf.float64,(1,1))

model = tf.matmul(x,tf.matmul(C,Tpp4))

loss = tf.reduce_sum(model)

optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

x_train = np.ones((1,1))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    sess.run(train,{x: x_train})
    curr_T1, curr_T2, curr_loss = sess.run([T1,T2,loss],{x: x_train})
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

#for i in range(1,Nc + 1):
#    def g(x):
#        if 0 <= x < 1/(Nf + 1):
#            z = curr_T[0,i - 1]*(Nf + 1)*x
#        elif 1/(Nf + 1) <= x < Nf/(Nf + 1):
#            for k in range(1,Nf):
#                if k/(Nf + 1) <= x < (k + 1)/(Nf + 1):
#                    z = curr_T[k - 1,i - 1]*(-(Nf + 1)*x + (k + 1)) + curr_T[k,i - 1]*((Nf + 1)*x - k)
#        elif Nf/(Nf + 1) <= x < 1:
#            z = curr_T[Nf - 1,i - 1]*(-(Nf + 1)*x + (Nf + 1))
#        else:
#            z = 0
#        return z
#
#    plt.figure(i)
#
#    x=[]
#    y=[]
#
#    a=np.linspace(0,1,10*(Nf + 1) - 1)
#    for j in a:
#        x.append(j)
#        y.append(g(j))
#
#    plt.plot(x,y)
#
#    plt.show()

np.savetxt('/Users/pengyifan/Desktop/1d_vaob/testfunction_%d_%d/grid_%d_%d/VariationalBasis/VariationalBasis.txt'%(M,D,Nc,K),curr_T)
