import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def build_system(p,q,r,x,h,alpha,beta):
    #Builds the tridiagonal system A.Y = B for the N-1 interior nodes 
    N = len(x) - 1
    A = np.zeros((N-1,N-1))
    B = np.zeros((N-1))

    for i in range(N-1):
        xi = x[i+1]

        #Coefficients from the the finite-difference formulas
        A_left = 1/(h**2) - p(xi)/(2*h)    #Left neighbor
        A_cent = -2/(h**2) + q(xi)         #Central point
        A_right = 1/(h**2) + p(xi)/(2*h)   #Right neighbor 

        A[i,i] = A_cent                #Main diagonal
        if i > 0: A[i,i-1] = A_left    #sub-diagonal
        if i < N-2: A[i,i+1] = A_right #super-diagonal

        B[i] = r(xi)                 #Right-hand side

        #Known boundary values move to the right-hand side
        if i == 0: B[i] = B[i] - A_left*alpha
        if i == N-2: B[i] = B[i] - A_right*beta

    return A,B

def FINITE_DIFFERENCE_METHOD(p,q,r,a,b,alpha,beta,N):
    h = (b-a)/N #step size
    x = np.linspace(a,b,N+1) #N+1 nodes

    A,B = build_system(p,q,r,x,h,alpha,beta)

    Y_intern = np.linalg.solve(A,B)
    Y_num = np.concatenate(([alpha],Y_intern,[beta])) #Attach the boundary values

    return x,Y_num

#Problem definition 
a, b, alpha, beta, N = 1, 2, 1, 2, 10

#ODE written as: y'' + p(x).y' + q(x).y = r(x)
p = lambda x: 2/x
q = lambda x: -2/(x**2)
r = lambda x: np.sin(np.log(x)) / (x**2)

#Calling FINITE_DIFFERENCE_METHOD()
x, Y_num = FINITE_DIFFERENCE_METHOD(p,q,r,a,b,alpha,beta,N)

#Calculing Analytic ODE: 
c1 = (69 + 12*np.sin(np.log(2)) + 4*np.cos(np.log(2)))/70
c2 = (11/10 - c1)
Y_original = lambda x: c1*x + c2/(x**2) - 0.3*np.sin(np.log(x)) - 0.1*np.cos(np.log(x))
Y_exact = Y_original(x)

#Calculing absolute error
abs_error = np.abs(Y_exact-Y_num)

#Calculing relative error
rel_error = abs_error/ np.abs(Y_num)

#Calculing until x and y
until_x = np.linspace(a,b,200)
until_y = Y_original(until_x)

#Creating dictionary 
dic = {
    'x':x,
    'y exact': Y_exact,
    'y numeric': Y_num,
    'absolute error': abs_error,
    'relative error': rel_error
}

#Creating DataFrame
df = pd.DataFrame(dic)

#Output
print(df)

# Comparison Graphic 
plt.plot(until_x,until_y,'k-',linewidth=2,label='Exact')
plt.plot(x,Y_num,'ro--',linewidth=1.5,label='Finite Difference')

plt.xlabel('x')
plt.ylabel('Value of y')
plt.title('Comparation: Finite Difference x Exact Solve, with N = 10')
plt.legend(loc='best')
plt.grid(True)

plt.show()