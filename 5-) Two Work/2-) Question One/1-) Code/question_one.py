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

#Convention this formule of Difference Finite is y'' + p(x)y' + q(x)y = r(x)

def FINITE_DIFFERENCE_METHOD(p,q,r,a,b,alpha,beta,N):
    h = (b-a)/N #step size
    x = np.linspace(a,b,N+1) #N+1 nodes

    A,B = build_system(p,q,r,x,h,alpha,beta)

    Y_intern = np.linalg.solve(A,B)
    Y_num = np.concatenate(([alpha],Y_intern,[beta])) #Attach the boundary values

    return x,Y_num

#Problem definition 
a, b, alpha, beta = 0, 1, 0, 2

#ODE's Problem: y'' = 4(y-x)
p = lambda x: 0
q = lambda x: -4
r = lambda x: -4*x

#Defining Analytic ODE: e²(e⁴-1)^(-1) (e^(2x) - e^(-2x)) + x
Y_original = lambda x: np.exp(2)*(np.exp(4) - 1)**(-1) * (np.exp(2*x) - np.exp(-2*x)) + x

#Creating dictionary to keep informations and a list of N to compare 
list_of_N = [4,8,16,32,64]
results = {}
for N in list_of_N:
    #Calling FINITE_DIFFERENCE_METHOD()
    x, Y_num = FINITE_DIFFERENCE_METHOD(p,q,r,a,b,alpha,beta,N)

    #Calculing Analytic ODE: 
    Y_exact = Y_original(x)

    #Calculing absolute error
    abs_error = np.abs(Y_exact-Y_num)

    #Calculing relative error
    rel_error = abs_error/ np.abs(Y_num)
    if Y_num[0] == 0 : rel_error[0] = 0  

    #Add N's results on the dictionary 
    data = {
        'x':x,
        'y exact': Y_exact,
        'y numeric': Y_num,
        'absolute error': abs_error,
        'relative error': rel_error
    }

    #Creating Dataframe
    df = pd.DataFrame(data)
    results[N] = df
    
#Outupt of DataFrame
pd.set_option('display.float_format', lambda x: f'{x:.8f}') #8 houses of precision 
for N,df in results.items():
    print("="*60)
    print(f'N = {N}')
    print("="*60)
    print(df.to_string())
    print()

#Plot graphic

#Calculing until x and y for graphic
until_x = np.linspace(a,b,200)
until_y = Y_original(until_x)
plt.plot(until_x,until_y,'k-',linewidth=1.2,label='Exact', zorder=1)

#Creating the dictionary for colors to see better a comparision on the graphic
colors = plt.cm.viridis(np.linspace(0,0.9,len(list_of_N)))
sizes = [16,12,8,4,2] #Sizes of anel to view in the graphic

for k,(N,df) in enumerate(results.items()):
    plt.plot(df['x'],df['y numeric'],linestyle='none',marker='o',markersize=sizes[k],markerfacecolor='none',markeredgecolor=colors[k],label=f'N={N}')

plt.xlabel('x')
plt.ylabel('Value of y')
plt.title('Comparation: Finite Difference x Exact Solve')
plt.legend(loc='best')
plt.grid(True,alpha=0.3)

plt.show()