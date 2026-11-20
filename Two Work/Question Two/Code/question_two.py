import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def RK4_pass(f,x,Y,h):
    """Perform a single step of the classical 4th-order Runge-Kutta method.

    f : callable f(x, Y) -> array, the right-hand side of Y' = f(x, Y)
    x : current point
    Y : current state vector [y, y']
    h : step size
    Returns the state vector at x + h.
    """
    k1 = h * f(x,Y)                        # slope at the start of the interval
    k2 = h * f(x + h/2, Y + k1/2)          # slope at the midpoint, using k1
    k3 = h * f(x + h/2, Y + k2/2)          # slope at the midpoint, using k2
    k4 = h * f(x + h, Y + k3)              # slope at the end of the interval
    return Y + (k1 + 2*k2 + 2*k3 + k4)/6   # weighted average of the slopes

def RK4(f,x,Y0,h):
    """Solve an initial value problem on a grid using RK4.

    x  : vector of N+1 equally spaced points
    Y0 : initial condition [y(a), y'(a)]
    Returns a (2, N+1) array: row 0 is y, row 1 is y'.
    """
    N = len(x) - 1
    Y = np.zeros((2,N+1)) # each column stores [y, y'] at one point 
    Y[:,0] = Y0           # set the initial condition
    for i in range(N):
        Y[:,i+1] = RK4_pass(f,x[i],Y[:,i],h)  # advance one step
    return Y

def linear__shooting(f_u,f_v,a,b,alpha,beta,N):
    """Solve the linear BVP y'' = p(x)y' + q(x)y + r(x),
    with y(a) = alpha and y(b) = beta, by the linear shooting method.

    f_u : system for u (full equation, includes r(x))
    f_v : system for v (homogeneous equation, without r(x))
    Returns the grid x, the solution y, and the arrays u and v.
    """
    h = (b-a)/N                     # step size
    x = np.linspace(a,b,N+1)        # grid with N+1 points

    u = RK4(f_u,x,[alpha,0],h)      # u(a) = alpha, u'(a) = 0
    v = RK4(f_v,x,[0,1],h)          # v(a) = 0,     v'(a) = 1

    c = (beta - u[0,N]) / v[0,N]    # choose c so that y(b) = beta
    y = u[0,:] + c*v[0,:]           # y = u + c*v (only the y row is used)

    return x,y,u,v

#a = x0, b = xf
a, b = 0, np.pi/2

#alpha = y0, beta = yf
alpha, beta = -0.3, -0.1

#Our ODE: y" = y' + 2y + cos(x) 
# with p = 1, q = 2 , r = cos(x)
p = lambda x: 1
q = lambda x: 2
r = lambda x: np.cos(x)

# u system: Y = [u, u'] -> [u', p(x)u' + q(x)u + r(x)]
f_u = lambda x,Y: np.array([Y[1],p(x)*Y[1] + q(x)*Y[0] + r(x)])

# v system: same as u, but without the non-homogeneous term r(x)
f_v = lambda x,Y: np.array([Y[1], p(x)*Y[1] + q(x)*Y[0]])

#Analytic ODE: -1/10(sin(x) + 3cos(x))
y_original = lambda x: (-1/10)*(np.sin(x) + 3*np.cos(x))

#Creating dictionary to keep informations and a list of N to compare 
list_of_N = [4,8,16,32,64]
results = {}

for N in list_of_N:
    #Calling linear__shooting()
    x, y_numeric, u, v = linear__shooting(f_u,f_v,a,b,alpha,beta,N)

    #Calculing analytic solution 
    y_exact = y_original(x)

    #Calculing absolute error
    abs_error = np.abs(y_exact - y_numeric)

    #Calculing relative error
    rel_error = abs_error/np.abs(y_numeric)
    if y_numeric[0] == 0: rel_error[0] = 0

    #Add N's results on the dictionary 
    data = {
        'x':x,
        'u':u[0,:],
        'v':v[0,:],
        'y exact': y_exact,
        'y numeric': y_numeric,
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
until_y = y_original(until_x)
plt.plot(until_x,until_y,'k-',linewidth=1.2,label='Exact', zorder=1)

#Creating the dictionary for colors to see better a comparision on the graphic
colors = plt.cm.viridis(np.linspace(0,0.9,len(list_of_N)))
sizes = [16,12,8,4,2] #Sizes of anel to view in the graphic

for k,(N,df) in enumerate(results.items()):
    plt.plot(df['x'],df['y numeric'],linestyle='none',marker='o',markersize=sizes[k],markerfacecolor='none',markeredgecolor=colors[k],label=f'N={N}')

plt.xlabel('x')
plt.ylabel('Value of y')
plt.title('Comparation: Linear Shooting x Exact Solve')
plt.legend(loc='best')
plt.grid(True,alpha=0.3)

plt.show()