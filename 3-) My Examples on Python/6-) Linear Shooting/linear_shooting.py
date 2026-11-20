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
a, b = 1, 2

#alpha = y0, beta = yf
alpha, beta = 1, 2

#N: number of subintervals
N = 20

# u system: Y = [u, u'] -> [u', p(x)u' + q(x)u + r(x)]
# with p = -2/x, q = 2/x^2, r = sin(ln x)/x^2
f_u = lambda x,Y: np.array([Y[1],-(2/x)*Y[1] + (2/(x**2))*Y[0] + np.sin(np.log(x))/(x**2)])

# v system: same as u, but without the non-homogeneous term r(x)
f_v = lambda x,Y: np.array([Y[1], -(2/x)*Y[1] + (2/(x**2))*Y[0]])

# y exact 
c1 = (4/7) * (1.725 + 0.3 * np.sin(np.log(2)) + 0.1*np.cos(np.log(2)))
c2 = 1.1 - c1
y_original = lambda x: c1*x + c2*(x**-2)  - 0.3*np.sin(np.log(x)) - 0.1*np.cos(np.log(x))

#Calling linear__shooting()
x, y_numeric, u, v = linear__shooting(f_u,f_v,a,b,alpha,beta,N)

#Calculing analytic solution 
y_exact = y_original(x)

#Calculing absolute error
abs_error = np.abs(y_exact - y_numeric)

#Calculing relative error
rel_error = abs_error/np.abs(y_numeric)
if y_numeric[0] == 0: rel_error[0] = 0

#Creating dictionary 
dic = {
    'x':x,
    'u':u[0,:],
    'v':v[0,:],
    'y exact': y_exact,
    'y numeric': y_numeric,
    'absolute error': abs_error,
    'relative error': rel_error
}

#Creating DataFrame
df = pd.DataFrame(dic)

#Output
print(df)

# Comparison Graphic 
plt.plot(x,y_exact,'k-',linewidth=2,label='Exact')
plt.plot(x,y_numeric,'ro--',linewidth=1.5,label='Linear Shooting')

plt.xlabel('x')
plt.ylabel('Value of y')
plt.title('Comparation: Linear Shooting x Exact Solve, with N = 20')
plt.legend(loc='best')
plt.grid(True)

plt.show()