import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Function RK4
def fourth(f,t,n,h,y0):
    #Defining vector of zeros 
    fourth_order_y = np.zeros(n)

    #Defining y(0) = 0.5
    fourth_order_y[0] = y0

    #Defining Runge-Kulta's method of Fourth Order
    for i in range(n-1):
        # Fourth Order
        K1 = f(t[i],fourth_order_y[i])
        K2 = f(t[i]+(h/2),fourth_order_y[i] + (h/2) * K1)
        K3 = f(t[i]+(h/2),fourth_order_y[i] + (h/2) * K2)
        K4 = f(t[i] + h, fourth_order_y[i] + h*K3)
        fourth_order_y[i+1] = fourth_order_y[i] + (h/6)*(K1 + 2*K2 + 2*K3 + K4)
    #End

    return fourth_order_y
#End Function

#Function Adam-Bashforth of Five pass
def AB5(f,t0,y0,y1,y2,y3,y4,tf,h):
    #f  : function of ODE
    #t0 : initial instance
    #t1 : final instance
    #y0 : initial condition
    #y1 : Second point calculated with RK because method to start need five points
    #y2 : Third point calculated with RK because method to start need five points
    #y3 : Fourth point calculated with RK because method to start need five points
    #y4 : Five point calculated with RK because method to start need five points
    #h  : Size of pass

    #N : Numbers of pass
    N = round((tf - t0) / h)

    #t : Mesh of time
    t = [t0 + i*h for i in range(N+1)]

    # Initial Points
    y = np.zeros(N+1)
    y[0] = y0
    y[1] = y1
    y[2] = y2
    y[3] = y3
    y[4] = y4

    # Initial Inclinations
    F = np.zeros(N+1)
    F[0] = f(t[0],y[0])
    F[1] = f(t[1],y[1])
    F[2] = f(t[2],y[2])
    F[3] = f(t[3],y[3])
    F[4] = f(t[4],y[4])

    # Main Bow, here where do Adam Bashforth of Four pass
    for n in range(4,N):
        y[n+1] = y[n] + h * (1901/720 * F[n] - 2774/720 * F[n-1] + 2616/720 * F[n-2] - 1274/720 * F[n-3] + 251/720 * F[n-4])
        F[n+1] = f(t[n+1],y[n+1])
    #End for

    #Return instances and numeric solution
    return t, y
#End Function

#Defining initial values 
h,t0, tf , y0 = 0.2, 0, 2, 0.5
t = np.arange(t0,tf+h,h)
n = len(t)

#Defining Our ODE: y - t² + 1
f = lambda t,y : y - t**2 + 1

#Analytic ODE
exact_y = lambda t: (t+1)**2 - 0.5*np.exp(t)

#Calculating y1
RK_y = fourth(f,t,n,h,y0)
y1 = RK_y[1]
y2 = RK_y[2]
y3 = RK_y[3]
y4 = RK_y[4]

#Calling Function AB5
AB5_t,AB5_y = AB5(f,t0,y0,y1,y2,y3,y4,tf,h)
AB5_t,AB5_y = np.array(AB5_t) , np.array(AB5_y)

#Solving Analytic ODE 
original_y = exact_y(AB5_t)

#Calculating Absolute and Relative error
abs_error = np.abs(AB5_y - original_y)
rel_error = np.abs(abs_error/(AB5_y))

#Create dictionary
dc = {
    'h': h,
    't': AB5_t,
    'Exact': original_y,
    'Adam-Bashforth of Five Pass': AB5_y,
    'Abs': abs_error,
    'Rel': rel_error
}

#Create DatFrame
df = pd.DataFrame(dc)

#Output
print(df)

# Comparison Graphic 
plt.plot(AB5_t,original_y,'k-',linewidth=2,label='Exact')
plt.plot(AB5_t,AB5_y,'ro--',linewidth=1.5,label='Adam-Bashforth of five pass')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Adam-Bashforth of five pass x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
