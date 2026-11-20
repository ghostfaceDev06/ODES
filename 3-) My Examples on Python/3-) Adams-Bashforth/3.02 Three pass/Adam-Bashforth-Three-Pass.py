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

#Function Adam-Bashforth of Three pass
def AB3(f,t0,y0,y1,y2,tf,h):
    #f  : function of ODE
    #t0 : initial instance
    #t1 : final instance
    #y0 : initial condition
    #y1 : Second point calculated with RK because method to start need three points
    #y2 : Third point calculated with RK because method to start need three points
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

    # Initial Inclinations
    F = np.zeros(N+1)
    F[0] = f(t[0],y[0])
    F[1] = f(t[1],y[1])
    F[2] = f(t[2],y[2])

    # Main Bow, here where do Adam Bashforth of Three pass
    for n in range(2,N):
        y[n+1] = y[n] + h * (23/12 * F[n] - 16/12 * F[n-1] + 5/12 * F[n-2])
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

#Calling Function AB3
AB3_t,AB3_y = AB3(f,t0,y0,y1,y2,tf,h)
AB3_t,AB3_y = np.array(AB3_t) , np.array(AB3_y)

#Solving Analytic ODE 
original_y = exact_y(AB3_t)

#Calculating Absolute and Relative error
abs_error = np.abs(AB3_y - original_y)
rel_error = np.abs(abs_error/(AB3_y))

#Create dictionary
dc = {
    'h': h,
    't': AB3_t,
    'Exact': original_y,
    'Adam-Bashforth of Three Pass': AB3_y,
    'Abs': abs_error,
    'Rel': rel_error
}

#Create DatFrame
df = pd.DataFrame(dc)

#Output
print(df)

# Comparison Graphic 
plt.plot(AB3_t,original_y,'k-',linewidth=2,label='Exact')
plt.plot(AB3_t,AB3_y,'ro--',linewidth=1.5,label='Adam-Bashforth of three pass')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Adam-Bashforth of three pass x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
