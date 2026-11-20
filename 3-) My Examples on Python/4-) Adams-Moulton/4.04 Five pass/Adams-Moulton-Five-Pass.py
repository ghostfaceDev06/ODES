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

#Function Adams-Moulton of Five pass
def AM5(f,t0,y0,y1,y2,y3,y4,tf,h,tol,iter_max):
    #f  : function of ODE
    #t0 : initial instance
    #tf : final instance
    #y0 : initial condition
    #y1 : Second point calculated with RK because method to start need five points
    #y2 : Third point calculated with RK because method to start need five points
    #y3 : Fourth point calculated with RK because method to start need five points
    #y4 : Fifth point calculated with RK because method to start need five points
    #h  : Size of pass
    #tol: Size of error
    #iter_max: max of iterations possible 

    #Coefficients 
    a = np.zeros(6)
    a[0] =  475/1440
    a[1] = 1427/1440 
    a[2] = -798/1440
    a[3] = 482/1440
    a[4] = -173/1440
    a[5] =  27/1440


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

    # Main Bow, here where do Adams Moulton of Four pass
    for n in range(4,N):

        #Known part
        known = a[1] * F[n] + a[2] * F[n-1] + a[3] * F[n-2] + a[4] * F[n-3] + a[5] * F[n-4]

        #initial kick
        new_y = y[n]

        for _ in range(iter_max):

            old_y = new_y
            new_y = y[n] + h * (a[0] * f(t[n+1],old_y) + known)

            if abs(new_y - old_y) < tol: break
            #End if
        #End for
            
        y[n+1] = new_y
        F[n+1] = f(t[n+1],y[n+1])
    #End for

    #Return instances and numeric solution
    return t, y
#End Function

#Defining initial values 
h,t0, tf , y0, tol, iter_max= 0.2, 0, 2, 0.5, 1e-12, 1000
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

#Calling Function AM5
AM5_t,AM5_y = AM5(f,t0,y0,y1,y2,y3,y4,tf,h,tol,iter_max)
AM5_t,AM5_y = np.array(AM5_t) , np.array(AM5_y)

#Solving Analytic ODE 
original_y = exact_y(AM5_t)

#Calculating Absolute and Relative error
abs_error = np.abs(AM5_y - original_y)
rel_error = np.abs(abs_error/(AM5_y))

#Create dictionary
dc = {
    'h': h,
    't': AM5_t,
    'Exact': original_y,
    'Adams-Moulton of Five Pass': AM5_y,
    'Abs': abs_error,
    'Rel': rel_error
}

#Create DatFrame
df = pd.DataFrame(dc)

#Output
print(df)

# Comparison Graphic 
plt.plot(AM5_t,original_y,'k-',linewidth=2,label='Exact')
plt.plot(AM5_t,AM5_y,'ro--',linewidth=1.5,label='Adams-Moulton of Five pass')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Adams-Moulton of five pass x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
