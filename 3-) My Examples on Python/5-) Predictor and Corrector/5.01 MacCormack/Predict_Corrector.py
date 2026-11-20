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

#Function Predictor and Corrector Method
def PC(f,t0,y0,y1,y2,y3,tf,h):
    #f  : function of ODE
    #t0 : initial instance
    #tf : final instance
    #y0 : initial condition
    #y1 : Second point calculated with RK because method to start need Four points
    #y2 : Third point calculated with RK because method to start need Four points
    #y3 : Fourth point calculated with RK because method to start need Four points
    #h  : Size of pass
    
    #Corrector 
    a = np.zeros(4)
    a[0] = 9/24
    a[1] = 19/24
    a[2] = -5/24 
    a[3] = 1/24

    #Predictor
    b = np.zeros(4)
    b[0] = 55/24
    b[1] = -59/24
    b[2] = 37/24
    b[3] = -9/24 


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

    # Initial Inclinations
    F = np.zeros(N+1)
    F[0] = f(t[0],y[0])
    F[1] = f(t[1],y[1])
    F[2] = f(t[2],y[2])
    F[3] = f(t[3],y[3])

    # Main Bow, here where do Predictor and Corrector method
    for n in range(3,N):

        #Predictor (Bashforth)
        y_pred = y[n] + h * (b[0]*F[n] + b[1]*F[n-1] + b[2]*F[n-2] + b[3]*F[n-3])

        #Check f on reavible point
        f_pred = f(t[n+1], y_pred)

        #Corrector (Moulton)
        y[n+1] = y[n] + h * (a[0]*f_pred + a[1]*F[n] + a[2]*F[n-1] + a[3]*F[n-2])

        #Check f on fixed point 
        F[n+1] = f(t[n+1], y[n+1])
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

#Calculating y1,y2 and y3
RK_y = fourth(f,t,n,h,y0)
y1 = RK_y[1]
y2 = RK_y[2]
y3 = RK_y[3]

#Calling Function PC
PC_t,PC_y = PC(f,t0,y0,y1,y2,y3,tf,h)
PC_t,PC_y = np.array(PC_t) , np.array(PC_y)

#Solving Analytic ODE 
original_y = exact_y(PC_t)

#Calculating Absolute and Relative error
abs_error = np.abs(PC_y - original_y)
rel_error = np.abs(abs_error/(PC_y))

#Create dictionary
dc = {
    'h': h,
    't': PC_t,
    'Exact': original_y,
    'Predictor and Corrector': PC_y,
    'Abs': abs_error,
    'Rel': rel_error
}

#Create DatFrame
df = pd.DataFrame(dc)

#Output
print(df)

# Comparison Graphic 
plt.plot(PC_t,original_y,'k-',linewidth=2,label='Exact')
plt.plot(PC_t,PC_y,'ro--',linewidth=1.5,label='Predictor and Corrector')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Predictor and Corrector x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
