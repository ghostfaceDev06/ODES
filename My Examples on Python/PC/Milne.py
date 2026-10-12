import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Function RK4
def fourth(f,t,n,h,y0):
    #Defining vector of zeros 
    fourth_order_y = np.zeros(n)

    #Defining y(0) = 0.5
    fourth_order_y[0] = y0

    #Defining Runge-Kutta's method of Fourth Order
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

#Function Milne
def milne(f,t0,tf,y0,y1,y2,y3,h):
    #f  : function of ODE
    #t0 : initial instance
    #tf : final instance
    #y0 : initial condition
    #y1 : Second point calculated with RK because method to start need Four points
    #y2 : Third point calculated with RK because method to start need Four points
    #y3 : Fourth point calculated with RK because method to start need Four points
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
    
    # Create vector of zeros for inclination
    F = np.zeros(N+1)
    
    #initial derivates
    F[0] = f(t[0],y[0])
    F[1] = f(t[1],y[1])
    F[2] = f(t[2],y[2])
    F[3] = f(t[3],y[3])
    
    # Main Bow, here where do Milne's method
    for j in range(3,N):
        
        #Predictor
        pred_y = y[j-3] + (4*h/3) * (2*F[j] - F[j-1] + 2*F[j-2])

        #Corrector
        pred_f = f(t[j+1],pred_y)

        y[j+1] = y[j-1] + (h/3) * (pred_f + 4*F[j] + F[j-1])

        #Check f on fixed point
        F[j+1] = f(t[j+1],y[j+1])
    #End for

    #Return instances and numeric solution
    return t, y
#End Function


#Defining initial values 
h,t0, tf , y0 = 0.2, 0, 2, 0.5
N = round((tf - t0) / h)
t = t0 + h * np.arange(N + 1)
n = N + 1

#Defining Our ODE: y - t² + 1
f = lambda t,y : y - t**2 + 1

#Analytic ODE: (t+1)² - 0.5e^t
exact_y = lambda t: (t+1)**2 - 0.5*np.exp(t)

#Calculating y1,y2 and y3
RK_y = fourth(f,t,n,h,y0)
y1 = RK_y[1]
y2 = RK_y[2]
y3 = RK_y[3]

#Calling Function Milne
milne_t,milne_y = milne(f,t0,tf,y0,y1,y2,y3,h)
milne_t,milne_y = np.array(milne_t) , np.array(milne_y)

#Solving Analytic ODE 
original_y = exact_y(milne_t)

#Calculating Absolute and Relative error
abs_error = np.abs(milne_y - original_y)
rel_error = np.abs(abs_error/(milne_y))

#Create dictionary
dc = {
    'h': h,
    't': milne_t,
    'Exact': original_y,
    'Milne': milne_y,
    'Abs': abs_error,
    'Rel': rel_error
}

#Create DataFrame
df = pd.DataFrame(dc)

#Output
print(df)

# Comparison Graphic 
plt.plot(milne_t,original_y,'k-',linewidth=2,label='Exact')
plt.plot(milne_t,milne_y,'ro--',linewidth=1.5,label='Milne')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Milne x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()