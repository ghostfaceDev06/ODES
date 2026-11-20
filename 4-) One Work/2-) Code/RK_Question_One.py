import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Important Definetions
#f : ODE 
#t : a <= t <= b
#n : Dimension of t
#h : Size of pass
#y0 : initial value
#original_y : Values of Analytic ODE to compare with numeric solution

#The great goal this code is to compare methods of Runge-Kutta: Euler , Modified Euler , Third and Fourth order.
#The requisits on the comparison is Absolute and Relative Error.

#Author: Raphael Meirelles de Paula da Câmara

# Function of Euler's Method
def euler(f,t,n,h,y0,original_y):
    #Defining zeros vector
    euler_y = np.zeros(n)

    #Defining y(0) 
    euler_y[0] = y0

    #Defining Euler's method
    for i in range(n-1):
    # Euler's method
        euler_f = f(t[i],euler_y[i])
        euler_y[i+1] = euler_y[i] + h * euler_f
    #End

    #Defining absolute error
    abs_error = np.abs(euler_y - original_y)

    #Calculathing relative error
    rel_error = np.abs(abs_error/euler_y) 

    #Return Y calculated for Euler also return absolute and relative
    return euler_y,abs_error,rel_error
#End Function

#Function of Modified Euler's Method
def modeuler(f,t,n,h,y0,original_y):
    #Defining vector of zeros 
    modeuler_y = np.zeros(n)

    #Defining y(0) 
    modeuler_y[0] = y0

    #Defining modified Euler's method 
    for i in range(n-1):
        # modified Euler's method
        K1 = f(t[i],modeuler_y[i])
        K2 = f(t[i+1],modeuler_y[i] + h*K1)
        modeuler_y[i+1] = modeuler_y[i] + (h/2) * ( K1 + K2) 
    #End 
   
    #Defining absolute error 
    abs_error = np.abs(modeuler_y - original_y)

    #Calculathing relative error
    rel_error = np.abs(abs_error/modeuler_y)

    #Return Y calculated for Modified Euler also return absolute and relative
    return modeuler_y,abs_error,rel_error
#End Function

#Function of Third Order's Method
def third(f,t,n,h,y0,original_y):
    #Defining vector of zeros 
    third_order_y = np.zeros(n)

    #Defining y(0) 
    third_order_y[0] = y0

    #Defining Runge-Kulta's method of Third Order
    for i in range(n-1):
        # Third Order
        K1 = f(t[i],third_order_y[i])
        K2 = f(t[i]+(h/2),third_order_y[i] + (h/2) * K1)
        K3 = f(t[i] + h, third_order_y[i] + 2*h*K2 - h*K1)
        third_order_y[i+1] = third_order_y[i] + (h/6)*(K1 + 4*K2 + K3)
    #End
    
    #Defining absolute error 
    abs_error = np.abs(third_order_y - original_y)

    #Calculathing relative error
    rel_error = np.abs(abs_error/third_order_y)

    #Return Y calculated for Third Order also return absolute and relative
    return third_order_y,abs_error,rel_error
#End Function

#Function of Fourth Order's Method
def fourth(f,t,n,h,y0,original_y):
    #Defining vector of zeros 
    fourth_order_y = np.zeros(n)

    #Defining y(0) 
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
   
    #Defining absolute error 
    abs_error = np.abs(fourth_order_y - original_y)

    #Calculathing relative error
    relative_error = np.abs(abs_error/fourth_order_y)

    #Return Y calculated for Third Order also return absolute and relative
    return fourth_order_y,abs_error,relative_error
#End Function 

#Defining initial values
y0, t0, tf, h = 1, 0, 1, 0.2
t = np.linspace(t0, tf, round((tf-t0)/h)+1)
n = len(t)

#Defining my ODE: cos(2t) + sin(3t)
f = lambda t,y: np.cos(2*t) + np.sin(3*t)

#Defining my Analytic ODE: (1/2)sin(2t) - (1/3)cos(3t) +(4/3)
analytic_ODE = lambda t: (1/2)*np.sin(2*t) - (1/3)*np.cos(3*t) +(4/3)

#Solving the Analytic ODE
exact_y = analytic_ODE(t)

#Calling Euler 
euler_values,euler_abs,euler_rel = euler(f,t,n,h,y0,exact_y)

#Calling Modified Euler
modeuler_values,modeuler_abs,modeuler_rel = modeuler(f,t,n,h,y0,exact_y)

#Calling Third Order
third_values,third_abs,third_rel = third(f,t,n,h,y0,exact_y)

#Calling Fourth Order
fourth_values,fourth_abs,fourth_rel = fourth(f,t,n,h,y0,exact_y)

#Creating dictionary mod: Complete
dcc = {
    't': t,
    'y': exact_y,
    'Euler': euler_values,
    'Mod Euler': modeuler_values,
    'Third': third_values,
    'Fourth': fourth_values,
    'Euler Abs': euler_abs,
    'Euler Rel': euler_rel,
    'Mod Euler Abs': modeuler_abs,
    'Mod Euler Rel': modeuler_rel,
    'Third Abs': third_abs,
    'Third Rel': third_rel,
    'Fourth Abs': fourth_abs,
    'Fourth Rel': fourth_rel
}

#Creating dictionary mod: Final Value
dcf = {
    't': t[-1],
    'y': exact_y[-1],
    'Euler': euler_values[-1],
    'Mod Euler': modeuler_values[-1],
    'Third': third_values[-1],
    'Fourth': fourth_values[-1],
    'Euler Abs': euler_abs[-1],
    'Euler Rel': euler_rel[-1],
    'Mod Euler Abs': modeuler_abs[-1],
    'Mod Euler Rel': modeuler_rel[-1],
    'Third Abs': third_abs[-1],
    'Third Rel': third_rel[-1],
    'Fourth Abs': fourth_abs[-1],
    'Fourth Rel': fourth_rel[-1]
}

#Setting at Dataframe
pd.set_option('display.float_format', lambda x: f'{x:.8f}')
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

#Create Dataframe of Complete
dfc = pd.DataFrame(dcc) 

#Create Dataframe of Final Value
dff = pd.DataFrame(dcf,index=[0])

#Output Dataframe Complete
print(dfc)

#Output Dataframe Final Value
print(dff)

#Output Third - Fourth, to show the difference of precision
print(f'Difference betwen Third and Fourth is :{np.abs(third_values[-1]-fourth_values[-1]):.16e}')

#Comparison graphic mode: Complete
plt.plot(t,exact_y,'k-',linewidth=3,label='Exact')
plt.plot(t,euler_values,'bo--',linewidth=1.5,label=' Euler ')
plt.plot(t,modeuler_values,'ro--',linewidth=1.5,label=' Mod Euler ')
plt.plot(t,third_values,'go--',linewidth=2,label=' Third Order ')
plt.plot(t,fourth_values,'yo--',linewidth=1.5,label=' Fourth Order ')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Runge-Kuttas methods x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()

#Comparison graphic mode: Final Value
names = ['Euler', 'Mod Euler', 'Third', 'Fourth']
vals = [euler_values[-1], modeuler_values[-1], third_values[-1], fourth_values[-1]]

plt.figure()
plt.axvline(exact_y[-1], color='k', linestyle='--', label='Exact')
plt.plot(vals, names, 'o', markersize=10)

plt.xlabel('y(1)')
plt.title('Values of methods at t = 1 vs exact solution')
plt.legend()
plt.grid(True)
plt.show()
