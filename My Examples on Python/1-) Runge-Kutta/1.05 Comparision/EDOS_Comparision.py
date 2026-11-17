import numpy as np
import matplotlib.pyplot as plt
import time 
import pandas as pd

""" The goat this code is to compare perfomance of methods, about 
Euler, Modified Euler, Third Order and Fourth Order """

# Function of Euler's Method
def euler(f,t,n,h,original_y):
    #Defining zeros vector
    euler_y = np.zeros(n)

    #Defining y(0) = -1
    euler_y[0] = -1

    #Defining Euler's method
    t0 = time.time() # initial time
    for i in range(n-1):
    # Euler's method
        euler_f = f(t[i],euler_y[i])
        euler_y[i+1] = euler_y[i] + h * euler_f
    #End
    t1 = time.time() # end time

    #Calculathing time to solve
    r_time = t1 - t0

    #Defining absolute error
    abs_error = np.abs(euler_y - original_y)

    #Calculathing relative error
    relative_error = []
    for i in range(n):
        rel_error = np.abs((abs_error[i]/euler_y[i])) * 100
        relative_error.append(rel_error)
    #End

    return euler_y,abs_error,relative_error,r_time

#Function of Modified Euler's Method
def modeuler(f,t,n,h,original_y):
    #Defining vector of zeros 
    modeuler_y = np.zeros(n)

    #Defining y(0) = -1
    modeuler_y[0] = -1

    #Defining modified Euler's method 
    t0 = time.time() # initial time
    for i in range(n-1):
        # modified Euler's method
        K1 = f(t[i],modeuler_y[i])
        K2 = f(t[i+1],modeuler_y[i] + h*K1)
        modeuler_y[i+1] = modeuler_y[i] + (h/2) * ( K1 + K2) 
    #End 
    t1 = time.time() # end time

    #Calculathing time to solve
    r_time = t1 - t0

    #Defining absolute error 
    abs_error = np.abs(modeuler_y - original_y)

    #Calculathing relative error
    relative_error = []
    for i in range(n):
        rel_error = np.abs((abs_error[i]/modeuler_y[i])) * 100
        relative_error.append(rel_error)
    #End

    return modeuler_y,abs_error,relative_error,r_time

#Function of Third Order's Method
def third(f,t,n,h,original_y):
    #Defining vector of zeros 
    third_order_y = np.zeros(n)

    #Defining y(0) = -1
    third_order_y[0] = -1

    #Defining Runge-Kulta's method of Third Order
    t0 = time.time() # initial time
    for i in range(n-1):
        # Third Order
        K1 = f(t[i],third_order_y[i])
        K2 = f(t[i]+(h/2),third_order_y[i] + (h/2) * K1)
        K3 = f(t[i] + h, third_order_y[i] + 2*h*K2 - h*K1)
        third_order_y[i+1] = third_order_y[i] + (h/6)*(K1 + 4*K2 + K3)
    #End
    t1 = time.time() # end time

    #Calculathing time to solve
    r_time = t1 - t0

    #Defining absolute error 
    abs_error = np.abs(third_order_y - original_y)

    #Calculathing relative error
    relative_error = []
    for i in range(n):
        rel_error = np.abs((abs_error[i]/third_order_y[i])) * 100
        relative_error.append(rel_error)
    #End

    return third_order_y,abs_error,relative_error,r_time

#Function of Fourth Order's Method
def fourth(f,t,n,h,original_y):
    #Defining vector of zeros 
    fourth_order_y = np.zeros(n)

    #Defining y(0) = -1
    fourth_order_y[0] = -1

    #Defining Runge-Kulta's method of Fourth Order
    t0 = time.time() # initial time
    for i in range(n-1):
        # Fourth Order
        K1 = f(t[i],fourth_order_y[i])
        K2 = f(t[i]+(h/2),fourth_order_y[i] + (h/2) * K1)
        K3 = f(t[i]+(h/2),fourth_order_y[i] + (h/2) * K2)
        K4 = f(t[i] + h, fourth_order_y[i] + h*K3)
        fourth_order_y[i+1] = fourth_order_y[i] + (h/6)*(K1 + 2*K2 + 2*K3 + K4)
    #End
    t1 = time.time() # end time

    #Calculathing time to solve
    r_time = t1 - t0

    #Defining absolute error 
    abs_error = np.abs(fourth_order_y - original_y)

    #Calculathing relative error
    relative_error = []
    for i in range(n):
        rel_error = np.abs((abs_error[i]/fourth_order_y[i])) * 100
        relative_error.append(rel_error)
    #End

    return fourth_order_y,abs_error,relative_error,r_time


#Defining initials values 
h, tf, t0= 0.1, 0.5, 0
t = np.arange(t0,tf+h,h)
n = len(t)

#Defining f(t,y) = -2t - y
f = lambda t,y: -2*t - y

#Defining exact solve -3e⁻t - 2t + 2
exact_y = lambda t: -3*np.exp(-t) -2*t + 2

#Defining analytic values 
original_y = exact_y(t)

#Calling Functions
euler_values,euler_abs,euler_rel,euler_process = euler(f,t,n,h,original_y)
modeuler_values,modeuler_abs,modeuler_rel,modeuler_process = modeuler(f,t,n,h,original_y)
third_values,third_abs,third_rel,third_process = third(f,t,n,h,original_y)
fourth_values,fourth_abs,fourth_rel,fourth_process = fourth(f,t,n,h,original_y)

#Creating dictionary 
dc = {
    't': t,
    'Analytic solution': original_y,
    'Euler': euler_values,
    'Modified Euler': modeuler_values,
    'Third Order': third_values,
    'Fourth Order': fourth_values,
    'Euler Abs': euler_abs,
    'Euler Rel': euler_rel,
    'Mod Euler Abs': modeuler_abs,
    'Mod Euler Rel': modeuler_rel,
    'Third Abs': third_abs,
    'Third Rel': third_rel,
    'Fourth Abs': fourth_abs,
    'Fourth Rel': fourth_rel
}

#Creating DataFrame
df = pd.DataFrame(dc)

#Print of Methods processing time 
print(f'Time of Euler: {euler_process} \t Time of mod Euler: {modeuler_process} \t Time of Third Order: {third_process} \t Time of Fourth Order: {fourth_process} \t') 

print(df)

#Comparison graphic
plt.plot(t,original_y,'k-',linewidth=3,label='Exact')
plt.plot(t,euler_values,'bo--',linewidth=1.5,label=' Euler ')
plt.plot(t,modeuler_values,'ro--',linewidth=1.5,label=' Mod Euelr ')
plt.plot(t,third_values,'go--',linewidth=1.5,label=' Third Order ')
plt.plot(t,fourth_values,'yo--',linewidth=1.5,label=' Fourth Order ')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Runge-Kultas methods x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
