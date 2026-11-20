import numpy as np
import matplotlib.pyplot as plt

#Defining initial values 
h, tf , t0 = 0.1 , 0.5, 0
t = np.arange(t0,tf + h,h)
n = len(t)

#Defining f(t,y) = -2t - y 
f = lambda t,y: -2*t - y

#Defining exact solution: -3e⁻t - 2t + 2
exact_y = lambda t: -3*np.exp(-t) - 2*t + 2

#Defining vector of zeros 
modeuler_y = np.zeros(n)

#Defining y(0) = -1
modeuler_y[0] = -1

#Defining modified Euler's method 
for i in range(n-1):
    # modified Euler's method
    K1 = f(t[i],modeuler_y[i])
    K2 = f(t[i+1],modeuler_y[i] + h*K1)
    modeuler_y[i+1] = modeuler_y[i] + (h/2) * ( K1 + K2) 
#End 

#Defining analytic values
original_values = exact_y(t)

#Defining absolute error 
abs_error = np.abs(modeuler_y - original_values)

#Calculathing relative error
relative_error = []
for i in range(n):
    rel_error = np.abs((abs_error[i]/modeuler_y[i])) * 100
    relative_error.append(rel_error)
#End

#Print the results
for ti,mey,oy,aerror,rerror in zip(t,modeuler_y,original_values,abs_error,relative_error):
    print(f't = {ti:.2f}\tmodEulerY = {mey:.4f}\tExactY = {oy:.4f}\tAbsolute Error = {aerror:.4f}\tRelative Error = {rerror:.4f}%')
#End

#Comparison graphic
plt.plot(t,original_values,'k-',linewidth=2,label='Exact')
plt.plot(t,modeuler_y,'ro--',linewidth=1.5,label=' modified Euler')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: modified Euler´s Method x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
