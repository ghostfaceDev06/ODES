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
fourth_order_y = np.zeros(n)

#Defining y(0) = -1
fourth_order_y[0] = -1

#Defining Runge-Kulta's method of Fourth Order
for i in range(n-1):
    # Fourth Order
    K1 = f(t[i],fourth_order_y[i])
    K2 = f(t[i]+(h/2),fourth_order_y[i] + (h/2) * K1)
    K3 = f(t[i]+(h/2),fourth_order_y[i] + (h/2) * K2)
    K4 = f(t[i] + h, fourth_order_y[i] + h*K3)
    fourth_order_y[i+1] = fourth_order_y[i] + (h/6)*(K1 + 2*K2 + 2*K3 + K4)
#End

#Defining analytic values
original_values = exact_y(t)

#Defining absolute error 
abs_error = np.abs(fourth_order_y - original_values)

#Calculathing relative error
relative_error = []
for i in range(n):
    rel_error = np.abs((abs_error[i]/fourth_order_y[i])) * 100
    relative_error.append(rel_error)
#End

#Print the results
for ti,foy,oy,aerror,rerror in zip(t,fourth_order_y,original_values,abs_error,relative_error):
    print(f't = {ti:.2f}\tFourth Order = {foy:.4f}\tExactY = {oy:.4f}\tAbsolute Error = {aerror:.4f}\tRelative Error = {rerror:.4f}%')
#End

#Comparison graphic
plt.plot(t,original_values,'k-',linewidth=2,label='Exact')
plt.plot(t,fourth_order_y,'ro--',linewidth=1.5,label=' Fourth Order ')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Runge-Kultas method of Fourth Order x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
