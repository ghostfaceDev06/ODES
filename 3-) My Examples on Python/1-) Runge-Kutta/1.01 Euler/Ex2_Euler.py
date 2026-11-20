import numpy as np
import matplotlib.pyplot as plt

#Example two utilized in classroom

#Defining initials values 
deltat, tf, t0= 0.2, 1, 0
t = np.arange(t0,tf+deltat,deltat)
n = len(t)

#Defining symbols variables

#Defining f(t,y) = 2t²y²
f = lambda t,y: 2*(t**2)*(y**2)

#Defining exact solve 1/((-2/3)t³+1)
exact_y = lambda t: 1/((-2/3)*(t**3)+1)

#Defining zeros vector
euler_y = np.zeros(n)

#Defining y(0) = 1
euler_y[0] = 1

#Defining Euler's method
for i in range(n-1):
    # Euler's method
    euler_f = f(t[i],euler_y[i])
    euler_y[i+1] = euler_y[i] + deltat * euler_f
#End 

#Defining analytic values 
u = exact_y(t)

#Defining absolute error
abs_error = np.abs(euler_y - u)

#Calculathing relative error
relative_error = []
for i in range(n):
    rel_error = np.abs((abs_error[i]/euler_y[i])) * 100
    relative_error.append(rel_error)

#Print the results 
for ti,ey,oy,aerror,rerror in zip(t,euler_y,u,abs_error,relative_error):
    print(f't = {ti:.2f}\tEulerY = {ey:.4f}\tExactY = {oy:.4f}\tAbsolute Error = {aerror:.4f}\tRelative Error = {rerror:.4f}%')
#End

# Comparison Graphic 
plt.plot(t,u,'k-',linewidth=2,label='Exata')
plt.plot(t,euler_y,'ro--',linewidth=1.5,label='Euler')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Euler´s Method x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
