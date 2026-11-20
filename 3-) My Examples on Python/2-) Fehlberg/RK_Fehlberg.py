import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

#Fehlberg Runge-Kutta
def fehlberg(f,t0,tf,w0,tol,hmax,hmin):

    #First pass : Defining initial values 
    t = t0
    w = w0
    h = hmax
    FLAG = 1
    
    t_hist = [t0] #where keep t to long algorithm 
    w_hist = [w0] #where keep w to long algorithm and also numeric solve this ODE
    h_hist = [h] #where keep h to long algorithm 
    mission = True #Check this algorithm is Success or Failed.

    #Second pass: Looping the algorithm 
    while FLAG == 1:

        #Third pass: Calculing operations this method 
        K1 = h*f(t,w)
        K2 = h*f(t + h/4,w + K1/4)
        K3 = h*f(t + 3*h/8,w + 3*K1/32 + 9*K2/32)
        K4 = h*f(t + 12*h/13,w + 1932*K1/2197 - 7200*K2/2197 + 7296*K3/2197)
        K5 = h*f(t + h,w + 439*K1/216 - 8*K2 + 3680*K3/513 - 845*K4/4104 )
        K6 = h*f(t + h/2,w - 8*K1/27 + 2*K2 - 3544*K3/2565 + 1859*K4/4104 - 11*K5/40) 

        #Fourth pass: Calculing R
        R = 1/h * abs(K1/360 -128*K3/4275 - 2197*K4/75240 + K5/50 + 2*K6/55)

        #Fifth pass: Check if R <= tol pouted 
        if R <= tol:
            #Sixth pass: change values of t and w
            t += h
            w += 25*K1/216 + 1408*K3/2565 + 2197*K4/4104 - K5/5

            #Seventh pass: add elements h, t and w on array
            h_hist.append(h)
            t_hist.append(t)
            w_hist.append(w)
        #End if

        #Eighth pass: Calculing Delta
        delta = 0.84*(tol/R)**(1/4)

        #Nineth pass 
        if delta < 0.1: #if delta < 0.1 , change h
            h = 0.1*h
        #End if
        
        elif delta >= 4.0: #if delta >= 4, change h
            h = 4.0*h
        #End elif

        else: #else nothing other options, change h.
            h = delta*h 
        #End else

        #Tenth pass: Check if h > hmax, change h
        if h > hmax:
            h = hmax
        #End if

        #Eleventh pass
        if t >= tf: #if t >= tf , finish algorithm with success 
            FLAG = 0
            mission = True
        #End if

        elif t + h > tf: #if t + h > tf , change h 
            h = tf - t
        #End elif

        elif h < hmin: # if h < hmin, so algorithm finish with failed.
            FLAG = 0
            mission = False
            print('h in excess')
        #End elif
    #End while        

    # Return h,t and w utilized to algorithm
    return h_hist,w_hist,t_hist,mission
#End function
        
#ODE our problem: y - t² + 1
f = lambda t,y: y - t**2 + 1

#Analytic ODE: (t + 1)² - 0.5e^t
exact_y = lambda t: (t+1)**2 - 0.5*np.exp(t)

#Defining values to a(t0),b(tf), initial condition(w0), max h(hmax) and min h(hmin).
tf, t0 , w0 , hmax , hmin , tol = 2, 0, 0.5 , 0.25, 0.01, 1e-5

#Calling function
h_hist,w_hist,t_hist,mission = fehlberg(f,t0,tf,w0,tol,hmax,hmin) 

#To transform in array
t_arr = np.array(t_hist)
w_arr = np.array(w_hist)
h_arr = np.array(h_hist)

#Calculing y this ODE
original_y = exact_y(t_arr)
n = len(t_arr)

#Calculing Absolute and relative error
abs_error = np.abs(w_arr - original_y) 
rel_error = np.abs(abs_error/abs(w_arr))

#Create dictionary
dc = {
    'h': h_arr,
    't': t_arr,
    'Exact': original_y,
    'Fehlberg': w_arr,
    'Abs': abs_error,
    'Rel': rel_error
}

#Create DatFrame
df = pd.DataFrame(dc)

#Output
print(df)

# Comparison Graphic 
plt.plot(t_arr,original_y,'k-',linewidth=2,label='Exact')
plt.plot(t_arr,w_arr,'ro--',linewidth=1.5,label='Fehlberg')

plt.xlabel('Time(t)')
plt.ylabel('Value of y')
plt.title('Comparation: Fehlberg Runge-Kutta x Exact Solve')
plt.legend(loc='best')
plt.grid(True)

plt.show()
