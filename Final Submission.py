#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
import concurrent.futures

class DrivenDampedPendulumAnalyser:
        
    '''
    
    This analyser can be used to study the chaotic behaviour of a Driven-Damped Pendulum.
    We produce: 
    - Phasespace Plots
    - Poincare Maps
    - Bifurcation plots (with and without multiprocessing)
    - Lyapunov exponents
    
    
    '''
        
    
    def __init__(self, damping = 0.5, driving_amplitude = 1, driving_frequency = 2/3, x0 = -np.pi/2, v0 = 0, Poincare = False):
        
        '''
        
        Initialises the class with default settings for our Driven-Damped Pendulum
        
        
        '''
        
        self.damping = damping
        self.driving_amplitude = driving_amplitude
        self.driving_frequency = driving_frequency
        self.x0 = x0
        self.v0 = v0
        self.Poincare = Poincare
        
    def SolveDDP(self, damping, driving_amplitude, driving_frequency, Poincare, x0, v0):
        
        '''
        
        Solves the Driven-Damped ODE by converting it into 2 coupled 1st order ODEs. Numerically solved using the
        default RK4 solver solve_ivp offers as this is optimal for our system.
        
        
        '''
        #converting to two 1st order coupled ODEs
        def dSdt(t,S):
            x, v = S[0], S[1]
            return[v, - np.sin(x) - self.damping*v + self.driving_amplitude*np.cos(self.driving_frequency*t)]

        #number of time steps
        if self.Poincare:
            t_end = 100000
            dt = 2*np.pi/(self.driving_frequency)
        elif not self.Poincare:
            t_end = 1000
            dt = 2*np.pi/(self.driving_frequency*1000)
        else:
            warnings.warn('Poincare takes boolean values, default set to False')
            t_end = 1000
            dt = 2*np.pi/(self.driving_frequency*1000)

        dt = 2*np.pi/(self.driving_frequency*100)
        n_steps = t_end/dt

        t = np.arange(0, t_end, dt)

        #solving ODE for position and velocity
        sol = solve_ivp(fun = dSdt, t_span = [t[0], t[-1]], y0 = [self.x0, self.v0], t_eval = t)

        #converting solutions from degrees to radians
        np.deg2rad(sol.y[0])
        np.deg2rad(sol.y[1])
        #angle wrapping [-pi, pi)
        sol.y[0] = (sol.y[0] + np.pi) % (2 * np.pi ) - np.pi
        sol.y[1] = (sol.y[1] + np.pi) % (2 * np.pi ) - np.pi

        return sol.y[0], sol.y[1]
        
    def PhaseSpacePlot(self,X,Y):
        
        '''
        
        Produces a phase space plot when the two arrays Position and Velocity are input
        
        
        '''
        if self.Poincare:
            t_end = 100000
            dt = 2*np.pi/(self.driving_frequency)
            n_steps = t_end/dt
        elif not self.Poincare:
            t_end = 1000
            dt = 2*np.pi/(self.driving_frequency*1000)
            n_steps = t_end/dt
        else:
            warnings.warn('Poincare takes boolean values, default set to False')
            t_end = 1000
            dt = 2*np.pi/(self.driving_frequency*1000)
            n_steps = t_end/dt
        
        #PhaseSpace/PoincareMap plot
        plt.plot(self.X[int(n_steps*0.25):], self.Y[int(n_steps*0.25):], ',', c = 'black')
        plt.xlabel('$\\theta$ (rad)')
        plt.ylabel('$\\omega$ (rad/s)')
        
        return plt.show()
        
    def Bifurcation(self, damping, driving_frequency):
        
        '''
        
        Produces a bifurcation using the for loop. Single processing is faster for smaller data sets than multiprocessing.
        
        
        '''
        #number of time steps
        t_end = 100000
        dt = 2*np.pi/(driving_frequency)
        n_steps = t_end/dt

        #array of independent variables of the bifurcation plot
        _driving_amplitude_array = np.arange(0.001, 1.5, 0.001)

        #iterating over the independent variable and plotting Poincare map data
        for i in range(len(_driving_amplitude_array)):
            self.SolveDDP(self.damping, self.driving_frequency, _driving_amplitude_array[i], Poincare = True)
            X = np.full(len(self.SolveDDP[1]),_driving_amplitude_array[i])
            plt.plot(X[int(n_steps*0.25):], self.SolveDDP[1][int(n_steps*0.25):], '.', markersize = 0.125, c = 'black')
            plt.xlabel('Driving Amplitude (rad/s^2)')
            plt.ylabel('$\\omega$ (rad/s)')
        
        return plt.show()
        
    def BifurcationPOOL(self, damping, driving_frequency):
        
        '''
        
        Produces a bifurcation diagram using the concurrent.futures module which bypasses the global lock interpreter
        python was developed with. All CPUs are used and multiple functions run concurrently which reduces time
        complexity.
        
        
        '''
        #number of time steps
        t_end = 100000
        dt = 2*np.pi/(driving_frequency)
        n_steps = t_end/dt

        #Redefining SolveDDP to work for ProcessPoolExecutor, new function takes only 1 argument     
        def SolveDDP_Pool(_driving_amplitude):
            Y = self.SolveDDP(self.damping, self.driving_frequency, _driving_amplitude, Poincare = True)[1]
            X = np.full(len(Y),_driving_amplitude)
            
            return X,Y
            
        #exeecutor assigns tasks to each CPU automatically, functions run concurrently
        if __name__ == '__main__':
            with concurrent.futures.ProcessPoolExecutor() as executor:
                _driving_amplitude_array = np.arange(0.001, 2, 0.001)
                results = executor.map(SolveDDP_Pool, _driving_amplitude_array)

                #executes return command and plots data
                for result in results:
                    print(result)
                    plt.plot(result[0][int(n_steps*0.25):], result[1][int(n_steps*0.25):], '.', markersize = 0.125, c = 'black')
                    plt.xlabel('Driving Amplitude (rad/s^2)')
                    plt.ylabel('$\\omega$ (rad/s)')
                
        return plt.show()
    
    def Lyapunov(self, x0, v0):
        
        '''
        
        Calculates the Lyapunov exponent of any two initial points for the default Driven Damped Pendulum
        
        
        '''
        #define the difference in distance value and the time step
        epsilon = 0.001
        dt = 2*np.pi/(self.driving_frequency*1000)
        
        #define two points very close to eachother
        x_close = self.x0 + epsilon
        v_close = self.v0 + epsilon
        
        point1_initial = np.array((self.x0, self.v0))
        point2_initial = np.array((x_close, v_close))
        
        #calculate absolute distance between two points
        dist_initial = np.linalg.norm(point1_initial, point2_initial)
        
        #find where points are final iteration
        sols0 = self.SolveDDP(self.x0, self.v0)
        sols_close = self.SolveDDP(x_close, v_close)
        
        point1_final = np.array((sols0[0][-1], sols0[1][-1]))
        point2_final = np.array((sols_close[0][-1], sols_close[1][-1]))
        
        #distance at final iteration
        dist_final = np.linalg.norm(point1_final, point2_final)
        
        #Lyapunov exponent value
        lyapunov = np.log(dist_final/dist_initial)/(len(sols0.y[0])*dt)
        
        return lyapunov


# In[9]:





# In[ ]:




