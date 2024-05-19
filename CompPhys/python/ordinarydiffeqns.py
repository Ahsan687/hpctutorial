#!/usr/bin/python

import numpy as ny
from time import time
import gc

#######################################################################

class ODESolvers(object):
    """ ODE solvers. """
    ############################################################
    ############################################################
    def __init__(self):
        pass

    ############################################################
    def cfd_step(self,fim1,f1i,h):
        return fim1 + 2.0*h*f1i

    ############################################################
    def cfd_step_order2(self,fim1,fi,f2i,h):
        return 2.0*fi - fim1 + h**2*f2i

    ############################################################

    def solve_cfd_order2(self,x0,f0,fp0,xN,N,func_f2,**kwargs):
        """Expect func to act as func(x,f,f1,**kwargs)."""
        # THIS IS ALSO LEAP-FROG
        if N < 1:
            raise ValueError("At least one step needed. Increase N.")
        i = 1
        soln = ny.zeros(N,dtype=float)
        soln[0] = f0
        h = (xN - x0)/(1.0*N)
        xi = x0 + h
        fim1 = 1.0*f0  
        fi = f0 + h*fp0 + 0.5*h**2*func_f2(x0,f0,fp0,**kwargs)
        while i < N:
            soln[i] = fi
            temp = 1.0*fi
            fpi = (fi - fim1)/h
            fi = 2*fi - fim1 + h**2*func_f2(xi,fi,fpi,**kwargs)
            fim1 = 1.0*temp
            xi += h 
            i += 1

        return soln

#######################################################################

class DerivativeFunctions(object):
    """ Derivative functions. """
    ############################################################
    ############################################################
    def __init__(self):
        pass

    ############################################################
    def func_f1_bernoulli_example(self,x,f):
        return 2.0*x**3*f**4 - f/x

    ############################################################
    def func_f2_sho(self,x,f,f1,omega=1.0,gamma=0.3):
        """ 2nd deriv for (under-damped) simple harmonic oscillator.
             x -> indep variable
              f -> dep variable
            f1 -> 1st deriv of dep variable
            omega = frequency; gamma = damping coefficient
        """
        # check for over-damped
        return -1.0*(omega**2*f + 2.0*gamma*f1)

    ############################################################

#######################################################################


if __name__=="__main__":
    import matplotlib.pyplot as plt

    odes = ODESolvers()
    derivs = DerivativeFunctions()

    NPHI = 3
    phi = NPHI*ny.pi/2
    A = 1.0
    omega = 1.0
    gamma = 0.002
    f0 = A*ny.cos(phi)
    fp0 = -A*omega*ny.sin(phi)
    x0 = 0.0
    xN = 80*ny.pi/omega
    func = derivs.func_f2_sho

    N = 1000
    x = ny.linspace(x0,xN,N)
    exact = A*ny.exp(-gamma*x)*ny.cos(ny.sqrt(omega**2-gamma**2)*x+phi)
    YMIN = -3.0*A
    YMAX = 3.0*A
    plt.ylim([YMIN,YMAX])
    plt.xlim([x0,xN])
    plt.text(x0 + 0.05*(xN-x0),YMIN + 0.9*(YMAX-YMIN),
             "$A = {0:.2f}\\,;\\, \\phi = {1:d}\\times\\,\\pi/2$".format(A,NPHI),fontsize=16)
    plt.plot(x,exact,'--',c='k',label="$\\rm exact$")
    plt.minorticks_on()
    hcrit = 2.0/omega
    Ncrit = ny.rint((xN-x0)/hcrit).astype(int)
    Nvals = [Ncrit,Ncrit*2,Ncrit*4,Ncrit*16]
    col = ['g','r','magenta','b']
    for n in range(len(Nvals)):
        N = Nvals[n]
        soln = odes.solve_cfd_order2(x0,f0,fp0,xN,N,func,omega=omega,gamma=gamma)
        x = ny.linspace(x0,xN,N)
        h = x[1]-x[0]
        # LABEL = "$\\Delta t/\\Delta t_{{\\rm crit}} = {0:.3f}$".format(h/hcrit)
        LABEL = "$\\omega \\Delta t = {0:.3f}$".format(omega*h)
        plt.plot(x,soln,'-',c=col[n],label=LABEL)
    plt.axhline(A,ls=':',c='k')
    plt.axhline(-A,ls=':',c='k')
    plt.legend(loc='lower right')
    plt.savefig('phasetest.png')
    plt.close()

