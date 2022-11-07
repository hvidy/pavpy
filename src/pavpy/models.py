import numpy as np
import scipy.special as sp


def ud(x,theta):
    radtheta = theta*4.8481368e-9
    z = np.pi * radtheta * x
    return (2*sp.jv(1,z)/z)**2

def ld(x,theta,u):
    radtheta = theta*4.8481368e-9
    z = np.pi * radtheta * x
    return (((1-u)/2 + u/3)**(-1) *((1-u)*sp.jv(1,z)/z + u*(np.pi/2)**(0.5)*sp.jv(1.5,z)/z**1.5))**2