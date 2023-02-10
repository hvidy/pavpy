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

def ldd_claret4(x,theta,a1,a2,a3,a4):
    radtheta = theta*4.8481368e-9
    z = np.pi * radtheta * x

    c = (0.5*(1-(a1+a2+a3+a4)) + 2*a1/5 + a2/3 + 2*a3/7 + a4/4)
    vis = ((1-(a1+a2+a3+a4))*sp.jv(1,z)/z + 
        a1*(2**0.25)*sp.gamma(1.25)*sp.jv(1.25,z)/z**1.25 +
        a2*(2**0.5)*sp.gamma(1.5)*sp.jv(1.5,z)/z**1.5 +
        a3*(2**0.75)*sp.gamma(1.75)*sp.jv(1.75,z)/z**1.75 +
        a4*2*sp.jv(2,z)/z**2)

    return (vis/c)**2


def ellipse(x,a,b,phi):
    return np.sqrt(2)*a*b/np.sqrt((b**2-a**2)*np.cos(2*x-2*phi)+a**2+b**2)