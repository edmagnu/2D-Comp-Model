# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:01:42 2018

@author: labuser
"""

# A simple 1D model for the phase dependent ionization and recombination in
# static fields

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
# import os
import random


def atomic_units():
    """Return a dictionary of atomic units"""
    au = {'GHz': 1.51983e-7, 'mVcm': 1.94469e-13, 'ns': 4.13414e7}
    return au


def dWi(phi, Emw, omega_mw):
    """Energy exchange from leaving the core for the first time."""
    return (3./2.) * (Emw / (omega_mw**(2./3.))) * np.cos(phi - np.pi/6)


def dWs(phi, Emw, fmw):
    """Energy exchange from slingshotting the core."""
    return np.sqrt(3) * (3./2.) * (Emw / (fmw**(2./3.))) * np.cos(phi)


def integrand_up(z, W, f):
    return -1/np.sqrt(2*(W - 1/z + f*z))


def tt_up(W, f):
    # au = atomic_units()
    zi = -6  # uphill
    if f == 0 and W < 0:  # above limit with field
        # print("Below limit without field")
        zt = 1/W  # turning position, W + -1/z = 0
        tt = quad(integrand_up, zi, zt, args=(W, f))  # integrator
    elif f == 0 and W >= 0:
        # print("Above limit without field")
        # Never turns
        zt = np.NaN
        tt = (np.NaN, np.NaN)
    elif f > 0:
        # print("With field")
        zt = -1/(2*f) * (W + np.sqrt(W**2 + 4*f))  # W + -1/z - fz = 0
        tt = quad(integrand_up, zi, zt, args=(W, f))  # integrator
    else:
        print("tt_uphill({0}, {1}) : something went wrong".format(W, f))
    # print(zt, tt[0]/au['ns'])
    return tt


def stat_rep(n, t, W):
    au = atomic_units()
    print("n = {0}\tt = {1} ns\tW = {2} GHz".format(
            n, t/au['ns'], W/au['GHz']))
    return


def main():
    au = atomic_units()
    # fname = os.path.join("..", "Turning Time", "data_raw.txt")
    # data = pd.read_csv(fname, index_col=0)
    # dataframe
    # initial conditions
    n = 0  # counter
    Wi = -3*au['GHz']  # starting energy
    print(n, Wi/au['GHz'])
    Ep = 2*au['mVcm']  # pulsed field
    Emw = 4*1e3*au['mVcm']  # MW field
    fmw = 2*np.pi*15.932/au['ns']  # MW frequency
    phi = 4*np.pi/6  # launch MW phase
    ti = 0  # launch time
    tf = 20*au['ns']  # stop time
    stat_rep(n, ti, Wi)
    obs = {'n': n, 't': ti, 'W': Wi}
    orbits = pd.DataFrame(obs, index=[n])
    # execution
    # initial energy exchange
    n = n + 1
    W = Wi + dWi(phi, Emw, fmw)
    # orbit
    tt = tt_up(W, Ep)[0]
    t = ti + tt
    stat_rep(n, t, W)
    obs = {'n': n, 't': t, 'W': W}
    orbits = orbits.append(obs, ignore_index=True)
    while t <= tf:
        n = n + 1
        phi = random.random()*2*np.pi  # random phase
        W = W + dWs(phi, Emw, fmw)  # new energy
        # orbit
        tt = tt_up(W, Ep)[0]
        t = t + tt
        stat_rep(n, t, W)
        obs = {'n': n, 't': t, 'W': W}
        orbits = orbits.append(obs, ignore_index=True)
    # tt = tt_up(-10*au['GHz'], 0*au['mVcm'])
    return orbits


au = atomic_units()
orbits = main()
orbits['W'] = orbits['W']/au['GHz']
orbits['t'] = orbits['t']/au['ns']
# au = atomic_units()
# phi = np.linspace(0, 2*np.pi, 1001)
# Emw = 4000*au['mVcm']
# wmw = 2*np.pi*15.932/au['ns']
# omega_mw = 2*np.pi*fmw
# print(np.sqrt(3)*3/2*Emw/(wmw**(2/3))/au['GHz'])
# print((1/au['GHz']) * (3./2.) * Emw/(np.power(wmw, 2/3)))
