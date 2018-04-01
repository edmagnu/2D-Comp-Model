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
import time
from scipy.integrate import quad, odeint
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


def stat_rep(n, ti, Wi, tf, Wf):
    au = atomic_units()
    rs = "n = {0}\t"
    rs = rs + "ti = {1} ns\t"
    rs = rs + "Wi = {2} GHz\t"
    rs = rs + "tf = {3} ns\t"
    rs = rs + "Wf = {4}  GHz"
    rs = rs.format(n, *np.round([ti/au['ns'], Wi/au['GHz'], tf/au['ns'],
                                 Wf/au['GHz']], 4))
    print(rs)
    return rs


def dW_orbit(W0, Ep, Emw, fmw, t0, tstop):
    orbits = pd.DataFrame()  # hold orbit info
    # set up start
    n = 0
    tf = t0
    Wf = W0
    while tf <= tstop:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = random.random()*2*np.pi  # random phase
        Wf = Wi + dWs(phi, Emw, fmw)  # new energy
        # orbit
        tt = tt_up(Wf, Ep)[0]
        tf = ti + tt
        # stat_rep(n, ti, Wi, tf, Wf)
        obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        orbits = orbits.append(obs, ignore_index=True)
    return orbits


def derivative(y, t, Ep):
    [r, v] = y
    dydt = [v, 1/r**2 + Ep]
    return dydt


def path_integration(r0, v0, t, Ep):
    y0 = [r0, v0]
    # print(y0)
    y = odeint(derivative, y0, t, args=(Ep,))
    return y


def int_path(W0, Ep, ti, tstop):
    # au = atomic_units()
    r0 = -6
    if W0 + 1/abs(r0) >= 0:
        v0 = -np.sqrt(2*(W0 + 1/abs(r0)))
    else:
        v0 = 0
        r0 = -abs(1/W0)
    t = np.linspace(ti, tstop, 10)
    y = path_integration(r0, v0, t, Ep)
    return t, y


def y_to_w(y):
    W = -1/np.abs(y[:, 0]) + np.power(y[:, 1], 2)/2
    return W


def orbits_plottable(orbits, fmw):
    au = atomic_units()
    Wpath = pd.DataFrame({'t': orbits['ti'], 'W': orbits['Wi']})
    Wpath = Wpath.append(pd.DataFrame(
            {'t': orbits['ti'] + 0.5/(fmw*au['ns']), 'W': orbits['Wf']}))
    Wpath = Wpath.append({'t': orbits.iloc[-1]['tf'],
                          'W': orbits.iloc[-1]['Wf']},
                         ignore_index=True)
    Wpath.sort_values(by='t', inplace=True)
    return Wpath


def main():
    clock_start = time.time()
    au = atomic_units()
    W0 = 1*au['GHz']
    Ep = 10*au['mVcm']
    Emw = 4*1000*au['mVcm']
    fmw = 2*np.pi*15.932/au['ns']
    t0 = 0*au['ns']
    tstop = 20*au['ns']
    # MW exchange and orbit
    orbits = dW_orbit(W0, Ep, Emw, fmw, t0, tstop)
    orbits.sort_values(by='ti', inplace=True)
    # integrate to t = 20 ns
    t = orbits.iloc[-1]['ti']
    W = orbits.iloc[-1]['Wf']
    # print(t/au['ns'], W/au['GHz'])
    t, y = int_path(W, Ep, t, tstop)
    # plot
    orbits[['ti', 'tf']] = orbits[['ti', 'tf']]/au['ns']
    orbits[['Wi', 'Wf']] = orbits[['Wi', 'Wf']]/au['GHz']
    # fig, ax = plt.subplots()
    # Wpath = orbits_plottable(orbits, fmw)
    # Wpath = Wpath[:-1]
    Worbit = y_to_w(y)
    clock_end = time.time()
    print("Number of orbits = {}".format(len(orbits)))
    print("Final Energy W = {} GHz".format(np.round(Worbit[-1]/au['GHz'], 2)))
    print("Runtime = {} ms".format(np.round(clock_end - clock_start, 2)))
    # Wpath = Wpath.append({'t': t[-1]/au['ns'], 'W': Worbit[-1]/au['GHz']},
    #                      ignore_index=True)
    # Wpath.plot(x='t', y='W', color='C0', marker='o', ax=ax)
    # ax.plot(t/au['ns'], Worbit/au['GHz'], marker='.', color='C1')
    # ax.set(xlabel='time (ns)', ylabel='Energy (GHz)')
    return


main()
