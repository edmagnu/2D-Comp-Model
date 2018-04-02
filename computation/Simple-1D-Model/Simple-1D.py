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
# import random


def atomic_units():
    """Return a dictionary of atomic units"""
    au = {'GHz': 1.51983e-7, 'mVcm': 1.94469e-13, 'ns': 4.13414e7}
    return au


def progress(source, i, total):
    """print an updating report of 'source: i/total'"""
    # start on fresh line
    if i == 0:
        print()
    # progress
    print("\r{0}: {1} / {2}".format(source, i+1, total), end="\r")
    # newline if we've reached the end.
    if i+1 == total:
        print()
    return


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
    # orbits = pd.DataFrame()  # hold orbit info
    # set up start
    n = 0
    tf = t0
    Wf = W0
    while tf <= tstop:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = np.random.random()*2*np.pi  # random phase
        Wf = Wi + dWs(phi, Emw, fmw)  # new energy
        # orbit
        tt = tt_up(Wf, Ep)[0]
        tf = ti + 2*tt
        # stat_rep(n, ti, Wi, tf, Wf)
        # obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        # orbits = orbits.append(obs, ignore_index=True)
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    return obs


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
    t = np.linspace(ti, tstop, 1)
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


def run_to_20ns(W0, Ep, Emw, fmw, t0, tstop):
    # MW exchange and orbit
    clock0 = time.time()
    orbits = dW_orbit(W0, Ep, Emw, fmw, t0, tstop)
    clock1 = time.time()
    # orbits.sort_values(by='ti', inplace=True)
    # integrate to t = 20 ns
    # t = orbits.iloc[-1]['ti']
    # W = orbits.iloc[-1]['Wf']
    t = orbits['ti']
    W = orbits['Wf']
    t, y = int_path(W, Ep, t, tstop)
    clock2 = time.time()
    Wfinal = y_to_w(np.array([y[-1]]))  # small time savings
    clock3 = time.time()
    return Wfinal, [clock0, clock1, clock2, clock3]


def time_run_to_20ns():
    au = atomic_units()
    trun = []
    n = 10
    W0s = (np.random.random(n)*2 - 1)*42*au['GHz']
    Eps = np.random.random(n)*300*au['mVcm']
    Emw = 4*1000*au['mVcm']
    fmw = 2*np.pi*15.932/au['ns']
    t0 = 0*au['ns']
    tstop = 20*au['ns']
    timings = [[]]
    for i in range(n):
        progress("time_run_to_20ns()", i, n)
        clock_start = time.time()
        Wfinal, timing = run_to_20ns(W0s[i], Eps[i], Emw, fmw, t0, tstop)
        clock_end = time.time()
        trun = trun + [clock_end - clock_start]
        timing = np.diff(timing)
        timings = timings + [timing]
    bins = np.linspace(0, max(trun)*1.2, 1001)
    print('mean = {} s'.format(np.mean(trun)))
    print('median = {} s'.format(np.median(trun)))
    plt.hist(trun, bins)
    plt.xlabel("time (s)")
    plt.ylabel("count")
    plt.title("{0} runs of 'run_to_20ns()'".format(n))
    return


def time_tt_up():
    au = atomic_units()
    n = 1000
    Ws = (np.random.random(n)*2 - 1)*42*au['GHz']
    Eps = np.random.random(n)*300*au['mVcm']
    trun = []
    for i in range(n):
        progress("time_tt_up()", i, n)
        # f = random.random()*300*au['mVcm']
        # W = (random.random()*2 - 1)*42*au['GHz']
        clock_start = time.time()
        tt_up(Ws[i], Eps[i])
        clock_end = time.time()
        trun = trun + [clock_end - clock_start]
    bins = np.linspace(0, max(trun)*1.2, 1001)
    print('mean = {} s'.format(np.mean(trun)))
    print('median = {} s'.format(np.median(trun)))
    plt.hist(trun, bins)
    plt.xlabel("time (s)")
    plt.ylabel("count")
    plt.title("{0} runs of 'tt_up()'".format(n))
    return trun


def main():
    au = atomic_units()
    W = 1000*au['GHz']
    f = 300*au['mVcm']
    tt = tt_up(W, f)[0]/au['ns']
    print(tt)
    return

trun, timings = time_run_to_20ns()
# trun = time_tt_up()
# main()
