# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:01:42 2018

@author: labuser
"""

# A simple 1D model for the phase dependent ionization and recombination in
# static fields

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint, quad
import numba
import turning_and_binding as tab
import timeit


# ==========
# utilities
# ==========
def atomic_units():
    """Return a dictionary of atomic units ['GHz', 'mVcm', 'ns']"""
    au = {'GHz': 1.51983e-7, 'mVcm': 1.94469e-13, 'ns': 4.13414e7}
    return au


def progress(source, i, total):
    """print an updating report of 'source: i/total'

    non-keyword arguments:
    source -- str: name of source function
    i -- int: current iteration (starts from 0)
    total --- int: total number of iterations
    """
    # start on fresh line
    if i == 0:
        print()
    # progress
    print("\r{0}: {1} / {2}".format(source, i+1, total), end="\r")
    # newline if we've reached the end.
    if i+1 == total:
        print()
    return
# ==========


# ==========
# MW Energy interactions
# ==========
@numba.jit
def dWi(phi, Emw, omega_mw):
    """Energy exchange from leaving the core for the first time."""
    return (3./2.) * (Emw / (omega_mw**(2./3.))) * math.cos(phi - np.pi/6)


@numba.jit
def dWs(phi, Emw, fmw):
    """Returns energy exchange (a.u.) from slingshotting the core.

    non-keyword arguments:
    phi -- float: MW phase (rad.) when e- return to core.
    Emw -- float: MW field amplitude (a.u.)
    fmw -- float: MW frequency (rad, NOT cycles!) (a.u.)
    """
    return np.sqrt(3) * (3./2.) * (Emw / (fmw**(2./3.))) * np.cos(phi)
# ==========


# ==========
# Orbits
# ==========
@numba.jit
def lookup_tt_f(W, Ep):
    Wint = int(W*1e8)
    if (Wint <= 15198) and (Wint >= -9119):
        tt = lut_up_f[Wint]
    elif Wint < -9119:  # below Wlim
        tt = np.NaN
    elif Wint > 15198:  # max of lut
        tt = tab.tt_up(W, Ep)
    return tt


@numba.jit
def dW_orbit(W0, Ep, Emw, fmw, t0, tstop, Wlim):
    """Use dWs and tt to calculate orbits until tstop

    Starting from an e- just returning to the core after a classical orbit,
    change the energy by the slingshot energy exchange dWs(phi, Emw, fmw), and
    then calculate the new turning time. Repeat until an orbit will return
    after tstop (usually 20 ns). If the e- energy falls below Wlim, stop
    and return it as the final energy. This saves execution time and cuts off
    the simulation when it stops making physical sense.

    non-keyword arguments:
    W0 -- float: e- Energy (a.u.) when starting the orbit
    Ep -- float: Static field strenght (a.u.)
    Emw -- float: MW field amplitude (a.u.)
    fmw -- float: MW frequency (rad, NOT cycles!) (a.u.)
    t0 -- starting time (a.u.)
    tstop -- stopping time (a.u.)
    Wlim -- cutoff e- energy (a.u.), -600 GHz means tt(W, 0) ~ 1/(15.9 GHz)

    returns - obs -- dict:
        'n' -- number of orbits
        'ti' -- start time of last orbit
        'Wi' -- start energy of last orbit
        'tf' -- ending time of last orbit
        'Wf' -- final energy of last orbit
    """
    # fist orbit
    n = 0
    ti = t0
    Wi = np.NaN
    Wf = W0
    # Wint = int(Wf*1e8)
    # if (Wint <= 15198) and (Wint >= -9119):
    #     tt = lut_up_f[Wint]
    # elif Wint < -9119:  # below Wlim
    #     tt = np.NaN
    # elif Wint > 15198:  # max of lut
    #     tt = tab.tt_up(Wf, Ep)
    tt = lookup_tt_f(Wf, Ep)
    tf = t0 + tt
    # MW exchange and orbit
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = np.random.random()*2*np.pi  # random phase
        Wf = Wi + dWs(phi, Emw, fmw)  # new energy
        # orbit
        # tt = tab.tt_up(Wf, Ep)
        # Wint = int(Wf*1e8)
        # if (Wint <= 15198) and (Wint >= -9119):
        #     tt = lut_up_f[Wint]
        # elif Wint < -9119:  # below Wlim
        #     tt = np.NaN
        # elif Wint > 15198:  # max of lut
        #     tt = tab.tt_up(Wf, Ep)
        tt = lookup_tt_f(Wf, Ep)
        tf = ti + 2*tt
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    return obs


def dW_orbit_track(W0, Ep, Emw, fmw, t0, tstop, Wlim):
    """Returns last obs, list of obs after each orbit"""
    track = pd.DataFrame()
    # fist orbit
    n = 0
    ti = t0
    Wi = np.NaN
    Wf = W0
    tt = tab.tt_up(Wf, Ep)
    tf = t0 + tt
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    track = track.append(obs, ignore_index=True)
    # MW exchange and orbit
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = np.random.random()*2*np.pi  # random phase
        Wf = Wi + dWs(phi, Emw, fmw)  # new energy
        # orbit
        # tt = tab.tt_up(Wf, Ep)
        Wint = int(Wf*1e8)
        if (Wint <= 15198) and (Wint >= -9119):
            tt = lut_up_f[Wint]
        elif Wint < -9119:  # below Wlim
            tt = np.NaN
        elif Wint > 15198:  # max of lut
            tt = tab.tt_up(Wf, Ep)
        tf = ti + 2*tt
        obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        track = track.append(obs, ignore_index=True)
    return obs, track


def catch_at_20ns(W, y, ti):
    """When field turns off, how long does e- take to return to the core.

    If r is NaN, then previous steps exited because W < Wlim, tf = NaN.
    Otherwise, determine if e- is outgoing or returning. If returning,
    tf is the time it takes to get from zi to r (trf). If outgoing, if W >= 0
    then never returns and tf = NaN. If W < 0, then e- returns at
    ti + tt + (tt - trf) = ti + 2*tt - trf

    non-keyword arguments:
    W -- float: e- energy (a.u.)
    y -- array: [radius, velocity] e- initial state (a.u.)
    ti -- starting time (a.u.), should be 20 ns

    returns: tf -- float: time (a.u.) when e- returns to the core.
    """
    [r, v] = y  # unpack
    if np.isnan(r):  # exited because W < Wfinal
        t = np.NaN  # indicate this is the final energy
    else:
        zi = math.copysign(6, r)  # get zi on the correct side.
        # how long does it take to get from core to current position
        trf = quad(tab.intg_up, zi, r, args=(W, 0))[0]
        if (r > 0) != (v > 0):  # r and v different signs, e- returning
            t = trf
        elif (r > 0) == (v > 0):  # r and v same signs, e- outgoing
            if W >= 0:  # e- never returns
                t = np.NaN
            else:
                tt = tab.tt_up(W, 0)  # turning time in zero field
                t = 2*tt - trf  # has to turn, and then come back to r
    # au = atomic_units()
    # rstring = ("catch():\n" +
    #            "W = {0} GHz\t" + "ti = {3} ns\n" +
    #            "r = {1} a.u.\t" + "v = {2} a.u.\n" +
    #            "trf = {4} ns\t" + "t = {5} ns\t" + "tf = {6} ns")
    # rstring = rstring.format(
    #         W/au['GHz'], r, v, ti/au['ns'], trf/au['ns'], t/au['ns'],
    #         (t + ti)/au['ns'])
    # print(rstring)
    return t + ti  # start time plus elapsed time


@numba.jit
def dW_orbit_Epoff(W0, Emw, fmw, t0, tstop, Wlim, tring):
    """Orbits and energy exchange w/o field as MW rings down.

    Starting from an e- just returning to the core after the static turns off,
    change the energy by the slingshot energy exchange dWs(phi, Emw, fmw), with
    Emw modified acounting for the ringdown. Then calculate the new turning
    time. Repeat until an orbit will return after tstop (usually 20 ns). If
    the e- energy falls below Wlim, stop and return it as the final energy.
    This saves execution time and cuts off the simulation when it stops making
    physical sense.

    non-keyword arguments:
    W0 -- float: e- Energy (a.u.) when starting the orbit
    Emw -- float: MW field amplitude (a.u.)
    fmw -- float: MW frequency (rad, NOT cycles!) (a.u.)
    t0 -- starting time (a.u.)
    tstop -- stopping time (a.u.)
    Wlim -- cutoff e- energy (a.u.), -600 GHz means tt(W, 0) ~ 1/(15.9 GHz)

    returns - obs -- dict:
        'n' -- number of orbits
        'ti' -- start time of last orbit
        'Wi' -- start energy of last orbit
        'tf' -- ending time of last orbit
        'Wf' -- final energy of last orbit
    """
    # set up start
    n = 0
    ti = np.NaN
    Wi = np.NaN
    tf = t0
    Wf = W0
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = np.random.random()*2*np.pi  # random phase
        Emw_t = Emw*math.exp(-ti/tring)  # ringing down
        Wf = Wi + dWs(phi, Emw_t, fmw)  # new energy
        # orbit
        # tt = tab.tt_up(Wf, 0)  # field is turned off
        Wint = int(Wf*1e8)
        if Wint < 0:
            tt = lut_up[Wint]
        else:
            tt = np.NaN
        tf = ti + 2*tt
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    return obs


def dW_orbit_Epoff_track(W0, Emw, fmw, t0, tstop, Wlim, tring):
    """Returns last obs, list of obs after each orbit"""
    # set up start
    n = 0
    ti = np.NaN
    Wi = np.NaN
    tf = t0
    Wf = W0
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    track = pd.DataFrame(obs, index=[0])
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = np.random.random()*2*np.pi  # random phase
        Emw_t = Emw*math.exp(-ti/tring)  # ringing down
        Wf = Wi + dWs(phi, Emw_t, fmw)  # new energy
        # orbit
        tt = tab.tt_up(Wf, 0)  # field is turned off
        tf = ti + 2*tt
        obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        track = track.append(obs, ignore_index=True)
    return obs, track
# ==========


# ==========
# Path integration
# ==========
@numba.jit
def derivative(y, t, Ep):
    """Return dydt of y=[r,v] for given field Ep (a.u.)"""
    [r, v] = y
    dydt = [v, 1/r**2 + Ep]
    return dydt


def path_integration(r0, v0, t, Ep):
    """Path integration field Ep (a.u.)

    non-keyword arguments:
    r0 -- float: initial radius (a.u.)
    v0 -- float: initial velocity (a.u.)
    t -- array: times to solve for, usually just [t0, tfinal] (a.u.)
    Ep -- float: static field (a.u.)

    returns y -- e- final state [r, v]
    """
    y0 = [r0, v0]
    y = odeint(derivative, y0, t, args=(Ep,))
    return y


def int_path(W0, Ep, ti, tstop):
    """Integrate e- from the core to final position at stopping time.

    If an electron's orbit is deemed to end after tstop, start from the core
    and integrate the path from ti to tstop, and return the e- final state.

    non-keyword arguments:
    W0 -- float: initial e- energy (a.u.)
    Ep -- float: static field (a.u.)
    ti -- float: starting time (a.u.)
    tstop -- float: stopping time (a.u.)

    returns:
    t -- array: array of times to integrate [ti, tstop]
    y -- array: final e- state [r, v]
    """
    r0 = -6
    if W0 + 1/abs(r0) >= 0:
        v0 = -math.sqrt(2*(W0 + 1/abs(r0)))
    else:
        v0 = 0
        r0 = -abs(1/W0)
    # t = np.linspace(ti, tstop, 2)
    t = [ti, tstop]
    y = path_integration(r0, v0, t, Ep)
    return t, y


def int_path_track(W0, Ep, ti, tstop):
    """Returns path with 100 points."""
    r0 = -6
    if W0 + 1/abs(r0) >= 0:
        v0 = -np.sqrt(2*(W0 + 1/abs(r0)))
    else:
        v0 = 0
        r0 = -abs(1/W0)
    t = np.linspace(ti, tstop, 100)
    y = path_integration(r0, v0, t, Ep)
    return t, y


@numba.jit
def y_to_w(y):
    """Return energy (a.u.) of e- based on y = [r, v]."""
    return -1/abs(y[0]) + y[1]**2/2


@numba.jit
def y_to_w_vec(y):
    """Return energy (a.u.) of e- based on y = [r, v]. W and y are vectors."""
    return -1/np.abs(y[:, 0]) + np.power(y[:, 1], 2)/2
# ==========


# ==========
# Full execution
# ==========
def run_to_20ns(W0, Ep, Emw, fmw, t0, tstop, Wlim):
    """Runs model from t0 to tstop, returns final energy.

    Given start energy and conditions, first run dW_orbit until an orbit ends
    after tstop. Then extract starting time and final energy of last orbit. If
    not below Wlim, use int_path to integrate to the final state of the e- at
    tstop, and use y_to_w() to extract final energy. If below Wlim, use that
    as the final energy.

    non-keyword arguments:
    W0 -- float: e- Energy (a.u.) when starting the orbit
    Ep -- float: Static field strenght (a.u.)
    Emw -- float: MW field amplitude (a.u.)
    fmw -- float: MW frequency (rad, NOT cycles!) (a.u.)
    t0 -- starting time (a.u.)
    tstop -- stopping time (a.u.)
    Wlim -- cutoff e- energy (a.u.), -600 GHz means tt(W, 0) ~ 1/(15.9 GHz)

    returns: Wfinal -- Final e- energy (a.u.)
    """
    # MW exchange and orbit
    obs = dW_orbit(W0, Ep, Emw, fmw, t0, tstop, Wlim)
    t = obs['ti']
    W = obs['Wf']
    if W > Wlim:
        t, y = int_path(W, Ep, t, tstop)  # returns start & stop
        y = y[-1]  # only the stop point
        Wfinal = y_to_w(y)
    else:
        Wfinal = W
        t, y = [np.NaN, np.NaN], [np.NaN, np.NaN]
    return Wfinal, y


def run_to_20ns_track(W0, Ep, Emw, fmw, t0, tstop, Wlim):
    """Keeps list of every orbit and the path integration."""
    # MW exchange and orbit
    obs, track = dW_orbit_track(W0, Ep, Emw, fmw, t0, tstop, Wlim)
    t = obs['ti']
    W = obs['Wf']
    if W > Wlim:
        t, y = int_path_track(W, Ep, t, tstop)
        Wfinal = y_to_w_vec(y)  # small time savings
    else:
        Wfinal = [W, W]
        t, y = [np.NaN, np.NaN], [[np.NaN, np.NaN]]
    return Wfinal, y, t, track


def run_after_20ns(W0, y0, Emw, fmw, t0, tstop, Wlim, tring):
    tc = catch_at_20ns(W0, y0, t0)
    obs = dW_orbit_Epoff(W0, Emw, fmw, tc, tstop, Wlim, tring)
    Wfinal = obs['Wf']
    return Wfinal, obs


def run_after_20ns_track(W0, y0, Emw, fmw, t0, tstop, Wlim, tring):
    """Keep lists of every orbit"""
    tc = catch_at_20ns(W0, y0, t0)
    # if tc < tstop and W0 > Wlim:
    obs, track = dW_orbit_Epoff_track(W0, Emw, fmw, tc, tstop, Wlim, tring)
    Wfinal = obs['Wf']
    # else:
    #     Wfinal = W0
    return Wfinal, tc, track


def run_to_stop(W0, Ep, Emw, fmw, t0, toff, tstop, Wlim, tring):
    Wfinal, y = run_to_20ns(W0, Ep, Emw, fmw, t0, toff, Wlim)
    Wfinal, obs = run_after_20ns(W0, y, Emw, fmw, toff, tstop, Wlim, tring)
    return Wfinal
# ==========


# ==========
# Timing
# ==========
def time_run_to_20ns(n):
    """execute run_to_20ns n with random W0, Ep. Plot histogram of times.

    Build n-length random arrays of W0 from -42 to 42 GHz and Ep from 0 to
    300 mV/cm. Build an array of run times for run_to_20ns, and then plot a
    histogram of the time distributions. Return the array of times.

    returns trun -- array: run times for run_to_20ns
    """
    au = atomic_units()
    trun = []
    # n = 10
    W0s = (np.random.random(n)*2 - 1)*42*au['GHz']
    Eps = np.random.random(n)*300*au['mVcm']
    Emw = 4*1000*au['mVcm']
    fmw = 2*np.pi*15.932/au['ns']
    t0 = 0*au['ns']
    tstop = 20*au['ns']
    Wlim = -600*au['GHz']
    for i in range(n):
        progress("time_run_to_20ns()", i, n)
        clock_start = time.clock()
        run_to_20ns(W0s[i], Eps[i], Emw, fmw, t0, tstop,
                    Wlim)
        clock_end = time.clock()
        trun = trun + [clock_end - clock_start]
    bins = np.linspace(0, max(trun)*1.2, 1001)
    print('mean = {} s'.format(np.mean(trun)))
    print('median = {} s'.format(np.median(trun)))
    fig, ax = plt.subplots()
    ax.hist(trun, bins)
    ax.set_title("{0} runs of 'run_to_20ns()'".format(n))
    plt.show()
    return trun


def time_run_to_stop(n):
    au = atomic_units()
    trun = []
    # n = 10
    W0s = (np.random.random(n)*3 - 2)*150*au['GHz']  # -100 -> 50 GHz
    # Eps = np.random.random(n)*300*au['mVcm']
    Eps = np.array([100*au['mVcm']]*n)
    Emw = 4*1000*au['mVcm']
    fmw = 2*np.pi*15.932/au['ns']
    t0 = 0*au['ns']
    toff = 20*au['ns']
    tring = 10*au['ns']
    tstop = toff + 5*tring
    Wlim = -600*au['GHz']
    global lut_up_f
    global lut_down_f
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Eps[0])
    for i in range(n):
        progress("time_run_to_stop()", i, n)
        clock_start = time.clock()
        run_to_stop(W0s[i], Eps[i], Emw, fmw, t0, toff, tstop, Wlim, tring)
        clock_end = time.clock()
        trun = trun + [clock_end - clock_start]
    bins = np.linspace(0, np.mean(trun)*2, float(n)/10)
    print('mean = {} s'.format(np.mean(trun)))
    print('median = {} s'.format(np.median(trun)))
    fig, ax = plt.subplots()
    ax.hist(trun, bins)
    ax.set_title("{0} runs of 'run_to_stop()'".format(n))
    plt.show()
    return trun


def timeit_run_to_stop():
    W0 = (np.random.random()*3 - 2)*150*1.51983e-7
    # Ep = np.random.random()*300*1.94469e-13
    Ep = 0
    Emw = 4*1000*1.94469e-13
    fmw = 2*np.pi*15.932/41341400.0
    t0 = 0*41341400.0
    toff = 20*41341400.0
    tring = 10*41341400.0
    tstop = toff + 5*tring
    Wlim = -600*1.51983e-7
    run_to_stop(W0, Ep, Emw, fmw, t0, toff, tstop, Wlim, tring)
    return


def timeit_run_to_20ns():
    W0 = (np.random.random()*3 - 2)*150*1.51983e-7
    Ep = np.random.random()*300*1.94469e-13
    Emw = 4*1000*1.94469e-13
    fmw = 2*np.pi*15.932/41341400.0
    t0 = 0*41341400.0
    toff = 20*41341400.0
    # tring = 10*41341400.0
    # tstop = toff + 5*tring
    Wlim = -600*1.51983e-7
    run_to_20ns(W0, Ep, Emw, fmw, t0, toff, Wlim)
    return
# ==========


# ==========
# Plotting
# ==========
def plot_run_to_20ns_track():
    """Plot the energy vs. time from each step in the model."""
    au = atomic_units()
    # ==========
    # Model
    # ==========
    # run_to_20ns_track() arguments
    W0 = (np.random.random()*3 - 2)*150*au['GHz']
    Ep = 100*au['mVcm']
    Emw = 4*1000*au['mVcm']
    fmw = 2*np.pi*15.932/au['ns']
    t0 = 0*au['ns']
    tstop = 20*au['ns']
    Wlim = -600*au['GHz']
    # lut
    global lut_up_f
    global lut_down_f
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Ep)
    # run to 20ns
    Wfinal, y, t, track = run_to_20ns_track(W0, Ep, Emw, fmw, t0, tstop, Wlim)
    # convert to plottable format
    # track -- Group 0
    df = track[['ti', 'Wi']].copy()
    df.rename(index=int, columns={'ti': 't', 'Wi': 'W'}, inplace=True)
    shift = track[['ti', 'Wf']].copy()
    shift.rename(index=int, columns={'ti': 't', 'Wf': 'W'}, inplace=True)
    shift['t'] = shift['t'] + 1  # 10 as shfit
    df = df.append(shift, ignore_index=True)
    df['group'] = 0
    df.sort_values(by='t', inplace=True)
    # integrated path -- Group 1
    df = df.append(pd.DataFrame({'t': t, 'W': Wfinal, 'group': 1}),
                   ignore_index=True)
    # catch -- Group 2
    # tf = catch_at_20ns(Wfinal[-1], y[-1], t[-1])
    # df = df.append(pd.DataFrame({'t': np.array([t[-1], tf]), 'W': Wfinal[-1],
    #                              'group': 2}), ignore_index=True)
    # orbits after Ep turns off -- Group 3
    W0 = Wfinal[-1]
    y0 = y[-1]
    t0 = t[-1]
    tring = 10*au['ns']  # guess at ringdown
    tstop = t0 + 5*tring  # very little MW left
    Wfinal, tc, track = run_after_20ns_track(W0, y0, Emw, fmw, t0, tstop, Wlim,
                                             tring)
    df = df.append(pd.DataFrame({'t': np.array([t0, tc]), 'W': W0,
                                 'group': 2}), ignore_index=True)
    tdf = track[['ti', 'Wi']].copy()
    tdf.rename(index=int, columns={'ti': 't', 'Wi': 'W'}, inplace=True)
    shift = track[['ti', 'Wf']].copy()
    shift.rename(index=int, columns={'ti': 't', 'Wf': 'W'}, inplace=True)
    shift['t'] = shift['t'] + 1  # 10 as shift
    tdf = tdf.append(shift, ignore_index=True)
    tdf['group'] = 3
    tdf.sort_values(by='t', inplace=True)
    df = df.append(tdf, ignore_index=True)
    # lab units
    df['t'] = df['t']/au['ns']
    df['W'] = df['W']/au['GHz']
    # ==========
    # Plot
    # ==========
    fig, ax = plt.subplots()
    ax.axvline(0, c='k')
    ax.axhline(0, c='k')
    df.where(df['group'] == 0).plot(x='t', y='W', label='dW', ax=ax, c='C0',
                                    marker='o')
    df.where(df['group'] == 3).plot(x='t', y='W', label='dWo', ax=ax, c='C3',
                                    marker='o')
    df.where(df['group'] == 2).plot(x='t', y='W', label='catch', ax=ax, c='C2',
                                    marker='o')
    df.where(df['group'] == 1).plot(x='t', y='W', label='intg', ax=ax, c='C1',
                                    marker='.')
    return df
# ==========


def main():
    # au = atomic_units()
    global lut_up
    global lut_down
    lut_up, lut_down = tab.import_lookup_table()
    # result = plot_run_to_20ns_track()
    result = time_run_to_stop(int(1e4))
    # n = 1.0e4
    # t = timeit.repeat(timeit_run_to_20ns, repeat=3, number=int(n))
    # print(np.array(t) / n)
    # plot_run_to_20ns_track()
    # f = 300*au['mVcm']
    # lut_up_f, lut_down_f = tab.import_lookup_table_f(f)
    # Ws = np.arange(-100, 100, 1)*au['GHz']
    # Wints = list((Ws*1e8).astype(int))
    # print(Wints)
    # tts = []
    # for Wint in Wints:
    #     tts = tts + [lut_up_f[Wint]]
    # tts = np.array(tts)
    # plt.plot(Ws/au['GHz'], tts/au['ns'])
    return result


if __name__ == '__main__':  # run if script is called directly
    result = main()
