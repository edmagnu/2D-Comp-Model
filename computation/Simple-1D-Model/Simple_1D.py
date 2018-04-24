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
from scipy.optimize import check_grad
import numba
import turning_and_binding as tab
import random
# import timeit


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
# up/down not relevant, maybe =+/-dWi() for up/down
@numba.jit
def dWi(phi, Emw, omega_mw):
    """Energy exchange from leaving the core for the first time."""
    return (3./2.) * (Emw / (omega_mw**(2./3.))) * math.cos(phi - math.pi/6)


# up/down not relevant, maybe =+/-dWi() for up/down
@numba.jit
def dWs(phi, Emw, fmw):
    """Returns energy exchange (a.u.) from slingshotting the core.

    non-keyword arguments:
    phi -- float: MW phase (rad.) when e- return to core.
    Emw -- float: MW field amplitude (a.u.)
    fmw -- float: MW frequency (rad, NOT cycles!) (a.u.)
    """
    return math.sqrt(3) * (3./2.) * (Emw / (fmw**(2./3.))) * math.cos(phi)
# ==========


# ==========
# Orbits
# ==========
# add dup to pick lut_up/down
@numba.jit
def lookup_tt(W, dup, lut_up, lut_down):
    """For after 20ns, turning time from lut_up/down"""
    Wint = int(W*1e8)
    if (Wint < 0) and (Wint >= -9119) and dup:
        tt = lut_up[Wint]
    elif (Wint < 0) and (Wint >= -9119) and not dup:
        tt = lut_down[Wint]
    elif (Wint < -9119) or (Wint >= 0):  # below Wlim
        tt = math.nan
    return tt


# add dup to conditions to pick between tt_up/down, lut_up/down_f
@numba.jit  # doesn't like nested if elif
def lookup_tt_f(W, Ep, dup, lut_up_f, lut_down_f):
    """For before 20ns, turning time from lut_up/down_f."""
    Wint = int(W*1e8)
    if (Wint <= 15198) and (Wint >= -9119) and dup:
        tt = lut_up_f[Wint]
    elif (Wint <= 15198) and (Wint >= -9119) and not dup:
        tt = lut_down_f[Wint]
    elif Wint < -9119:  # below Wlim
        tt = math.nan
    elif (Wint > 15198) and dup:  # max of lut
        tt = tab.tt_up(W, Ep)
    elif (Wint > 15198) and not dup:
        tt = tab.tt_down(W, Ep)
    return tt


# only need up/down for tt, added dup to lookup_tt_f call
# @numba.jit
def dW_orbit(W0, Ep, Emw, fmw, t0, tstop, Wlim, dup,
             lut_up_f, lut_down_f, tr=False):
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
    t0 -- float: starting time (a.u.)
    tstop -- float: stopping time (a.u.)
    Wlim -- float: cutoff e- energy (a.u.), -600 GHz -> tt ~ 1/(15.9 GHz)
    dup -- Bool: True/False means up/down hill e-

    returns: obs -- dict: final orbit state:
        'n' -- number of orbits
        'ti' -- start time of last orbit
        'Wi' -- start energy of last orbit
        'tf' -- ending time of last orbit
        'Wf' -- final energy of last orbit
    """
    # fist orbit
    n = 0
    ti = t0
    Wi = math.nan
    Wf = W0
    tt = lookup_tt_f(Wf, Ep, dup, lut_up_f, lut_down_f)
    tf = t0 + tt
    if tr is True:
        track = pd.DataFrame()
        obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        track = pd.DataFrame(obs, index=[0])
    # MW exchange and orbit
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = random.random()*2*math.pi  # random phase
        Wf = Wi + dWs(phi, Emw, fmw)  # new energy
        # orbit
        tt = lookup_tt_f(Wf, Ep, dup, lut_up_f, lut_down_f)
        tf = ti + 2*tt
        if tr is True:
            obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
            track = track.append(obs, ignore_index=True)
    if tr is False:
        obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        return obs
    else:
        return obs, track


# lookup_tt_f added dup
def dW_orbit_track(W0, Ep, Emw, fmw, t0, tstop, Wlim, dup):
    """Returns last obs, list of obs after each orbit"""
    track = pd.DataFrame()
    # fist orbit
    n = 0
    ti = t0
    Wi = math.nan
    Wf = W0
    tt = lookup_tt_f(Wf, Ep, dup)
    tf = t0 + tt
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    track = track.append(obs, ignore_index=True)
    # MW exchange and orbit
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = random.random()*2*math.pi  # random phase
        Wf = Wi + dWs(phi, Emw, fmw)  # new energy
        # orbit
        tt = lookup_tt_f(Wf, Ep, dup)
        tf = ti + 2*tt
        obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        track = track.append(obs, ignore_index=True)
    return obs, track


# integrand intg_up/down and tt_up/down
def catch_at_20ns(W, y, ti, dup):
    """When field turns off, how long does e- take to return to the core.

    If r is NaN, then previous steps exited because W < Wlim, tf = NaN.
    Otherwise, determine if e- is outgoing or returning. If returning,
    tf is the time it takes to get from zi to r (trf). If outgoing, if W >= 0
    then never returns and tf = NaN. If W < 0, then e- returns at
    ti + tt + (tt - trf) = ti + 2*tt - trf

    non-keyword arguments:
    W -- float: e- energy (a.u.)
    y -- [float, float]: [radius, velocity] e- initial state (a.u.)
    ti -- float: starting time (a.u.), should be 20 ns
    dup -- Bool: True/False means up/down hill e-

    returns: tf -- float: time (a.u.) when e- returns to the core.
    """
    [r, v] = y  # unpack
    if math.isnan(r):  # exited because W < Wfinal
        t = math.nan  # indicate this is the final energy
    else:
        zi = math.copysign(6, r)  # get zi on the correct side.
        # how long does it take to get from core to current position
        if dup is True:
            trf = quad(tab.intg_up, zi, r, args=(W, 0))[0]
        else:
            trf = quad(tab.intg_down, zi, r, args=(W, 0))[0]
        # returning or not returning?
        if (r > 0) != (v > 0):  # r and v different signs, e- returning
            t = trf
        elif (r > 0) == (v > 0):  # r and v same signs, e- outgoing
            # Leaving, or up/downhill turning time?
            if W >= 0:  # e- never returns
                t = math.nan
            elif dup is True:
                tt = tab.tt_up(W, 0)  # turning time in zero field
                t = 2*tt - trf  # has to turn, and then come back to r
            elif dup is False:
                tt = tab.tt_down(W, 0)  # turning time in zero field
                t = 2*tt - trf  # has to turn, and then come back to r
            else:
                print("catch_at_20ns() A: Something went wrong.")
        else:
            print("catch_at_20ns() B: Something went wrong.")
    return t + ti  # start time plus elapsed time


# need dup for lookup table
def dW_orbit_Epoff(W0, Emw, fmw, t0, tstop, Wlim, tring, dup,
                   lut_up, lut_down, tr=False):
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
    t0 -- float: starting time (a.u.)
    tstop -- float: stopping time (a.u.)
    Wlim -- float: cutoff e- energy (a.u.), -600 GHz is tt ~ 1/(15.9 GHz)
    dup -- Bool: True/False means up/down hill e-

    returns - obs -- dict:
        'n' -- number of orbits
        'ti' -- start time of last orbit
        'Wi' -- start energy of last orbit
        'tf' -- ending time of last orbit
        'Wf' -- final energy of last orbit
    """
    # set up start
    n = 0
    ti = math.nan
    Wi = W0
    tf = t0
    Wf = W0
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    if tr is True:
        track = pd.DataFrame(obs, index=[0])
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = random.random()*2*math.pi  # random phase
        Emw_t = Emw*math.exp(-ti/tring)  # ringing down
        Wf = Wi + dWs(phi, Emw_t, fmw)  # new energy
        # orbit
        tt = lookup_tt(Wf, dup, lut_up, lut_down)
        tf = ti + 2*tt
        if tr is True:
            obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
            track = track.append(obs, ignore_index=True)
    if tr is False:
        obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
        return obs
    else:
        return obs, track


# need dup for lookup table
def dW_orbit_Epoff_track(W0, Emw, fmw, t0, tstop, Wlim, tring, dup):
    """Returns last obs, list of obs after each orbit"""
    # set up start
    n = 0
    ti = math.nan
    Wi = W0
    tf = t0
    Wf = W0
    obs = {'n': n, 'ti': ti, 'Wi': Wi, 'tf': tf, 'Wf': Wf}
    track = pd.DataFrame(obs, index=[0])
    while tf <= tstop and Wf > Wlim:
        # new conditions
        n = n + 1
        ti = tf
        Wi = Wf
        phi = random.random()*2*math.pi  # random phase
        Emw_t = Emw*math.exp(-ti/tring)  # ringing down
        Wf = Wi + dWs(phi, Emw_t, fmw)  # new energy
        # orbit
        tt = lookup_tt(Wf, dup)
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
    # [r, v] = y
    dydt = [y[1], 1/y[0]**2 + Ep]
    return dydt


# split dervitive into up/down
@numba.jit
def derivative_up(y, t, Ep):
    """Return dydt of y=[r,v] for given field Ep (a.u.)  for uphill e-"""
    [r, v] = y
    # uphill is r<0, Coulomb + & field +
    return np.array([v, 1/r**2 + Ep])
    # dydt


@numba.jit
def jac_up(y, t, Ep):
    """jacobian of derivative_up. j[i, j] = d/dy[j] (dy/dt[i])"""
    # [r, v] = y
    jac = np.zeros((2, 2))
    jac[0, 1] = 1  # d/dv (dr/dt) = d/dv (v)
    jac[1, 0] = -2/y[0]**3  # d/dr (dv/dt) = d/dr (1/r^2 + Ep)
    # return np.array([[0, 1], [-2/y[0]**3, 0]])
    return jac


@numba.jit
def derivative_down(y, t, Ep):
    """Return dydt of y=[r,v] for given field Ep (a.u.) for downhill e-"""
    [r, v] = y
    # downhill is r>0, Coulomb - & field +
    return np.array([v, -1/r**2 + Ep])
    # return dydt


@numba.jit
def jac_down(y, t, Ep):
    """jacobian of derivative_down. j[i, j] = d/dy[j] (dy/dt[i])"""
    jac = np.zeros((2, 2))
    jac[0, 1] = 1  # d/dv (dr/dt) = d/dv (v)
    jac[1, 0] = 2/y[0]**3  # d/dr (dv/dt) = d/dr (-1/r^2 + Ep)
    # return np.array([[0, 1], [2/y[0]**3, 0]])
    return jac


def check_jac():
    # y = np.array([-100.0, -0.14135])
    y = np.array([-10.0, -0.4472])
    t = 0
    Ep = 100*1.94469e-13
    fd = np.array(derivative_down(y, t, Ep))
    delta = 1000*np.finfo(float).eps
    # print(fd)
    dr = y + np.array([delta, 0])
    fd_dr = np.array(derivative_down(dr, t, Ep))
    dv = y + np.array([0, delta])
    fd_dv = np.array(derivative_down(dv, t, Ep))
    jd = jac_down(y, t, Ep)
    dfdr = (fd_dr - fd)/delta
    dfdv = (fd_dv - fd)/delta
    rep = "df0/dx0 = {}\n".format(jd[0, 0])
    rep = rep + "dfr/dr = {}\n".format(dfdr[0])
    rep = rep + "\n"
    rep = rep + "df0/dx1 = {}\n".format(jd[0, 1])
    rep = rep + "dfr/dv = {}\n".format(dfdv[0])
    rep = rep + "\n"
    rep = rep + "df1/dx0 = {}\n".format(jd[1, 0])
    rep = rep + "dfv/dr = {}\n".format(dfdr[1])
    rep = rep + "\n"
    rep = rep + "df1/dx1 = {}\n".format(jd[1, 1])
    rep = rep + "dfv/dv = {}\n".format(dfdv[1])
    print(rep)
    # y = -y
    # f_up = derivative_up(y, t, Ep)
    # j_up = jac_up(y, t, Ep)
    return


# add dup to choose derivative_up/down
def path_integration(r0, v0, t, Ep, dup):
    """Path integration field Ep (a.u.)

    non-keyword arguments:
    r0 -- float: initial radius (a.u.)
    v0 -- float: initial velocity (a.u.)
    t -- array: times to solve for, usually just [t0, tfinal] (a.u.)
    Ep -- float: static field (a.u.)
    dup -- Bool: True/False means up/down hill e-

    returns y -- e- final state [r, v]
    """
    y0 = [r0, v0]
    if dup:
        y = odeint(derivative_up, y0, t, Dfun=jac_up, args=(Ep,))
        # y = odeint(derivative_up, y0, t, args=(Ep,))
    else:
        y = odeint(derivative_down, y0, t, Dfun=jac_down, args=(Ep,))
        # y = odeint(derivative_down, y0, t, args=(Ep,))
    return y


def cgtfunc(x):
    return x[0]**2 - 0.5*x[1]**3


def cgtgrad(x):
    return [2*x[0], -1.5*x[1]**2]


def check_grad_test():
    return check_grad(cgtfunc, cgtgrad, [1.5, -1.5])


# [r0, v0] is +/- for down/up, add dup to path_integration()
def int_path(W0, Ep, ti, tstop, dup, tr=False):
    """Integrate e- from the core to final position at stopping time.

    If an electron's orbit is deemed to end after tstop, start from the core
    and integrate the path from ti to tstop, and return the e- final state.

    non-keyword arguments:
    W0 -- float: initial e- energy (a.u.)
    Ep -- float: static field (a.u.)
    ti -- float: starting time (a.u.)
    tstop -- float: stopping time (a.u.)
    dup -- Bool: True/False means up/down hill e-

    returns:
    t -- array: array of times to integrate [ti, tstop]
    y -- array: final e- state [r, v]
    """
    r0 = 6
    if W0 + 1/abs(r0) >= 0:
        v0 = math.sqrt(2*(W0 + 1/abs(r0)))
    else:
        v0 = 0
        r0 = abs(1/W0)
    # -[r0, v0] for uphill e-
    if dup is True:
        r0 = -r0
        v0 = -v0
    if tr is False:
        t = [ti, tstop]
    else:
        t = np.linspace(ti, tstop, 100)
    y = path_integration(r0, v0, t, Ep, dup)
    return t, y


# [r0, v0] is +/- for down/up, add dup to path_integration()
def int_path_track(W0, Ep, ti, tstop, dup):
    """Returns path with 100 points."""
    r0 = 6
    if W0 + 1/abs(r0) >= 0:
        v0 = math.sqrt(2*(W0 + 1/abs(r0)))
    else:
        v0 = 0
        r0 = abs(1/W0)
    # -[r0, v0] for uphill e-
    if dup is True:
        r0 = -r0
        v0 = -v0
    t = np.linspace(ti, tstop, 100)
    y = path_integration(r0, v0, t, Ep, dup)
    return t, y


# no dup
@numba.jit
def y_to_w(y):
    """Return energy (a.u.) of e- based on y = [r, v]."""
    return -1/abs(y[0]) + y[1]**2/2


# no dup
@numba.jit
def y_to_w_vec(y):
    """Return energy (a.u.) of e- based on y = [r, v]. W and y are vectors."""
    return -1/np.abs(y[:, 0]) + np.power(y[:, 1], 2)/2
# ==========


# ==========
# Full execution
# ==========
# dup in dW_orbit() and int_path()
def run_to_20ns(W0, Ep, Emw, fmw, t0, tstop, Wlim, dup,
                lut_up_f, lut_down_f, tr=False):
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
    t0 -- float: starting time (a.u.)
    tstop -- float: stopping time (a.u.)
    Wlim -- float: cutoff e- energy (a.u.), -600 GHz is tt ~ 1/(15.9 GHz)
    dup -- Bool: True/False means up/down hill e-

    returns: Wfinal -- Final e- energy (a.u.)
    """
    # MW exchange and orbit
    if tr is False:
        obs = dW_orbit(W0, Ep, Emw, fmw, t0, tstop, Wlim, dup,
                       lut_up_f, lut_down_f, tr=tr)
    else:
        obs, track = dW_orbit(W0, Ep, Emw, fmw, t0, tstop, Wlim, dup,
                              lut_up_f, lut_down_f, tr=tr)
    t = obs['ti']
    W = obs['Wf']
    if W > Wlim:
        t, y = int_path(W, Ep, t, tstop, dup, tr=tr)  # returns start & stop
        if tr is False:
            y = y[-1]  # only the stop point
            Wfinal = y_to_w(y)
        else:
            Wfinal = y_to_w_vec(y)
    else:
        if tr is False:
            Wfinal = W
            t, y = [math.nan, math.nan], [math.nan, math.nan]
        else:
            Wfinal = [W, W]
            # mimick list of ys
            t, y = [math.nan, math.nan], [[math.nan, math.nan]]*2
    if tr is False:
        return Wfinal, y
    else:
        return Wfinal, y, t, track


# dup in dW_orbit() and int_path()
def run_to_20ns_track(W0, Ep, Emw, fmw, t0, tstop, Wlim, dup):
    """Keeps list of every orbit and the path integration."""
    # MW exchange and orbit
    obs, track = dW_orbit_track(W0, Ep, Emw, fmw, t0, tstop, Wlim, dup)
    t = obs['ti']
    W = obs['Wf']
    if W > Wlim:
        t, y = int_path_track(W, Ep, t, tstop, dup)
        Wfinal = y_to_w_vec(y)  # small time savings
    else:
        Wfinal = [W, W]
        # mimick list of ys
        t, y = [math.nan, math.nan], [[math.nan, math.nan]]*2
    return Wfinal, y, t, track


# dup in catch_at_20ns() and dW_orbit_Epoff
def run_after_20ns(W0, y0, Emw, fmw, t0, tstop, Wlim, tring, dup,
                   lut_up, lut_down, tr=False):
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
    t0 -- float: starting time (a.u.)
    tstop -- float: stopping time (a.u.)
    Wlim -- float: cutoff e- energy (a.u.), -600 GHz is tt ~ 1/(15.9 GHz)
    dup -- Bool: True/False means up/down hill e-

    returns: Wfinal, obs
        Wfinal -- float: Final e- energy (a.u.)
        obs -- dict: Final sate from dW_orbit_Epoff()
    if track is True: Wfinal, tc, track
        tc -- float: time from catch_at_20ns() (a.u.)
        track -- list: obs dicts from dW_Orbits_Epoff()
    """
    tc = catch_at_20ns(W0, y0, t0, dup)
    if tr is False:
        obs = dW_orbit_Epoff(
                W0, Emw, fmw, tc, tstop, Wlim, tring, dup,
                lut_up, lut_down, tr=tr)
    else:
        obs, track = obs, track = dW_orbit_Epoff(
                W0, Emw, fmw, tc, tstop, Wlim, tring, dup,
                lut_up, lut_down, tr=tr)
    Wfinal = obs['Wf']
    if tr is False:
        return Wfinal
    else:
        return Wfinal, tc, track


# dup in catch_at_20ns() and dW_orbit_Epoff
def run_after_20ns_track(W0, y0, Emw, fmw, t0, tstop, Wlim, tring, dup):
    """Keep lists of every orbit"""
    tc = catch_at_20ns(W0, y0, t0, dup)
    # if tc < tstop and W0 > Wlim:
    obs, track = dW_orbit_Epoff_track(W0, Emw, fmw, tc, tstop, Wlim, tring,
                                      dup)
    Wfinal = obs['Wf']
    return Wfinal, tc, track


# dup in run_to_20ns() and run_after_20ns()
def run_to_stop(W0, Ep, Emw, fmw, t0, toff, tstop, Wlim, tring, dup,
                lut_up, lut_down, lut_up_f, lut_down_f, tr=False):
    """Joins run_to_20ns() to run_after_20ns(), returns Wfinal (a.u.)

    non-keyword arguments:
    W0 -- float: e- Energy (a.u.) when starting the orbit
    Ep -- float: Static field strenght (a.u.)
    Emw -- float: MW field amplitude (a.u.)
    fmw -- float: MW frequency (rad, NOT cycles!) (a.u.)
    t0 -- float: starting time (a.u.)
    toff -- float: When Ep turns off, should be 20ns (a.u.)
    tstop -- float: stopping time (a.u.)
    Wlim -- float: cutoff e- energy (a.u.), -600 GHz  tt(W, 0) ~ 1/(15.9 GHz)
    dup -- Bool: True/False means up/down hill e-

    keyword argument: track -- Bool: extra returns to track orbits

    returns:
        Wfinal -- float: final energy (a.u.)
    if track == True:
        t -- array: Times from path integration
        y -- array: [[r,v]] from path integration
        track1 -- array: dicts of obs from dW_orbits_to_20ns()
        tc -- float: time from catch_after_20ns()
        track2 -- array: dicts of obs from dW_orbit_after_20ns()
    """
    # global lut_up
    # global lut_down
    # global lut_up_f
    # global lut_down_f
    if tr is False:
        Wfinal, y = run_to_20ns(
                W0, Ep, Emw, fmw, t0, toff, Wlim, dup,
                lut_up_f, lut_down_f, tr=tr)
        t0 = toff
        y0 = y
        W0 = Wfinal
        Wfinal = run_after_20ns(
                W0, y0, Emw, fmw, t0, tstop, Wlim, tring, dup,
                lut_up, lut_down, tr=tr)
        return Wfinal
    else:
        Wfinal, y, t, track1 = run_to_20ns(
                W0, Ep, Emw, fmw, t0, toff, Wlim, dup,
                lut_up_f, lut_down_f, tr=tr)
        t0 = t[-1]
        y0 = y[-1]
        W0 = Wfinal[-1]
        Wfinal, tc, track2 = run_after_20ns(
                W0, y0, Emw, fmw, t0, tstop, Wlim, tring, dup,
                lut_up, lut_down, tr=tr)
        return Wfinal, t, y, track1, tc, track2


# dup in run_to_20ns() and run_after_20ns()
def run_to_stop_track(W0, Ep, Emw, fmw, t0, toff, tstop, Wlim, tring, dup):
    """run_to_stop tracking each orbit"""
    Wfinal, y, t, track1 = run_to_20ns_track(
            W0, Ep, Emw, fmw, t0, toff, Wlim, dup)
    t0 = t[-1]
    y0 = y[-1]
    W0 = Wfinal[-1]
    Wfinal, tc, track2 = run_after_20ns_track(
            W0, y0, Emw, fmw, t0, tstop, Wlim, tring, dup)
    return Wfinal, t, y, track1, tc, track2
# ==========


# ==========
# Timing
# ==========
# randomly define dup in setup, add to run_to_20ns()
def time_run_to_20ns(n):
    """n executions run_to_20ns with random W0, Ep. Plot histogram of times.

    Build n-length random arrays of W0 from -100 to 50 GHz and Ep from 0 to
    300 mV/cm. Build an array of run times for run_to_20ns(), and then plot a
    histogram of the time distributions. Return the array of times.

    returns trun -- list: run times for run_to_20ns()
    """
    au = atomic_units()
    trun = []
    W0s = (np.random.random(n) - 2/3)*150*au['GHz']
    # Eps = np.random.random(n)*300*au['mVcm']
    Eps = np.random.choice(range(0, 300+1, 1), size=n)*1.94469e-13
    Emw = 4*1000*au['mVcm']
    fmw = 2*math.pi*15.932/au['ns']
    t0 = 0*au['ns']
    tstop = 20*au['ns']
    Wlim = -600*au['GHz']
    dups = np.random.choice([True, False], size=n)  # np.bool_ !!!!!
    # field lookup tables
    global lut_up_f
    global lut_down_f
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Eps[0])
    for i in range(n):
        progress("time_run_to_20ns()", i, n)
        clock_start = time.clock()
        run_to_20ns(W0s[i], Eps[i], Emw, fmw, t0, tstop, Wlim,
                    bool(dups[i]), tr=False)  # np.bool_ !!!!!
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


# randomly define dup in setup, add to run_to_stop()
def time_run_to_stop(n):
    """n executions run_to_stop() with random W0, Ep. Plot histogram of times.

    Build n-length random arrays of W0 from -100 to 50 GHz and Ep from 0 to
    300 mV/cm. Build an array of run times for run_to_20ns(), and then plot a
    histogram of the time distributions. Return the array of times.

    returns trun -- list: run times for run_to_stop()
    """
    au = atomic_units()
    trun = []
    W0s = (np.random.random(n) - 2/3)*150*au['GHz']  # -100 -> 50 GHz
    # Eps = np.random.random(n)*300*au['mVcm']
    # Eps = np.array([100*au['mVcm']]*n)
    Eps = np.random.choice(range(0, 301, 1), size=n)*1.94469e-13
    Emw = 4*1000*au['mVcm']
    fmw = 2*math.pi*15.932/au['ns']
    t0 = 0*au['ns']
    toff = 40*au['ns']
    tring = 180*au['ns']
    tstop = toff + 5*tring
    Wlim = -600*au['GHz']
    dups = np.random.choice([True, False], size=n)  # np.bool_ !!!!!
    # field lookup tables
    lut_up, lut_down = tab.import_lookup_table()
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Eps[0])
    # execute
    for i in range(n):
        progress("time_run_to_stop()", i, n)
        clock_start = time.clock()
        run_to_stop(
                W0s[i], Eps[i], Emw, fmw, t0, toff, tstop, Wlim, tring,
                bool(dups[i]), lut_up, lut_down, lut_up_f, lut_down_f,
                tr=False)  # np.bool_ !!!!!
        clock_end = time.clock()
        trun = trun + [clock_end - clock_start]
    # plot
    bins = np.linspace(0, np.mean(trun)*2, int(n/10))
    print('mean = {} s'.format(np.mean(trun)))
    print('median = {} s'.format(np.median(trun)))
    fig, ax = plt.subplots()
    ax.hist(trun, bins)
    ax.set(title="{0} runs of 'run_to_stop()'".format(n), xlabel="time (s)")
    plt.show()
    return trun


# randomly define dup in setup, add to run_to_stop()
def timeit_run_to_stop():
    """timeit callable for run_to_stop()"""
    W0 = (random.random()*3 - 2)*150*1.51983e-7
    # Ep = random.random()*300*1.94469e-13
    Ep = random.choice(range(0, 301, 1))*1.94469e-13
    # Ep = 0
    Emw = 4*1000*1.94469e-13
    fmw = 2*math.pi*15.932/41341400.0
    t0 = 0*41341400.0
    toff = 20*41341400.0
    tring = 10*41341400.0
    tstop = toff + 5*tring
    Wlim = -600*1.51983e-7
    dup = random.choice([True, False])
    # field lookup tables
    global lut_up_f
    global lut_down_f
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Ep)
    # execute
    run_to_stop(W0, Ep, Emw, fmw, t0, toff, tstop, Wlim, tring, dup)
    return


# randomly define dup in setup, ad to run_to_20ns()
def timeit_run_to_20ns():
    """timeit callable for run_to_20ns()"""
    W0 = (random.random()*3 - 2)*150*1.51983e-7
    Ep = random.choice(range(0, 301, 1))*1.94469e-13
    Emw = 4*1000*1.94469e-13
    fmw = 2*math.pi*15.932/41341400.0
    t0 = 0*41341400.0
    toff = 20*41341400.0
    # tring = 10*41341400.0
    # tstop = toff + 5*tring
    Wlim = -600*1.51983e-7
    dup = random.choice([True, False])
    # field lookup tables
    global lut_up_f
    global lut_down_f
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Ep)
    # execute
    run_to_20ns(W0, Ep, Emw, fmw, t0, toff, Wlim, dup)
    return
# ==========


# ==========
# Plotting
# ==========
# randomly define dup, add to run_to_20ns_track(), run_after_20ns_track() and
# add as title to plot.
def plot_run_to_stop():
    """Plot the energy vs. time from each step in the model."""
    au = atomic_units()
    # ==========
    # Model
    # ==========
    # run_to_20ns_track() arguments
    W0 = (random.random() - 2/3)*150*au['GHz']
    Ep = random.choice(range(0, 301, 1))*au['mVcm']
    Emw = 4*1000*au['mVcm']
    fmw = 2*math.pi*15.932/au['ns']
    t0 = 0*au['ns']
    toff = 20*au['ns']
    Wlim = -600*au['GHz']
    tring = 10*au['ns']  # guess at ringdown
    tstop = toff + 5*tring  # very little MW left
    dup = random.choice([True, False])  # otherwise, numpy.bool_
    # dup = True
    # field lookup tables
    global lut_up_f
    global lut_down_f
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Ep)
    # run to stop
    Wfinal, t, y, track1, tc, track2 = run_to_stop(
            W0, Ep, Emw, fmw, t0, toff, tstop, Wlim, tring, dup, tr=True)
    intg_w = y_to_w_vec(np.array(y))
    # track1 -- Group 0
    df = track1[['ti', 'Wi']].copy()
    df.rename(index=int, columns={'ti': 't', 'Wi': 'W'}, inplace=True)
    shift = track1[['ti', 'Wf']].copy()
    shift.rename(index=int, columns={'ti': 't', 'Wf': 'W'}, inplace=True)
    shift['t'] = shift['t'] + 1  # 10 as shfit
    df = df.append(shift, ignore_index=True)
    df['group'] = 0
    df.sort_values(by='t', inplace=True)
    # integrated path -- Group 1
    df = df.append(pd.DataFrame({'t': t, 'W': intg_w, 'group': 1}),
                   ignore_index=True)
    # catch -- Group 2
    df = df.append(pd.DataFrame(
            {'t': np.array([t[-1], tc]), 'W': intg_w[-1], 'group': 2}),
            ignore_index=True)
    # track2 -- Group 3
    tdf = track2[['ti', 'Wi']].copy()
    tdf.rename(index=int, columns={'ti': 't', 'Wi': 'W'}, inplace=True)
    shift = track2[['ti', 'Wf']].copy()
    shift.rename(index=int, columns={'ti': 't', 'Wf': 'W'}, inplace=True)
    shift['t'] = shift['t'] + 1  # 10 as shift
    # stopping point
    ostop = track2.iloc[-1][['tf', 'Wf']]
    ostop.rename({'tf': 't', 'Wf': 'W'}, inplace=True)
    shift = shift.append(ostop)
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
    df.where(df['group'] == 0).plot(
            x='t', y='W', label='dW', ax=ax, c='C0', marker='o')
    df.where(df['group'] == 3).plot(
            x='t', y='W', label='dWo', ax=ax, c='C3', marker='o')
    df.where(df['group'] == 2).plot(
            x='t', y='W', label='catch', ax=ax, c='C2', marker='o')
    df.where(df['group'] == 1).plot(
            x='t', y='W', label='intg', ax=ax, c='C1', marker='.')
    ax.set(title="dup = {}".format(dup))
    return df
# ==========


def main():
    # au = atomic_units()
    # lookup tables
    # lut_up, lut_down = tab.import_lookup_table()
    # check_jac()
    # ==========
    # plot_run_to_stop()
    # ==========
    # for i in range(100):
    #     progress("\nmain()", i, 100)
    #     print()
    #     result = plot_run_to_stop_track()
    #     plt.close()
    # result = plot_run_to_stop()
    # ==========
    # timing
    # ==========
    result = time_run_to_stop(int(1e4))
    # n = 1.0e4
    # t = timeit.repeat(timeit_run_to_20ns, repeat=3, number=int(n))
    # print(np.array(t) / n)
    # ==========
    # test lut_field
    # ==========
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
