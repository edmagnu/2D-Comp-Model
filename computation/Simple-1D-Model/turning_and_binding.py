# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:04:10 2018

@author: edmag
"""

# Turning and Binding Time integral calculation.
# See 1DModel.pdf and turning.nb in 2D-Comp-Model\
# Eric Magnuson, University of Virginia, VA

import numpy as np
from scipy.integrate import quad  # turning and binding time integrals
from scipy.optimize import minimize_scalar
import itertools
import pandas as pd
import timeit
import numba
import matplotlib.pyplot as plt
import time
import os


def atomic_units():
    """Return a dictionary of atomic units"""
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
# integrands, dt/dz for up/down-hill electrons
# numba.jit x10 speedup
# ==========
@numba.jit
def intg_up(z, W, f):
    """Return dt/dz for uphill electron in Coulomb & Static Field

    non-keyword arguments
    z -- electron position (a.u.)
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)
    """
    return -1/np.sqrt(2*(W - 1/z + f*z))


@numba.jit
def intg_down(z, W, f):
    """Return dt/dz for downhill electron in Coulomb & Static Field

    non-keyword arguments
    z -- electron position (a.u.)
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)
    """
    return 1/np.sqrt(2*(W + 1/z + f*z))
# ==========


# ==========
# Turning Times
# numba.jit does not imporve, requires passing functions
# ==========
def tt_up(W, f):
    """Turning time for an uphill electron. np.NaN if never turns.

    non-keyword arguments
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)

    returns tuple -- (turning time, error est) in a. u.
    """
    zi = -6  # starting position, uphill -> z < 0
    if f == 0 and W < 0:  # below limit without field
        zt = 1/W  # turning position, W + -1/z = 0
        tt = quad(intg_up, zi, zt, args=(W, f))[0]  # integrator
    elif f == 0 and W >= 0:  # above limit without field
        # Never turns
        zt = np.NaN
        tt = np.NaN
    elif f > 0:  # always turns if there is a field
        zt = -1/(2*f) * (W + np.sqrt(W**2 + 4*f))  # W + -1/z - fz = 0
        tt = quad(intg_up, zi, zt, args=(W, f))[0]  # integrator
    else:  # should never get here
        print("tt_up({0}, {1}) : something went wrong".format(W, f))
        zt = None
        tt = np.NaN
    return tt


def tt_down(W, f):
    """Turning time for an downhill electron. np.NaN if never turns.

    non-keyword arguments
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)

    returns tuple -- (turning time, error est) in a. u.
    """
    zi = 6  # starting position, downhill -> z > 0
    il = -2*f**0.5  # Ionization limit
    if f > 0 and W < il:  # with field and bound
        zt = -1/(2*f) * (W + np.sqrt(W**2 - 4*f))  # W + 1/z -fz = 0
        tt = quad(intg_down, zi, zt, args=(W, f))[0]  # integrator
    elif f == 0 and W < il:  # no field and bound
        zt = -1/W  # turning position, W + 1/z = 0
        tt = quad(intg_down, zi, zt, args=(W, f))[0]  # integrator
    elif W >= il:  # Unbound
        # Never turns
        zt = np.NaN
        tt = np.NaN
    else:  # should never get here
        print("tt_down({0}, {1}) : something went wrong".format(W, f))
        zt = None
        tt = np.NaN
    return tt
# ==========


# ==========
# Binding Times
# numba.jit does not imporve, requires passing functions
# ==========
def tb_up(W, f):
    """Binding time for an uphill electron. np.NaN if W never crosses 0.

    non-keyword arguments
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)

    returns tuple -- (binding time, error est) in a. u.
    """
    zi = -6  # starting position, uphill z < 0
    if f > 0 and W > 0:  # with field and above limit
        zb = -W/f  # W - fz = 0
        tb = quad(intg_up, zi, zb, args=(W, f))[0]  # integrator
    elif f > 0 and W <= 0:  # with field and below limit
        # Is always bound
        zb = np.NaN
        tb = np.NaN
    elif f == 0:  # No field
        # Cannot lose energy to field
        zb = np.NaN
        tb = np.NaN
    else:  # should never get here
        print("tb_up({0}, {1}) : something went wrong".format(W, f))
        zb = None
        tb = np.NaN
    return tb


def tb_down(W, f):
    """Binding time for downhill electron. np.NaN if W never crosses 0.

    non-keyword arguments
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)

    returns tuple -- (binding time, error est) in a. u.
    """
    zi = 6  # starting position, downhilll z > 0
    il = -2*f**0.5  # Ionization limit
    if f > 0 and W > il and W < 0:  # field, W between 0 and Ionization Limit
        zb = -W/f  # W - fz = 0
        tb = quad(intg_down, zi, zb, args=(W, f))[0]  # integrator
    elif f == 0 or W <= il or W >= 0:  # any other condition
        # Can never cross W = 0
        zb = np.NaN
        tb = np.NaN
    else:  # should never get here
        print("tb_down({0}, {1}) : something went wrong".format(W, f))
        zb = None
        tb = np.NaN
    return tb
# ==========


# ==========
# test times
# ==========
def check_nan(v1, v2):
    """Check v1 == v2 are the same after rounding, including NaN."""
    if np.isnan(v1) or np.isnan(v2):
        t = np.isnan(v1) and np.isnan(v2)
    else:
        t = np.round(v1, 3) == np.round(v2, 3)
    return t


def test_tt_tb():
    """Test turing and binding times for a variety of Energies and Fields

    For field f = 0, 10 mV/cm and W = -30, -10, 0, 10 GHz, calculate the
    turning and binding times. Check that they match values returned by the
    pre-existing Mathematica notebook 'turning.nb'.
    Prints (W, f, up/down, tt, tb, check pass/fail) for each condition.
    """
    au = atomic_units()
    # fields and energies
    fs = [0, 10]
    Ws = [-30, -10, 0, 10]
    # build dataframe of turning and binding times, and checks
    df = pd.DataFrame()
    dfc = pd.read_csv("checks.txt", index_col=0)
    test = True
    for f, W in itertools.product(fs, Ws):
        mask = np.logical_and(dfc['f'] == f, dfc['W'] == W)
        obs = dfc[mask]
        fau = f*au['mVcm']
        Wau = W*au['GHz']
        ttu = tt_up(Wau, fau)/au['ns']
        test = test and check_nan(ttu, float(obs['ttu']))
        tbu = tb_up(Wau, fau)/au['ns']
        test = test and check_nan(tbu, float(obs['tbu']))
        ttd = tt_down(Wau, fau)/au['ns']
        test = test and check_nan(ttd, float(obs['ttd']))
        tbd = tb_down(Wau, fau)/au['ns']
        test = test and check_nan(tbd, float(obs['tbd']))
        obs = {'f': f, 'W': W, 'ttu': ttu, 'tbu': tbu, 'ttd': ttd, 'tbd': tbd,
               'check': test}
        df = df.append(obs, ignore_index=True)
    # rearrange
    df['check'] = df['check'].astype(bool)
    df = df[['f', 'W', 'check', 'ttd', 'tbd', 'ttu', 'tbu']]
    print(df)
    return df
# ==========


# ==========
# Solve for fields for each W
# numba.jit, numba.vectorize no speedup
# ==========
def tt_up_W_target(W, f):
    """tt_up(W, f) - 20 ns"""
    return abs(tt_up(W, f) - 826828000.0)  # 20 ns in a.u.


def tt_up_W(f):
    """Given field f (a.u.), find W (a.u.) so that tt_up(W, f) = 20ns."""
    if f > 0:
        bracket = (-100*1.51983e-07, 100*1.51983e-07)
        result = minimize_scalar(tt_up_W_target, bracket=bracket, tol=1e-010,
                                 args=f)
    elif f == 0:
        bound = (-100*1.51983e-07, 0 + np.finfo(float).eps)
        msopts = {'xatol': 1e-010}  # 1 MHz
        # bounded to keep from returning NaN
        result = minimize_scalar(tt_down_W_target, method='Bounded',
                                 bounds=bound, args=f, options=msopts)
    return result['x']


def tr_up_W_target(W, f):
    """tt_up(W, f) - 10 ns"""
    return abs(tt_up(W, f) - 826828000.0/2)  # 10 ns in a.u.


def tr_up_W(f):
    """Given field f (a.u.) find W (a.u.) so that tt_up = 10 ns."""
    if f > 0:
        bracket = (-100*1.51983e-07, 100*1.51983e-07)
        result = minimize_scalar(tr_up_W_target, bracket=bracket, tol=1e-010,
                                 args=f)
    elif f == 0:
        bound = (-100*1.51983e-07, 0 + np.finfo(float).eps)
        msopts = {'xatol': 1e-010}  # 1 MHz
        # bounded to keep from returning NaN
        result = minimize_scalar(tr_up_W_target, method='Bounded',
                                 bounds=bound, args=f, options=msopts)
    return result['x']


def tt_down_W_target(W, f):
    """absolute value of tt_down(W, f) - 20ns"""
    return abs(tt_down(W, f) - 826828000.0)  # 20 ns in a.u.


def tt_down_W(f):
    """Given field f (a.u.), find W (a.u.) so that tt_down(W, f) = 20ns."""
    # upper bound avoids integration errors
    bound = (-600*1.51983e-07, -2*np.sqrt(f) - 100*np.finfo(float).eps)
    msopts = {'xatol': 1e-010}  # 1 MHz
    # bounded to keep from returning NaN
    result = minimize_scalar(tt_down_W_target, method='Bounded', bounds=bound,
                             args=f, options=msopts)
    return result['x']


def tr_down_W_target(W, f):
    """absolute value of tt_down(W, f) - 20ns"""
    return abs(tt_down(W, f) - 826828000.0/2)  # 20 ns in a.u.


def tr_down_W(f):
    """Given field f (a.u.), find W (a.u.) so that tt_down(W, f) = 20ns."""
    # upper bound avoids integration errors
    bound = (-600*1.51983e-07, -2*np.sqrt(f) - 100*np.finfo(float).eps)
    msopts = {'xatol': 1e-010}  # 1 MHz
    # bounded to keep from returning NaN
    result = minimize_scalar(tr_down_W_target, method='Bounded', bounds=bound,
                             args=f, options=msopts)
    return result['x']


def tb_up_W_target(W, f):
    """tb_up(W, f) - 20 ns"""
    return abs(tb_up(W, f) - 826828000.0)  # 20 ns in a.u.


def tb_up_W(f):
    """Given field f (a.u.), find W (a.u.) so that tb_up(W, f) = 20ns."""
    if f > 0:
        # bound to avoid NaN
        # max for f=300 mV/cm
        bound = (0 + np.finfo(float).eps, 10000*1.51983e-07)
        msopts = {'xatol': 1e-10}  # 1 MHz
        # bounded to keep from returning NaN
        result = minimize_scalar(tb_up_W_target, method='Bounded',
                                 bounds=bound, args=f, options=msopts)
    if f == 0:
        result = {'x': 0}
    return result['x']


def tb_down_W_target(W, f):
    """tb_down(W, f) - 20ns"""
    return abs(tb_down(W, f) - 826828000.0)  # 20 ns in a.u.


def tb_down_W(f):
    """Given field f (a.u.), find W (a.u.) so that tb_down(W, f) = 20ns."""
    if f > 0:
        # bound to avoid NaN
        # il < W < 0
        bound = (-2*f**0.5 + np.finfo(float).eps, 0 - np.finfo(float).eps)
        msopts = {'xatol': 1e-10}  # 1 MHz
        # bounded to keep from returning NaN
        result = minimize_scalar(tb_down_W_target, method='Bounded',
                                 bounds=bound, args=f, options=msopts)
    if f == 0:
        result = {'x': 0}
    return result['x']


def tp_up_W_target(W, f):
    """2*tt_up(W, f) - tb_up(W, f) - 20 ns"""
    return abs(2*tt_up(W, f) - tb_up(W, f) - 826828000.0)  # 20 ns in a.u.


def tp_up_W(f):
    """Given field f (a.u.), find W (a.u.) so that 2*tt_up - tb_up = 20ns."""
    if f > 0:
        # bound to avoid NaN
        # max for f=300 mV/cm
        bound = (0 + np.finfo(float).eps, 10000*1.51983e-07)
        msopts = {'xatol': 1e-10}  # 1 MHz
        # bounded to keep from returning NaN
        result = minimize_scalar(tp_up_W_target, method='Bounded',
                                 bounds=bound, args=f, options=msopts)
    if f == 0:
        result = {'x': np.NaN}
    return result['x']
# ==========


# ==========
# bulk
# ==========
def plot_tp_problem():
    """tp_up_W behaves badly when f < ~ 13 mV/cm, trim W < tol in minimize"""
    # df = bulk()
    au = atomic_units()
    df = 0.1
    f = np.arange(0, 20+df, df)*au['mVcm']
    Wp = np.array(list(map(tp_up_W, f)))
    Wt = np.array(list(map(tt_up_W, f)))
    Wb = np.array(list(map(tb_up_W, f)))
    Wp[Wp < 1e-10] = np.NaN
    plt.axvline(0)
    plt.axhline(0)
    plt.plot(f/au['mVcm'], Wp/au['GHz'], label="tp")
    plt.plot(f/au['mVcm'], Wt/au['GHz'], label="tt")
    plt.plot(f/au['mVcm'], Wb/au['GHz'], label="tb")
    plt.plot([0], [0], '.k')
    plt.plot([max(f[np.isnan(Wp)])/au['mVcm']], [0], '.k')
    plt.legend()
    return f, Wp, Wt, Wb


def bulk():
    """Save dataframe of energies for a range of fields for time conditions."""
    au = atomic_units()
    # fields from 0.1 to 300 mV/cm
    df = 0.1
    df = pd.DataFrame({'f': np.arange(0, 300+df, df)*au['mVcm']})
    # find turning and binding times
    funcs = [tt_up_W, tt_down_W,
             tr_up_W, tr_down_W,
             tb_up_W, tb_down_W,
             tp_up_W]
    keys = ['tt_up', 'tt_down',
            'tr_up', 'tr_down',
            'tb_up', 'tb_down',
            'tp_up']
    for i in range(len(funcs)):
        progress('bulk()', i, len(funcs))
        df[keys[i]] = df['f'].apply(funcs[i])
    # ionization limit
    df['il'] = -2*np.sqrt(df['f'])
    # tp_up is a special case
    df.loc[df['tp_up'] < 1e-10, 'tp_up'] = np.NaN
    # save
    df.to_csv("fields_and_times.txt")
    return df


def lookup_table():
    """Build a lookup table of turning times for f = 0."""
    au = atomic_units()
    f = 0
    dW = 1e-8
    Ws = np.arange(-600*au['GHz'], 0, dW)
    # print(len(Ws))
    tts = np.array(list(map(tt_up, Ws, itertools.repeat(f))))
    df = pd.DataFrame({'W': Ws, 'tt': tts})
    df.to_hdf("lookup_table_up.h5", 'df')
    tts = np.array(list(map(tt_down, Ws, itertools.repeat(f))))
    df = pd.DataFrame({'W': Ws, 'tt': tts})
    df.to_hdf("lookup_table_down.h5", 'df')
    return df


def lookup_table_f(f):
    """Build a lookup table of turning times for arbitrary f (a.u.)."""
    au = atomic_units()
    dW = 1e-8
    Ws = np.arange(-600*au['GHz'], 1000*au['GHz'], dW)
    tts = np.array(list(map(tt_up, Ws, itertools.repeat(f))))
    df = pd.DataFrame({'W': Ws, 'tt': tts})
    fstring = str(int(round(f/au['mVcm'], 0)))
    fout = "lookup_table_up_" + fstring + ".h5"
    fout = os.path.join("lut", fout)
    df.to_hdf(fout, 'df')
    tts = np.array(list(map(tt_down, Ws, itertools.repeat(f))))
    df = pd.DataFrame({'W': Ws, 'tt': tts})
    fout = "lookup_table_down_" + fstring + ".h5"
    fout = os.path.join("lut", fout)
    df.to_hdf(fout, 'df')
    return


def import_lookup_table():
    """Import the lookup table of tt for f = 0, and round values.

    returns:
    lut_up -- dict for uphill electrons in f = 0
    lut_down -- dict for downhill electrons in f = 0
    """
    # load
    lut_up = pd.read_hdf("lookup_table_up.h5", 'df')
    lut_down = pd.read_hdf("lookup_table_down.h5", 'df')
    # round
    lut_up['W'] = np.floor(lut_up['W']*1e8).astype(int)
    lut_down['W'] = np.floor(lut_down['W']*1e8).astype(int)
    # reindex
    lut_up.set_index('W', inplace=True, verify_integrity=True)
    lut_down.set_index('W', inplace=True, verify_integrity=True)
    # to dicts, > x10 speedup
    lut_up = lut_up.to_dict()['tt']
    lut_down = lut_down.to_dict()['tt']
    return lut_up, lut_down


def import_lookup_table_f(f):
    """Import the lookup table of tt for f (a.u.), and round values.

    returns:
    lut_up -- dict for uphill electrons in f
    lut_down -- dict for downhill electrons in f
    """
    au = atomic_units()
    fstring = str(int(round(f/au['mVcm'], 0)))
    # load
    fin = "lookup_table_up_" + fstring + ".h5"
    fin = os.path.join("lut", fin)
    lut_up = pd.read_hdf(fin, 'df')
    fin = "lookup_table_down_" + fstring + ".h5"
    fin = os.path.join("lut", fin)
    lut_down = pd.read_hdf(fin, 'df')
    # round
    lut_up['W'] = np.floor(lut_up['W']*1e8).astype(int)
    lut_down['W'] = np.floor(lut_down['W']*1e8).astype(int)
    # reindex
    lut_up.set_index('W', inplace=True, verify_integrity=True)
    lut_down.set_index('W', inplace=True, verify_integrity=True)
    # to dicts, > x10 speedup
    lut_up = lut_up.to_dict()['tt']
    lut_down = lut_down.to_dict()['tt']
    return lut_up, lut_down


def build_lut_f():
    """Build luts for a range of fs"""
    au = atomic_units()
    df = 1
    fs = range(0, 300+df, df)  # mV/cm
    imax = len(fs)
    c1 = time.clock()
    for i, f in enumerate(fs):
        progress("build_lut_f()", i, imax)
        lookup_table_f(f*au['mVcm'])
    c2 = time.clock()
    print((c2-c1)/imax)
    return
# ==========


# ==========
# timing
# numba.jit no speedup
# ==========
def time_tt():
    """For random field and energy, get tt_up and tt_down"""
    f = np.random.random()*300*1.94469e-13
    W = (np.random.random()*3 - 2)*150*1.51983e-07
    tt_up(W, f)
    tt_down(W, f)
    return


def timetest():
    """For random field f (0-300 mV/cm), get tt, tb for up, down"""
    f = np.random.random()*300*1.94469e-13  # 0 to 300 mV/cm
    tt_up_W(f)
    tt_down_W(f)
    tb_up_W(f)
    tb_down_W(f)
    return


def timing(n):
    """Ave run time of timetest, getting up/down tt/tb for a random field."""
    t = timeit.timeit(time_tt, number=int(n))
    return t/n


@numba.jit
def time_lookup():
    """Load lookup tables and time repeated index calls, print s/lookup."""
    # load
    # df = plot_conds()
    # lookup_table()
    lut_up, lut_down = import_lookup_table()
    n = int(1e7)
    i = np.random.choice(list(lut_up.keys()))
    c1 = time.clock()
    for j in range(n):
        lut_up[i]
    c2 = time.clock()
    result = print((c2 - c1)/n)
    return result
# ==========


# ==========
# Figure
# ==========
def plot_up():
    """Plot field vs. energy from 'fields_and_times.txt' for uphill elec."""
    au = atomic_units()
    df = pd.read_csv("fields_and_times.txt", index_col=0)
    # lab units
    Wkeys = ['tt_up', 'tt_down', 'tr_up', 'tr_down', 'tb_up', 'tb_down',
             'tp_up', 'il']
    df[Wkeys] = df[Wkeys]/au['GHz']
    df['f'] = df['f']/au['mVcm']
    fig, ax = plt.subplots()
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_up', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_up', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_up', ax=ax, label="$t_B = 20 ns$")
    df.plot(x='f', y='tp_up', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=(-20, 100),
           xlabel="Field (mV/cm)", ylabel="$W = v^2/2 - 1/r$  (GHz)",
           title="Uphill $e^-$")
    return df


def plot_down():
    """Plot field vs. energy from 'fields_and_times.txt' for uphill elec."""
    au = atomic_units()
    df = pd.read_csv("fields_and_times.txt", index_col=0)
    # lab units
    Wkeys = ['tt_up', 'tt_down', 'tr_up', 'tr_down', 'tb_up', 'tb_down',
             'tp_up', 'il']
    df[Wkeys] = df[Wkeys]/au['GHz']
    df['f'] = df['f']/au['mVcm']
    fig, ax = plt.subplots()
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_down', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_down', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_down', ax=ax, label="$t_B = 20 ns$")
    # df.plot(x='f', y='tp_down', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=(-70, 10),
           xlabel="Field (mV/cm)", ylabel="Orbit Energy (GHz)",
           title="Downhill $e^-$")
    return df


def plot_up_down():
    """Plot uphill and downhill figures with shared y-axis."""
    au = atomic_units()
    df = pd.read_csv("fields_and_times.txt", index_col=0)
    # lab units
    Wkeys = ['tt_up', 'tt_down', 'tr_up', 'tr_down', 'tb_up', 'tb_down',
             'tp_up', 'il']
    df[Wkeys] = df[Wkeys]/au['GHz']
    df['f'] = df['f']/au['mVcm']
    # plot
    fig, axes = plt.subplots(ncols=2, sharey=True)
    # uphill figure
    ax = axes[0]
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_up', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_up', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_up', ax=ax, label="$t_B = 20 ns$")
    df.plot(x='f', y='tp_up', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=(-70, 100),
           xlabel="Field (mV/cm)", ylabel="Orbit Energy (GHz)",
           title="Uphill $e^-$")
    # downhill figure
    ax = axes[1]
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_down', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_down', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_down', ax=ax, label="$t_B = 20 ns$")
    # df.plot(x='f', y='tp_down', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=(-70, 100),
           xlabel="Field (mV/cm)", ylabel="Orbit Energy (GHz)",
           title="Downhill $e^-$")
    fig.tight_layout()
    return df


def plot_conds():
    """Plot a 2x2 grid showing expected conditions at -5 and -21 GHz."""
    au = atomic_units()
    df = pd.read_csv("fields_and_times.txt", index_col=0)
    # lab units
    Wkeys = ['tt_up', 'tt_down', 'tr_up', 'tr_down', 'tb_up', 'tb_down',
             'tp_up', 'il']
    df[Wkeys] = df[Wkeys]/au['GHz']
    df['f'] = df['f']/au['mVcm']
    # plot
    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True)
    # ==========
    # DIL + 2 GHz, W + Delta W_MW = +38 GHz -> -48 GHz
    ylims = ((2-7) - 43 - 5, (2-7) + 43 + 5)
    # uphill
    ax = axes[0, 0]
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_up', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_up', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_up', ax=ax, label="$t_B = 20 ns$")
    df.plot(x='f', y='tp_up', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=ylims,
           xlabel="Field (mV/cm)", ylabel="Orbital Energy  (GHz)",
           title="Uphill $e^-$ at DIL + 2 GHz")
    ax.legend().remove()
    # downhill
    ax = axes[1, 0]
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_down', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_down', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_down', ax=ax, label="$t_B = 20 ns$")
    # df.plot(x='f', y='tp_down', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=ylims,
           xlabel="Field (mV/cm)", ylabel="Orbital Energy  (GHz)",
           title="Downhill $e^-$ at DIL + 2 GHz")
    ax.legend().remove()
    # ==========
    # DIL - 14 GHz, W + Delta W_MW =  GHz -> -41 GHz
    ylims = ((-14-7) - 43 - 5, (-14-7) + 43 + 5)
    # uphill
    ax = axes[0, 1]
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_up', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_up', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_up', ax=ax, label="$t_B = 20 ns$")
    df.plot(x='f', y='tp_up', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=ylims,
           xlabel="Field (mV/cm)",  # ylabel="Orbital Energy  (GHz)",
           title="Uphill $e^-$ at DIL - 14 GHz")
    ax.legend().remove()
    # downhill
    ax = axes[1, 1]
    ax.axhline(0, c='grey')
    ax.axvline(0, c='grey')
    df.plot(x='f', y='tt_down', ax=ax, label="$t_T = 20 ns$")
    df.plot(x='f', y='tr_down', ax=ax, label="$t_T = 10 ns$")
    df.plot(x='f', y='tb_down', ax=ax, label="$t_B = 20 ns$")
    # df.plot(x='f', y='tp_down', ax=ax, label="$t_U = 20 ns$")
    ax.set(xlim=(0, 100), ylim=ylims,
           xlabel="Field (mV/cm)",  # ylabel="Orbital Energy  (GHz)",
           title="Downhill $e^-$ at DIL - 14 GHz")
    ax.legend().remove()
    # ==========
    fig.tight_layout()
    return df
# ==========


def main():
    build_lut_f()
    return


if __name__ == '__main__':  # run if script is called directly
    result = main()
