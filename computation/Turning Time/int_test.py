# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 23:03:00 2018

@author: edmag
"""

from scipy.integrate import ode
from scipy.integrate import quad
import numpy as np


def atomic_units():
    """Return a dictionary of atomic units, ["GHz"], ["mVcm"], and ["ns"]"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
    return au


def tt_int(z, w, f):
    return -(2*(w - 1/z + f*z))**(-0.5)


def calc_tt(w, f):
    # system conditions
    # w = -10*au["GHz"]  # initial energy
    # f = 0*au["mVcm"]  # field
    zi = -6  # inner turning point
    # outer turning point
    if (f == 0) and (w < 0):
        zt = 1/w
        tt = quad(tt_int, zi, zt, args=(w, f))[0]
    elif (f == 0) and (w >= 0):
        zt = np.NaN
        tt = np.NaN
    elif f > 0:
        zt = -1/(2*f)*(w + (w**2 + 4*f)**0.5)
        tt = quad(tt_int, zi, zt, args=(w, f))[0]
    # integrate to find turning time
    tt = quad(tt_int, zi, zt, args=(w, f))[0]
    return tt


# import atomic units
au = atomic_units()
w = -1*au["GHz"]
f = 0*au["mVcm"]
print(calc_tt(w, f)/au["ns"])
