# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:18:37 2018

@author: edmag
"""

from numba.pycc import CC
import numpy as np


cc = numba.pycc.CC('integrands')


@cc.export('intg_up', 'f8(f8, f8, f8)')
# ==========
# integrands, dt/dz for up/down-hill electrons
# ==========
# @numba.jit('f8(f8, f8, f8)', nopython=True)  # x10 speedup
def intg_up(z, W, f):
    """Return dt/dz for uphill electron in Coulomb & Static Field

    non-keyword arguments
    z -- electron position (a.u.)
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)
    """
    return -1/np.sqrt(2*(W - 1/z + f*z))


@cc.export('intg_down', 'f8(f8, f8, f8)')
# @numba.jit('f8(f8, f8, f8)', nopython=True)  # x10 speedup
def intg_down(z, W, f):
    """Return dt/dz for downhill electron in Coulomb & Static Field

    non-keyword arguments
    z -- electron position (a.u.)
    W -- electron kinetic + Coulomb energy (a.u.)
    f -- static field strength (a.u.)
    """
    return 1/np.sqrt(2*(W + 1/z + f*z))
# ==========


if __name__ == '__main__':  # run if script is called directly
    cc.compile()
