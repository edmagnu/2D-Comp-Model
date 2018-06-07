# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:19:00 2018

@author: edmag
"""

import turning_and_binding as tab
import Simple_1D as s1d
import random
import numpy as np
import pandas as pd
import multiprocessing
import time
import os


def bulk_f(W0s, Ep, Emw, w_mw, t0, toff, tring, tstop, Wlim, dup):
    au = tab.atomic_units()
    # lookup tables
    lut_up, lut_down = tab.import_lookup_table()
    lut_up_f, lut_down_f = tab.import_lookup_table_f(Ep)
    # final energies DataFrame
    n = 2000  # uncertainty of ~ 1%
    m = n*len(W0s)
    Wfs = np.ones(m)*np.NaN
    df = pd.DataFrame()
    df['Wf'] = Wfs
    df['W0'] = np.repeat(W0s, n)
    df['Ep'] = Ep
    df['Emw'] = Emw
    df['w_mw'] = w_mw
    df['t0'] = t0
    df['toff'] = toff
    df['tring'] = tring
    df['tstop'] = tstop
    df['Wlim'] = Wlim
    df['dup'] = dup
    # build results
    for i in df.index:
        tab.progress("bulk_f(): ", i, m)
        W0 = df.loc[i, 'W0']
        if np.isnan(df.loc[i, 'toff']):
            toff = (20 + 20*random.random())*au['ns']
            df.loc[i, 'toff'] = toff
        args = (W0, Ep, Emw, w_mw, t0, toff, tstop, Wlim, tring, dup,
                lut_up, lut_down, lut_up_f, lut_down_f)
        df.loc[i, 'Wf'] = s1d.run_to_stop(*args, tr=False)
    Epstring = str(int(round(Ep/au['mVcm'], 1)*10))
    Epstring = Epstring.zfill(4)
    fname = "wfinals" + "_" + Epstring + "_"
    if dup is True:
        fname = fname + "u"
    else:
        fname = fname + "d"
    fname = fname + ".h5"
    fname = os.path.join("wfinals", fname)
    print(fname)
    df.to_hdf(fname, 'df')
    return df


def main():
    au = tab.atomic_units()
    # Bulk settings
    W0s = np.arange(-100, 50 + 1, 1)*au['GHz']  # required for runtime
    Eps = np.array([61, 62, 63, 64, 66, 67, 68, 69, 71, 72, 73, 74, 76, 77, 78, 79])*au['mVcm']
    Emw = 4*1000*au['mVcm']
    w_mw = 2*np.pi*15.932/au['ns']
    t0 = 0*au['ns']
    # toff = 40*au['ns']
    toff = np.NaN  # set randomly inside bulk_f()
    tring = 180*au['ns']  # from "Data\MW Resonance\mw_analysis.py"
    tstop = toff + 5*tring
    Wlim = -600*au['GHz']  # one orbit ~ mw cycle
    dup = True
    c1 = time.clock()
    workers = []
    for Ep in Eps:
        args = (W0s, Ep, Emw, w_mw, t0, toff, tring, tstop, Wlim, dup)
        # df = bulk_f(W0s, Ep, Emw, w_mw, t0, toff, tring, tstop, Wlim, dup)
        workers = (workers + [
                multiprocessing.Process(target=bulk_f, args=(*args,))])
        # df = bulk_f(*args)
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    print(time.clock() - c1)
    return


if __name__ == '__main__':  # run if script is called directly
    multiprocessing.freeze_support()
    result = main()
