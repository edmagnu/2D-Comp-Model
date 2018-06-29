# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:19:00 2018

@author: edmag
"""

import turning_and_binding as tab
import Simple_1D as s1d
import random
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import multiprocessing
import time
import os


def xticks_2p():
    """Return ticks and ticklabels starting at pi/6 separated by pi/2"""
    ticklabels = [r"$\pi/6$", r"$4\pi/6$", r"$7\pi/6$", r"$10\pi/6$"]
    ticks = [np.pi/6, 4*np.pi/6, 7*np.pi/6, 10*np.pi/6]
    return ticks, ticklabels


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
    Eps = np.array([61, 62, 63, 64, 66, 67, 68, 69, 71, 72, 73, 74, 76, 7, 78,
                    79])*au['mVcm']
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


def phase_filter(w0, dW, W, udn=0):
    arg = (W - w0)/dW
    phi = np.ones(len(arg))*np.NaN
    # Energy too high
    mask = (arg > 1)
    phi[mask] = np.inf
    # energy too low
    mask = (arg < -1)
    phi[mask] = -np.inf
    # energy in range
    mask = (arg <= 1) & (arg >= -1)
    phi[mask] = np.arccos(arg[mask])
    # up, low at pi/6, high at 7pi/6
    if udn == 1:
        phi = (np.pi - phi) + np.pi/6
    # down, high at pi/6, low at 7pi/6
    elif udn == -1:
        phi = phi + np.pi/6
    # if neither +/- 1, unspecified and just return phi
    return phi


def phase_filter_test():
    # fig for up and down
    fig, ax = plt.subplots(nrows=2, sharex=True)
    # uphill
    udn = 1  # uphill
    dW = 43  # 4 mV/cm MW field
    W0 = 0  # laser tuning
    # range of energy to consider, -100 -> 100 GHz
    df = pd.DataFrame({'W': np.arange(-100, 101, 1)})
    # get phis
    df['phi'] = phase_filter(W0, dW, df['W'], udn)
    # plotting
    df.plot(x='phi', y='W', ax=ax[0])
    xticks, xticklabels = xticks_2p()
    ax[0].set(xticks=xticks, xticklabels=xticklabels, title="Uphill")
    ax[0].legend().remove()
    # downhill
    udn = -1  # downhill
    df['phid'] = phase_filter(W0, dW, df['W'], udn)
    # plotting
    df.plot(x='phid', y='W', ax=ax[1])
    xticks, xticklabels = xticks_2p()
    ax[1].set(title="Downhill",
              xlabel="Phase (rad.)", ylabel="W (GHz)")
    ax[1].legend().remove()
    # reference lines
    for i in [0, 1]:
        ax[i].axvline(np.pi/6, c='k')
        ax[i].axvline(7*np.pi/6, c='k')
        ax[i].axhline(0, c='k')
        ax[i].axhline(-dW, c='k')
        ax[i].axhline(dW, c='k')
    return


def conv_model(x, x0):
    """Model of AM laser envelope to convolve over data["bound"].
    Returns np.array of 0.5 + np.cos(x + x0)"""
    return 0.5*(1 + np.cos(x - x0))


def laser_envelope(data):
    """Takes masked data, builds a laser envelope from -2pi to 4pi
    Returns DataFrame amlaser["phi", "I"]"""
    # Build phase from -2pi to 4pi
    phis = data['phi']
    lphi = len(phis)
    phis.index = range(0, lphi)
    phis_t = data['phi'] - 2*np.pi
    phis_t.index = range(-lphi, 0)
    phis = phis.append(phis_t)
    phis_t = data['phi'] + 2*np.pi
    phis_t.index = range(lphi, 2*lphi)
    phis = phis.append(phis_t)
    phis.sort_values(inplace=True)
    # build into amlaser
    amlaser = pd.DataFrame()
    amlaser['phi'] = phis
    amlaser['I'] = conv_model(amlaser['phi'], np.pi)/(lphi*0.5)
    return amlaser


def limit_model(W, DIL, fwhm):
    c = fwhm / (2*np.sqrt(2*np.log(2)))
    t = (W - DIL)/(np.sqrt(2)*c)
    return 0.5*erfc(t)


def limit_model_test():
    W = np.arange(-30, 21, 1)
    DIL = -7
    fwhm = 5
    vals = limit_model(W, DIL, fwhm)
    plt.plot(W, vals)
    for i in [DIL - fwhm/2, DIL, DIL + fwhm/2]:
        plt.axvline(i, c='grey', ls='dashed')
    for i in [0, 1]:
        plt.axhline(i, c='grey', ls='dashed')
    plt.xlabel("Tuning (GHz)")
    plt.ylabel("Signal")
    return


def field_analysis():
    au = tab.atomic_units()
    # figure
    fig, axes = plt.subplots(nrows=3)
    # settings
    fstr = "0100"
    Wlas = 0  # laser tuning
    dW = 43  # 4 mV/cm MW field
    udn = 1  # uphill
    field = float(fstr)/10
    # load data
    fname = os.path.join("wfinals", "wfinals_" + fstr + "_u.h5")
    df = pd.read_hdf(fname)
    # lab units
    df['W0'] = df['W0']/au['GHz']
    df['Wf'] = df['Wf']/au['GHz']
    # ==========
    # survival dataframe
    # ==========
    survival = pd.DataFrame()
    W0s = df['W0'].unique()
    survival['W0'] = W0s
    bounds = np.ones(len(W0s))*np.NaN
    # Get bound signal mean from each set of 2000 trials
    for i, W0 in enumerate(W0s):
        mask = (df['W0'] == W0)
        bounds[i] = sum(df.loc[mask, 'Wf'] < -7)/2000  # 2000 tests
    survival['bound'] = bounds
    # plot
    ax = axes[0]
    survival.plot(x='W0', y='bound', ax=ax)
    ax.set(xlabel="Orbit Energy (GHz)", ylabel="P(Survival)",
           title="E_P = {0} mV/cm".format(field))
    # ==========
    # phase
    # ==========
    survival['phi'] = phase_filter(Wlas, dW, survival['W0'].values, udn)
    # fold
    survival_a = survival.copy()
    survival_a['phi'] = 14/6*np.pi - survival_a['phi']  # 7pi/6 -> 13pi/6
    survival = survival.append(survival_a, ignore_index=True)
    mask = (survival['phi'] > 2*np.pi)
    survival.loc[mask, 'phi'] = survival.loc[mask, 'phi'] - 2*np.pi
    survival.sort_values(by='phi', inplace=True)
    # plot
    ax = axes[1]
    survival.plot(x='phi', y='bound', ax=ax)
    xticks, xticklabels = xticks_2p()
    ax.set(xticks=xticks, xticklabels=xticklabels,
           xlabel="Phase (rad)", ylabel="P(Survival)")
    # ==========
    # convolution
    # ==========
    # regularly spaced values
    phis = np.arange(0, 2*180 + 1, 1)*np.pi/180
    mask = (survival['phi'] != np.inf) & (survival['phi'] != -np.inf)
    xp = survival.loc[mask, 'phi'].values
    xp = np.append(xp, [xp[0] + 2*np.pi])
    xp = np.insert(xp, 0, xp[-1] - 2*np.pi)
    yp = survival.loc[mask, 'bound'].values
    yp = np.append(yp, [yp[0]])
    yp = np.insert(yp, 0, yp[-1])
    bounds = np.interp(phis, xp, yp)
    dfp = pd.DataFrame({'phi': phis, 'p': bounds})
    # convolve
    amlaser = laser_envelope(dfp)
    conv = np.convolve(dfp['p'], amlaser['I'], mode='same')
    dfp['conv'] = conv[range(len(dfp['phi']), 2*len(dfp['phi']))]
    # plot
    ax = axes[2]
    dfp.plot(x='phi', y='conv', ax=ax)
    ax.set(xticks=xticks, xticklabels=xticklabels,
           xlabel="Phase (rad)", ylabel="Signal")
    # clean
    for ax in axes:
        ax.legend().remove()
    fig.tight_layout()
    return survival


def field_ps(fstr):
    au = tab.atomic_units()
    # load
    fname = os.path.join("wfinals", "wfinals_" + fstr + "_u.h5")
    df = pd.read_hdf(fname)
    # lab units
    df['W0'] = df['W0']/au['GHz']
    df['Wf'] = df['Wf']/au['GHz']
    # survival
    surv = pd.DataFrame()
    W0s = df['W0'].unique()
    surv['W0'] = W0s
    bounds = np.ones(len(W0s))*np.NaN
    # Get bound signal mean from each set of 2000 trials
    DIL = -7
    fwhm = 5
    for i, W0 in enumerate(W0s):
        mask = (df['W0'] == W0)
        bounds[i] = sum(limit_model(df.loc[mask, 'Wf'], DIL, fwhm))/2000
    surv['p'] = bounds
    return surv


def build_heatmap():
    df = pd.DataFrame({'f': np.arange(0, 101, 1)})
    # fstrs = np.array(["    "]*len(fs))
    rec = pd.DataFrame()
    df['fstr'] = ""
    df['fname'] = ""
    for i in df.index:
        df.loc[i, 'fstr'] = str(int(df.loc[i, 'f']*10)).zfill(4)
        df.loc[i, 'fname'] = "wfinals" + "_" + df.loc[i, 'fstr'] + "_u.h5"
    for i in df.index:
        tab.progress("build_heatmap()", i, len(df.index))
        surv = field_ps(df.loc[i, 'fstr'])
        surv['f'] = df.loc[i, 'f']
        rec = rec.append(surv, ignore_index=True)
    rec = rec[['f', 'W0', 'p']]
    rec.sort_values(by=['f', 'W0'])
    rec.to_csv("heatmap_u.csv")
    return df, rec


def heatmap():
    df = pd.read_csv("heatmap_u.csv", index_col=0)
    # reogranize
    cols = df['f'].unique()
    rows = df['W0'].unique()
    # rows = np.flip(rows, axis=0)
    hm = pd.DataFrame(index=rows)
    for col in cols:
        mask = (df['f'] == col)
        hm[col] = df.loc[mask, 'p'].values
    # orient the plot properly
    hm = hm[::-1]
    ax = sns.heatmap(hm, cmap='Reds', vmin=0, vmax=1,
                     cbar_kws={'label': "P(Survival)"})
    ax.set(xlabel="Pulsed Field (mV/cm)", ylabel="W0 (GHz)",
           title="Uphill, DIL = -7 GHz")
    # df.plot.hexbin(x='f', y='W0', C='p')
    plt.tight_layout()
    plt.savefig("heatmap_u.pdf")
    return df, hm


def hm_convolution():
    dfhm = pd.read_csv("heatmap_u.csv", index_col=0)
    convrec = pd.DataFrame()
    fitrec = pd.DataFrame()
    # W0s = df['W0'].unique()
    fnum = len(dfhm['f'].unique())
    dW = 43
    Wlas = -50
    udn = 1
    for i, f in enumerate(dfhm['f'].unique()):
        tab.progress("hm_convolution()", i, fnum)
        mask = (dfhm['f'] == f)
        df = dfhm[mask].copy()
        df['phi'] = phase_filter(Wlas, dW, df['W0'].values, udn)
        # fold
        df_a = df.copy()
        df_a['phi'] = 14/6*np.pi - df_a['phi']  # 7pi/6 -> 13pi/6
        df = df.append(df_a, ignore_index=True)
        mask = (df['phi'] > 2*np.pi)
        df.loc[mask, 'phi'] = df.loc[mask, 'phi'] - 2*np.pi
        df.sort_values(by='phi', inplace=True)
        # convolution
        # regularly spaced values
        phis = np.arange(0, 2*180 + 1, 1)*np.pi/180
        mask = (df['phi'] != np.inf) & (df['phi'] != -np.inf)
        xp = df.loc[mask, 'phi'].values
        xp = np.append(xp, [xp[0] + 2*np.pi])
        xp = np.insert(xp, 0, xp[-1] - 2*np.pi)
        yp = df.loc[mask, 'p'].values
        yp = np.append(yp, [yp[0]])
        yp = np.insert(yp, 0, yp[-1])
        bounds = np.interp(phis, xp, yp)
        dfp = pd.DataFrame({'phi': phis, 'p': bounds})
        # convolve
        amlaser = laser_envelope(dfp)
        conv = np.convolve(dfp['p'], amlaser['I'], mode='same')
        dfp['conv'] = conv[range(len(dfp['phi']), 2*len(dfp['phi']))]
        # params
        y0 = np.mean(dfp['conv'])
        a = (max(dfp['conv']) - min(dfp['conv']))/2
        phi = dfp.loc[dfp['conv'] == max(dfp['conv']), 'phi'].iloc[0]
        # popt = [a, phi, y0]
        # record field
        dfp['f'] = f
        dfp['a'] = a
        dfp['y0'] = y0
        dfp['phi0'] = phi
        convrec = convrec.append(dfp, ignore_index=True)
        fitrec = fitrec.append(dfp.loc[0, ['f', 'a', 'y0', 'phi0']])
    # plot
    fig, axes = plt.subplots(nrows=3, sharex=True)
    fitrec.plot(x='f', y='y0', ax=axes[0])
    fitrec.plot(x='f', y='a', ax=axes[1])
    fitrec.plot(x='f', y='phi0', ax=axes[2])
    return convrec, fitrec


if __name__ == '__main__':  # run if script is called directly
    multiprocessing.freeze_support()
    # df = field_analysis()
    # df = field_ps("0050")
    # df, rec = build_heatmap()
    # df, hm = heatmap()
    dfconv, dffit = hm_convolution()
    # phase_filter_test()
    # limit_model_test()
    # result = main()
