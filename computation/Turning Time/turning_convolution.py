# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:10:00 2018

@author: labuser
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def atomic_units():
    """Return a dictionary of atomic units"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
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


def xticks_2p():
    """Return ticks and ticklabels starting at pi/6 separated by pi/2"""
    ticklabels = [r"$\pi/6$", r"$4\pi/6$", r"$7\pi/6$", r"$10\pi/6$"]
    ticks = [np.pi/6, 4*np.pi/6, 7*np.pi/6, 10*np.pi/6]
    return ticks, ticklabels


def phase_filter(w0, dW, W):
    arg = (W - w0)/dW
    if arg > 1:
        phi = -np.inf
    elif arg < -1:
        phi = np.inf
    else:
        phi = np.arccos(arg)
    return phi


def conv_model(x, x0):
    """Model of AM laser envelope to convolve over data["bound"].
    Returns np.array of 0.5 + np.cos(x + x0)"""
    return 0.5*(1 + np.cos(x - x0))


def laser_envelope(data):
    """Takes masked data, builds a laser envelope from -2pi to 4pi
    Returns DataFrame amlaser["phi", "I"]"""
    # Build phase from -2pi to 4pi
    phis = data["phi"]
    lphi = len(phis)
    phis.index = range(0, lphi)
    phis_t = data["phi"] - 2*np.pi
    phis_t.index = range(-lphi, 0)
    phis = phis.append(phis_t)
    phis_t = data["phi"] + 2*np.pi
    phis_t.index = range(lphi, 2*lphi)
    phis = phis.append(phis_t)
    phis.sort_values(inplace=True)
    # build into amlaser
    amlaser = pd.DataFrame()
    amlaser["phi"] = phis
    amlaser["I"] = conv_model(amlaser["phi"], np.pi)/(len(data)*0.5)
    return amlaser


# ==========
# Uphill
# ==========
def build_marks():
    au = atomic_units()
    xticks, xticklabels = xticks_2p()
    # import data
    fname = "picked_w.txt"
    picked_tot = pd.read_csv(fname, index_col=0)
    # dfdilb = pd.read_csv("dil_binding.txt", index_col=0)
    # convert to lab units
    picked_tot['W'] = picked_tot['W']/au['GHz']
    picked_tot['field'] = picked_tot['field']/au['mVcm']
    # build DataFrames of tt, tb, tp
    picked = picked_tot[picked_tot['Dir'] == -1].copy(deep=True)
    # NaN mask
    mask_NaN = (np.logical_not(np.isnan(picked['W'])))
    mask_NaN = mask_NaN & (np.logical_not(np.isnan(picked['field'])))
    # turning time = 10 ns
    mask = (picked['kind'] == 'tt=10')
    mask = mask & mask_NaN
    dftt10 = picked[mask].copy(deep=True)
    dftt10.sort_values(by='field', inplace=True)
    # tplus = 20
    mask = (picked['kind'] == 'tplus=20')
    mask = mask & mask_NaN
    dftp20 = picked[mask].copy(deep=True)
    dftp20.sort_values(by='field', inplace=True)
    # tb = 20
    mask = (picked['kind'] == 'tb=20')
    mask = mask & mask_NaN
    dftb20 = picked[mask].copy(deep=True)
    dftb20.sort_values(by='field', inplace=True)
    # massage tt10
    dftt10['field'] = np.round(dftt10['field'], 0)
    mask = (dftt10['field'] <= 100.0) & (dftt10['field'] >= 0.0)
    dftt10 = dftt10[mask][['field', 'W']]
    dftt10.sort_values(by='field')
    # massage tb20
    dftb20['field'] = np.round(dftb20['field'], 0)
    dftb20 = dftb20[['field', 'W']]
    dftb20 = dftb20.append({'field': 0, 'W': 0}, ignore_index=True)
    dftb20 = dftb20.append(pd.DataFrame({'field': np.arange(55, 101),
                                         'W': np.ones(46)*np.inf}),
                           ignore_index=True)
    dftb20 = dftb20[['field', 'W']]
    dftb20.sort_values(by='field', inplace=True)
    # massage tp20
    dftp20['field'] = np.round(dftp20['field'], 0)
    dftp20 = dftp20[['field', 'W']]
    dftp20 = dftp20.append(pd.DataFrame({'field': np.arange(64, 101),
                                         'W': np.ones(37)*np.inf}),
                           ignore_index=True)
    mask = dftt10['field'] < 15.0
    dftp20 = dftp20.append(dftt10[mask], ignore_index=True)
    dftp20 = dftp20[['field', 'W']]
    dftp20.sort_values(by='field', inplace=True)
    # indexes
    dftt10.reset_index(drop=True, inplace=True)
    dftp20.reset_index(drop=True, inplace=True)
    dftb20.reset_index(drop=True, inplace=True)
    # combine into dataframe of marked energies
    dfmarks = pd.DataFrame(dftt10['field'])
    dfmarks['tt10'] = dftt10['W']
    dfmarks['tp20'] = dftp20['W']
    dfmarks['tb20'] = dftb20['W']
    return dfmarks


def prob_and_conv(dfmarks, dW, W0, f, dphi):
    # pick observation
    obs = dfmarks.loc[f]
    for key in ['tt10', 'tp20', 'tb20']:
        phi = phase_filter(W0, dW, obs[key])
        obs[key] = (np.pi - phi) + np.pi/6
    # build simple probability
    dfprob = pd.DataFrame({'phi': np.arange(np.pi/6, 7*np.pi/6 + dphi, dphi)})
    dfprob['p'] = np.nan
    mask = (dfprob['phi'] < obs['tt10'])
    dfprob.loc[mask, 'p'] = 0.5
    mask = (dfprob['phi'] >= obs['tt10']) & (dfprob['phi'] < obs['tp20'])
    dfprob.loc[mask, 'p'] = 0
    mask = (dfprob['phi'] >= obs['tp20']) & (dfprob['phi'] < obs['tb20'])
    dfprob.loc[mask, 'p'] = 1  # goldilocks
    mask = (dfprob['phi'] >= obs['tb20'])
    dfprob.loc[mask, 'p'] = 0
    # fold
    dfprob_a = dfprob.copy()
    dfprob_a.drop([0, 180], inplace=True)  # don't repeat endpoints
    dfprob_a['phi'] = 14/6*np.pi - dfprob_a['phi']
    dfprob = dfprob.append(dfprob_a, ignore_index=True)
    dfprob['phi'] = np.mod(dfprob['phi'], 2*np.pi)
    dfprob = dfprob.sort_values(by='phi')
    # convolution
    amlaser = laser_envelope(dfprob)
    conv = np.convolve(dfprob['p'], amlaser['I'], mode='same')
    dfprob['conv'] = conv[range(len(dfprob['phi']), 2*len(dfprob['phi']))]
    # parameters
    y0 = np.mean(dfprob['conv'])
    a = (max(dfprob['conv']) - min(dfprob['conv']))/2
    phi = dfprob.loc[dfprob['conv'] == max(dfprob['conv']), 'phi'].iloc[0]
    popt = [a, phi, y0]
    return dfprob, popt


def prob_and_conv_dil(dfmarks, dW, W0, f, dphi):
    # pick observation
    obs = dfmarks.loc[f]
    dfdilb = pd.read_csv("dil_binding.txt", index_col=0)
    if f <= 10:
        obs['wi'] = dfdilb.loc[f, 'wi']
        obs['wf'] = dfdilb.loc[f, 'wf']
    else:
        obs['wi'] = np.NaN
        obs['wf'] = np.NaN
    # print(obs)
    obs_W = obs.copy()
    for key in ['tt10', 'tp20', 'tb20', 'wi', 'wf']:
        phi = phase_filter(W0, dW, obs[key])
        obs[key] = (np.pi - phi) + np.pi/6
    # build simple probability
    dfprob = pd.DataFrame({'phi': np.arange(np.pi/6, 7*np.pi/6 + dphi, dphi)})
    dfprob['W'] = W0 - np.cos(dfprob['phi'] - np.pi/6)*dW
    dfprob['p'] = np.nan
    if f > 10:
        mask = (dfprob['phi'] < obs['tt10'])
        dfprob.loc[mask, 'p'] = 0.5
        dfprob.loc[np.logical_not(mask), 'p'] = 0.0
        # mask = (dfprob['phi'] >= obs['tt10']) & (dfprob['phi'] < obs['tp20'])
        # dfprob.loc[mask, 'p'] = 0
        # mask = (dfprob['phi'] >= obs['tp20']) & (dfprob['phi'] < obs['tb20'])
        # dfprob.loc[mask, 'p'] = 1  # goldilocks
        # mask = (dfprob['phi'] >= obs['tb20'])
        # dfprob.loc[mask, 'p'] = 0
    if f <= 10:
        mask = (dfprob['phi'] < obs['tt10'])
        dfprob.loc[mask, 'p'] = 0.5
        if np.isnan(obs['wi']):
            mask = (dfprob['phi'] < obs['wf']) & (dfprob['phi'] >= obs['tt10'])
            dfprob.loc[mask, 'p'] = 0  # goldilocks
        else:
            mask = (dfprob['phi'] < obs['wf']) & (dfprob['phi'] >= obs['wi'])
            dfprob.loc[mask, 'p'] = 0  # goldilocks
            mask = (dfprob['phi'] < obs['wi']) & (dfprob['phi'] >= obs['tt10'])
            dfprob.loc[mask, 'p'] = 0
        mask = (dfprob['phi'] >= obs['wf'])
        dfprob.loc[mask, 'p'] = 0
    # mask = ((dfprob['phi'] < obs['tt10']) &
    #         (dfprob['phi'] >= (obs['tt10'] - np.pi)))
    mask = (dfprob['W'] < obs_W['tt10'])
    # print("W(tt10) = ", obs_W['tt10'])
    Wdist = np.abs(dfprob.loc[mask, 'W'] - obs_W['tt10'])
    Wdist[Wdist > dW] = dW
    pdist = np.arccos(Wdist/dW)
    pdist[np.isnan(pdist)] = np.pi
    val = 0.5 - 0.5*pdist/np.pi
    # print(len(dfprob[mask]), len(val))
    dfprob.loc[mask, 'p'] = val
    # plt.plot(dfprob.loc[mask, 'phi'], dist)
    # dfprob.loc[mask, 'p'] = (1 - dfprob['phi'] - obs['tt10'])/(2*np.pi))*0.5
    # fold
    dfprob_a = dfprob.copy()
    dfprob_a.drop([0, 180], inplace=True)  # don't repeat endpoints
    dfprob_a['phi'] = 14/6*np.pi - dfprob_a['phi']
    dfprob = dfprob.append(dfprob_a, ignore_index=True)
    dfprob['phi'] = np.mod(dfprob['phi'], 2*np.pi)
    dfprob = dfprob.sort_values(by='phi')
    # convolution
    amlaser = laser_envelope(dfprob)
    conv = np.convolve(dfprob['p'], amlaser['I'], mode='same')
    dfprob['conv'] = conv[range(len(dfprob['phi']), 2*len(dfprob['phi']))]
    # parameters
    y0 = np.mean(dfprob['conv'])
    a = (max(dfprob['conv']) - min(dfprob['conv']))/2
    phi = dfprob.loc[dfprob['conv'] == max(dfprob['conv']), 'phi'].iloc[0]
    popt = [a, phi, y0]
    return dfprob, popt


def tt_conv_plot(dW, W0, f, dphi):
    xticks, xticklabels = xticks_2p()
    dfmarks = build_marks()
    fig, ax = plt.subplots(nrows=3, figsize=(6, 9))
    dfmarks.plot(x='field', y='tt10', linestyle='', marker='.', label='tt10',
                 alpha=0.5, ax=ax[0])
    dfmarks.plot(x='field', y='tb20', linestyle='', marker='.', label='tb20',
                 alpha=0.5, ax=ax[0])
    dfmarks.plot(x='field', y='tp20', linestyle='', marker='.', label='tp20',
                 alpha=0.5, ax=ax[0])
    ax[0].legend()
    ax[0].set(ylabel="Energy (GHz)", xlabel="Field (mV/cm)",
              title="Uphill Turning")
    # mark
    ax[0].axvline(f, color='grey')
    for W in [W0-dW, W0, W0+dW]:
        ax[0].axhline(W, color='grey')
    # get probability and convolution
    dfprob, popt = prob_and_conv(dfmarks, dW, W0, f, dphi)
    dfprob.plot(x='phi', y='p', ax=ax[1])
    ax[1].set(xticks=xticks, xticklabels=xticklabels, xlabel="Phase (rad.)",
              ylabel=r"$P_{Survival}$", title="Simple Probability")
    ax[1].legend().remove()
    dfprob.plot(x='phi', y='conv', ax=ax[2])
    ax[2].axvline(popt[1], color='k')
    ax[2].axhline(popt[2], color='k')
    ax[2].axhline(popt[0] + popt[2], color='k')
    ax[2].set(xticks=xticks, xticklabels=xticklabels, xlabel="Phase (rad.)",
              ylabel="Norm. Signal", title="Expected Signal")
    fig.tight_layout()
    # fit to model
    return dfprob


def params_bulk(dW, W0, dphi):
    # up
    dfmarks = build_marks()
    params = dfmarks.copy()
    params['Dir'] = -1
    params['dW'] = dW
    params['W0'] = W0
    params['a'] = np.nan
    params['phi'] = np.nan
    params['y0'] = np.nan
    for i in dfmarks.index:
        f = dfmarks.loc[i, 'field']
        dfprob, popt = prob_and_conv_dil(dfmarks, dW, W0, f, dphi)
        if popt[1] >= 4/6*np.pi:
            params.loc[i, 'a'] = -popt[0]
        else:
            params.loc[i, 'a'] = popt[0]
        params.loc[i, 'phi'] = popt[1]
        params.loc[i, 'y0'] = popt[2]
    params = params[['Dir', 'field', 'dW', 'W0', 'a', 'phi', 'y0']]
    params_up = params.copy()
    # down
    dfmarks = build_marks_down()
    params = dfmarks.copy()
    params['Dir'] = 1
    params['dW'] = dW
    params['W0'] = W0
    params['a'] = np.nan
    params['phi'] = np.nan
    params['y0'] = np.nan
    for i in dfmarks.index:
        f = dfmarks.loc[i, 'field']
        dfprob, popt = prob_and_conv_down(dfmarks, dW, W0, f, dphi)
        if popt[1] >= 4/6*np.pi:
            params.loc[i, 'a'] = -popt[0]
        else:
            params.loc[i, 'a'] = popt[0]
        params.loc[i, 'phi'] = popt[1]
        params.loc[i, 'y0'] = popt[2]
    params = params[['Dir', 'field', 'dW', 'W0', 'a', 'phi', 'y0']]
    params_down = params.copy()
    # massage
    params = params_up.append(params_down)
    params = params.sort_values(by=['Dir', 'field'])
    params = params[['Dir', 'field', 'dW', 'W0', 'a', 'phi', 'y0']]
    folder = "simple_conv"
    fname = "tconv_params_" + str(int(np.round(W0, 0))) + ".csv"
    fname = os.path.join(folder, fname)
    print(fname)
    params.to_csv(fname)
    return params


def bulk_plot(params_tot):
    # startup
    xticks, xticklabels = xticks_2p()
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex='col', sharey='row',
                           figsize=(8, 10.5))
    W0 = params_tot['W0'].unique()[0]
    # up
    mask = (params_tot['Dir'] == -1)
    params = params_tot[mask]
    params.plot(x='field', y='y0', ax=ax[0, 0])
    ax[0, 0].set(title=r"Uphill, $W_0$ = {} GHz".format(int(W0)),
                 ylabel="Mean")
    params.plot(x='field', y='a', ax=ax[1, 0])
    ax[1, 0].set(ylabel="Amplitude")
    params.plot(x='field', y='phi', ax=ax[2, 0])
    ax[2, 0].set(yticks=xticks, yticklabels=xticklabels, ylabel="Phase")
    for i in [0, 1, 2]:
        ax[i, 0].grid()
        ax[i, 0].legend().remove()
    # down
    mask = (params_tot['Dir'] == 1)
    params = params_tot[mask]
    params.plot(x='field', y='y0', ax=ax[0, 1])
    ax[0, 1].set(title=r"Downhill, $W_0$ = {} GHz".format(int(W0)),
                 ylabel="Mean")
    params.plot(x='field', y='a', ax=ax[1, 1])
    ax[1, 1].set(ylabel="Amplitude")
    params.plot(x='field', y='phi', ax=ax[2, 1])
    ax[2, 1].set(yticks=xticks, yticklabels=xticklabels, ylabel="Phase")
    for i in [0, 1, 2]:
        ax[i, 1].grid()
        ax[i, 1].legend().remove()
    # combination
    mask = (params_tot['Dir'] == -1)
    params_up = params_tot[mask]
    mask = (params_tot['Dir'] == 1)
    params_down = params_tot[mask]
    params_both = params_up.drop(labels='Dir', axis=1).copy()
    params_both['y0'] = (params_up['y0'] + params_down['y0'])/2
    params_both['a'] = (params_up['a'] + params_down['a'])/2
    params = params_both.copy()
    params.plot(x='field', y='y0', ax=ax[0, 2])
    ax[0, 2].set(title=r"Both, $W_0$ = {} GHz".format(int(W0)),
                 ylabel="Mean")
    params.plot(x='field', y='a', ax=ax[1, 2])
    ax[1, 2].set(ylabel="Amplitude")
    # params.plot(x='field', y='phi', ax=ax[2, 2])
    # ax[2, 2].set(yticks=xticks, yticklabels=xticklabels, ylabel="Phase")
    for i in [0, 1]:
        ax[i, 2].grid()
        ax[i, 2].legend().remove()
    for i in [0, 1, 2]:
        ax[i, 2].set(xlabel="Field (mV/cm)")
    fig.tight_layout()
    folder = "simple_conv"
    fname = "tconv_params_" + str(int(np.round(W0, 0))) + "_ysgl.pdf"
    fname = os.path.join(folder, fname)
    plt.savefig(fname)
    return


# ==========
# Downhill
# ==========
def build_marks_down():
    au = atomic_units()
    # load
    picked_tot = pd.read_csv("picked_w.txt", index_col=0)
    xmin = 0
    xmax = 100
    # convert to lab units
    picked_tot['W'] = picked_tot['W']/au['GHz']
    picked_tot['field'] = picked_tot['field']/au['mVcm']
    # downhill
    picked = picked_tot[picked_tot['Dir'] == 1].copy(deep=True)
    # NaN mask
    mask_NaN = (np.logical_not(np.isnan(picked['W'])))
    mask_NaN = mask_NaN & (np.logical_not(np.isnan(picked['field'])))
    # DIL W = -2 E^0.5
    dfdil = pd.DataFrame({'field': np.arange(xmin, xmax+1, 1)})
    dfdil['W'] = -2*np.sqrt(dfdil['field']*au['mVcm'])/au['GHz']
    dfdil['field'] = np.round(dfdil['field'], 0)
    dfdil.sort_values(by='field', inplace=True)
    # tb = 20 ns
    mask = (picked['kind'] == 'tb=20')
    mask = mask & mask_NaN
    dftb20 = picked[mask].copy(deep=True)
    dftb20 = dftb20[['field', 'W']]
    dftb20 = dftb20.append({'field': 0, 'W': 0}, ignore_index=True)
    dftb20['field'] = np.round(dftb20['field'], 0)
    dftb20.sort_values(by='field', inplace=True)
    x = np.arange(min(dftb20['field']), max(dftb20['field']) + 1, 1)
    y = np.interp(x, dftb20['field'], dftb20['W'])
    dftb20 = pd.DataFrame({'field': x, 'W': y})
    mask = dfdil['field'] > max(dftb20['field'])
    dftb20 = dftb20.append(dfdil[mask], ignore_index=True)
    dftb20.sort_values(by='field', inplace=True)
    # tt = 10 ns
    mask = (picked['kind'] == 'tt=10')
    mask = mask & mask_NaN
    dftt10 = picked[mask].copy(deep=True)
    dftt10['field'] = np.round(dftt10['field'], 0)
    dftt10.sort_values(by='field', inplace=True)
    x = np.arange(min(dftt10['field']), max(dftt10['field']) + 1, 1)
    y = np.interp(x, dftt10['field'], dftt10['W'])
    dftt10 = pd.DataFrame({'field': x, 'W': y})
    mask = dfdil['field'] > max(dftt10['field'])
    dftt10 = dftt10.append(dfdil[mask], ignore_index=True)
    dftt10.sort_values(by='field', inplace=True)
    # reindex
    dfdil.reset_index(drop=True, inplace=True)
    dftt10.reset_index(drop=True, inplace=True)
    dftb20.reset_index(drop=True, inplace=True)
    # marks
    dfmarks = pd.DataFrame(dfdil['field'])
    dfmarks['DIL'] = dfdil['W']
    dfmarks['tb20'] = dftb20['W']
    dfmarks['tt10'] = dftt10['W']
    return dfmarks


def tt_conv_plot_down(dW, W0, f, dphi):
    xticks, xticklabels = xticks_2p()
    # build
    dfmarks = build_marks_down()
    dfprob, popt = prob_and_conv_down(dfmarks, dW, W0, f, dphi)
    # plots
    fig, ax = plt.subplots(nrows=3, figsize=(6, 9))
    # marks
    dfmarks.plot(x='field', y='DIL', linestyle='-', marker='.', label="DIL",
                 alpha=0.5, ax=ax[0])
    dfmarks.plot(x='field', y='tt10', linestyle='-', marker='.', label="tt10",
                 alpha=0.5, ax=ax[0])
    dfmarks.plot(x='field', y='tb20', linestyle='-', marker='.', label="tb20",
                 alpha=0.5, ax=ax[0])
    ax[0].axvline(f, color='grey')
    for W in [W0-dW, W0, W0+dW]:
        ax[0].axhline(W, color='grey')
    ax[0].legend()
    ax[0].set(xlabel="Field (mV/cm)", ylabel="Energy (GHz)",
              title="Downhill Turning")
    # probability
    dfprob.plot(x='phi', y='p', ax=ax[1])
    ax[1].set(xticks=xticks, xticklabels=xticklabels,
              ylabel=r"$P_{Survival}$", title="Simple Probability")
    # convolution
    dfprob.plot(x='phi', y='conv', ax=ax[2])
    ax[2].axvline(popt[1], color='k')
    ax[2].axhline(popt[2], color='k')
    ax[2].axhline(popt[0] + popt[2], color='k')
    ax[2].set(xticks=xticks, xticklabels=xticklabels, xlabel="Phase (rad.)",
              ylabel="Norm. Signal", title="Expected Signal")
    fig.tight_layout()
    return


def prob_and_conv_down(dfmarks, dW, W0, f, dphi):
    # pick observation
    obs = dfmarks.loc[f].copy()
    obs_W = obs.copy()
    for key in ['DIL', 'tb20', 'tt10']:
        phi = phase_filter(W0, -dW, obs[key])
        obs[key] = (np.pi - phi) + np.pi/6
    # build simple probability
    dfprob = pd.DataFrame({'phi': np.arange(np.pi/6, 7*np.pi/6 + dphi, dphi)})
    dfprob['W'] = W0 + np.cos(dfprob['phi'] - np.pi/6)*dW
    dfprob['p'] = np.nan
    mask = (dfprob['phi'] > obs['tt10'])
    dfprob.loc[mask, 'p'] = 0.5
    mask = (dfprob['phi'] <= obs['tt10']) & (dfprob['phi'] > obs['tb20'])
    dfprob.loc[mask, 'p'] = 0  # goldilocks
    mask = (dfprob['phi'] <= obs['tb20'])
    dfprob.loc[mask, 'p'] = 0
    # 2nd exchange
    mask = (dfprob['W'] < obs_W['tt10'])
    Wdist = np.abs(dfprob.loc[mask, 'W'] - obs_W['tt10'])
    Wdist[Wdist > dW] = dW
    pdist = np.arccos(Wdist/dW)
    pdist[np.isnan(pdist)] = np.pi
    val = 0.5 - 0.5*pdist/np.pi
    # print(len(dfprob[mask]), len(val))
    dfprob.loc[mask, 'p'] = val
    # fold
    dfprob_a = dfprob.copy()
    dfprob_a.drop([0, 180], inplace=True)  # don't repeat endpoints
    dfprob_a['phi'] = 14/6*np.pi - dfprob_a['phi']
    dfprob = dfprob.append(dfprob_a, ignore_index=True)
    dfprob['phi'] = np.mod(dfprob['phi'], 2*np.pi)
    dfprob = dfprob.sort_values(by='phi')
    # convolution
    amlaser = laser_envelope(dfprob)
    conv = np.convolve(dfprob['p'], amlaser['I'], mode='same')
    dfprob['conv'] = conv[range(len(dfprob['phi']), 2*len(dfprob['phi']))]
    # parameters
    y0 = np.mean(dfprob['conv'])
    a = (max(dfprob['conv']) - min(dfprob['conv']))/2
    phi = dfprob.loc[dfprob['conv'] == max(dfprob['conv']), 'phi'].iloc[0]
    popt = [a, phi, y0]
    return dfprob, popt


def tt_conv_plot_both(dW, W0, f, dphi):
    xticks, xticklabels = xticks_2p()
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row',
                           figsize=(9, 6))
    # ==========
    # Uphill
    # ==========
    # build
    dfmarks = build_marks()
    dfprob, popt = prob_and_conv(dfmarks, dW, W0, f, dphi)
    # probability
    dfprob.plot(x='phi', y='p', ax=ax[0, 0])
    ax[0, 0].set(xticks=xticks, xticklabels=xticklabels,
                 ylabel=r"$P_{Survival}$", title="Uphill")
    # convolution
    dfprob.plot(x='phi', y='conv', ax=ax[1, 0])
    ax[1, 0].set(xticks=xticks, xticklabels=xticklabels, xlabel="Phase (rad.)",
                 ylabel="Norm. Signal")
    dfprob_up = dfprob.copy()
    # ==========
    # Downhill
    # ==========
    dfmarks = build_marks_down()
    dfprob, popt = prob_and_conv_down(dfmarks, dW, W0, f, dphi)
    # probability
    dfprob.plot(x='phi', y='p', ax=ax[0, 1])
    ax[0, 1].set(xticks=xticks, xticklabels=xticklabels,
                 ylabel=r"$P_{Survival}$", title="Downhill")
    # convolution
    dfprob.plot(x='phi', y='conv', ax=ax[1, 1])
    ax[1, 1].set(xticks=xticks, xticklabels=xticklabels, xlabel="Phase (rad.)",
                 ylabel="Norm. Signal")
    dfprob_down = dfprob.copy()
    # ==========
    # Both
    # ==========
    dfprob['p'] = (dfprob_up['p'] + dfprob_down['p'])/2
    dfprob['conv'] = (dfprob_up['conv'] + dfprob_down['conv'])/2
    # probability
    dfprob.plot(x='phi', y='p', ax=ax[0, 2])
    ax[0, 2].set(xticks=xticks, xticklabels=xticklabels,
                 ylabel=r"$P_{Survival}$", title="Combined")
    # convolution
    dfprob.plot(x='phi', y='conv', ax=ax[1, 2])
    ax[1, 2].set(xticks=xticks, xticklabels=xticklabels, xlabel="Phase (rad.)",
                 ylabel="Norm. Signal")
    dfprob_down = dfprob.copy()
    # ==========
    # finish
    for i in [0, 1, 2]:
        ax[0, i].legend().remove()
        ax[1, i].legend().remove()
    suptitle = (r"Simple Model, $E_{pulsed}$ = " +
                r"{0} mV/cm, $W_0$ = {1} GHz".format(int(f), int(W0)))
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join("simple_conv", "{0}mVcm_{1}GHz_both.pdf".format(
            int(f), int(W0)))
    fig.savefig(fname)
    return dfprob_up, dfprob_down


# ==========
# up_and_down_orbits.pdf
# ==========
def interp_match(df1, df2, kx, ky):
    x = np.intersect1d(df1[kx], df2[kx])
    y1 = np.interp(x, df1[kx], df1[ky])
    y2 = np.interp(x, df2[kx], df2[ky])
    return x, y1, y2


def turning_time_figure(field_picks, W0_pick):
    au = atomic_units()
    # import data
    # dname = os.path.join("..", "2D-Comp-Model", "computation",
    #                      "Turning Time")
    # fname = os.path.join(dname, "data_raw.txt")
    data_tot = pd.read_csv("data_raw.txt", index_col=0)
    # fname = os.path.join(dname, "picked_w.txt")
    picked_tot = pd.read_csv("picked_w.txt", index_col=0)
    # convert to lab units
    picked_tot['W'] = picked_tot['W']/au['GHz']
    picked_tot['field'] = picked_tot['field']/au['mVcm']
    # plot
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3), sharey=True)
    ymin, ymax = -100, 100
    xmin, xmax = 0, 100
    # ==========
    # uphill figure
    # ==========
    ax = axes[0]
    colors = ['C2', 'C3', 'C9']
    # mask uphill values (Dir == -1)
    picked = picked_tot[picked_tot['Dir'] == -1].copy(deep=True)
    # NaN mask
    mask_NaN = (np.logical_not(np.isnan(picked['W'])))
    mask_NaN = mask_NaN & (np.logical_not(np.isnan(picked['field'])))
    # turning time = 10 ns
    mask = (picked['kind'] == 'tt=10')
    mask = mask & mask_NaN
    dftt10 = picked[mask].copy(deep=True)
    dftt10.sort_values(by='field', inplace=True)
    dftt10.plot(x='field', y='W', label='tturn=10', color='k', lw=3, ax=ax)
    # upper binding time = 20 ns
    mask = (picked['kind'] == 'tplus=20')
    mask = mask & mask_NaN
    dftplus = picked[mask].copy(deep=True)
    dftplus.sort_values(by='field', inplace=True)
    dftplus.plot(x='field', y='W', label='tplus=20', color='k', lw=3, ax=ax)
    # lower binding time = 20 ns
    mask = (picked['kind'] == 'tb=20')
    mask = mask & mask_NaN
    dftbind = picked[mask].copy(deep=True)
    obs = dftbind.iloc[-1].copy(deep=True)
    obs['W'] = 0
    obs['field'] = 0
    dftbind = dftbind.append(obs, ignore_index=True).copy(deep=True)
    dftbind.sort_values(by='field', inplace=True)
    dftbind.plot(x='field', y='W', label='tb=20', color='k', lw=3, ax=ax)
    # fill between
    # If turning time < 10 ns, return is < 20 ns, survival is chaotic
    ax.fill_between(dftt10['field'], dftt10['W'], ymin, color=colors[2])
    # If between tbind and tplus, it's in a safe "long orbit"
    x, y1, y2 = interp_match(dftbind, dftplus, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[0])
    mask = (dftplus['field'] >= max(x))
    ax.fill_between(dftplus[mask]['field'], ymax, dftplus[mask]['W'],
                    color=colors[0])
    mask = (dftbind['field'] <= min(x))
    x, y1, y2 = interp_match(dftbind[mask], dftt10, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[0])
    # If above tbind or between tt10, it's lost
    ax.fill_between(dftbind['field'], ymax, dftbind['W'], color=colors[1])
    x, y1, y2 = interp_match(dftplus, dftt10, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[1])
    mask = (dftt10['field'] >= max(x))
    ax.fill_between(dftt10[mask]['field'], ymax, dftt10[mask]['W'],
                    color=colors[1])
    # text labels
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    align = {'verticalalignment': 'top',
             'horizontalalignment': 'center'}
    ax.text(80, 95, "(d)", **align, bbox=props)  # chaotic
    ax.text(50, 95, "(c)", **align, bbox=props)  # Immediate Ionization
    ax.text(32, 95, "(b)", **align, bbox=props)  # Goldylocks Zone
    ax.text(10, 95, "(a)", **align, bbox=props)  # Late Return & Ionization
    # touch ups
    ax.set(xlabel="Pulsed Field (mV/cm)", xlim=(xmin, xmax),
           ylabel=r"$W_0 + \Delta W_{MW}(\phi)$ (GHz)", ylim=(ymin, ymax),
           title="Uphill Electrons")
    # add lines
    for f in field_picks:
        ax.axvline(f, color='k', lw=3, linestyle='dashed')
    ax.axhline(W0_pick, color='k', lw=3, linestyle='dashed')
    ax.legend().remove()
    # ==========
    # downhill figure
    # ==========
    ax = axes[1]
    colors = ['C2', 'C3', 'C9']
    # mask uphill values (Dir == 1)
    picked = picked_tot[picked_tot['Dir'] == 1].copy(deep=True)
    # NaN mask
    mask_NaN = (np.logical_not(np.isnan(picked['W'])))
    mask_NaN = mask_NaN & (np.logical_not(np.isnan(picked['field'])))
    # lower binding time = 20 ns
    mask = (picked['kind'] == 'tb=20')
    mask = mask & mask_NaN
    dftbind = picked[mask].copy(deep=True)
    obs = dftbind.iloc[-1].copy(deep=True)
    obs['W'] = 0
    obs['field'] = 0
    dftbind = dftbind.append(obs, ignore_index=True)
    dftbind.sort_values(by='field', inplace=True)
    dftbind.plot(x='field', y='W', label='tb=20', color='k', lw=3, ax=ax)
    # turning time = 10 ns
    mask = (picked['kind'] == 'tt=10')
    mask = mask & mask_NaN
    dftt10 = picked[mask].copy(deep=True)
    dftt10.sort_values(by='field', inplace=True)
    obs = dftbind.iloc[-4]
    dftt10 = dftt10.append(obs, ignore_index=True)
    dftt10.plot(x='field', y='W', label='tturn=10', color='k', lw=3, ax=ax)
    # DIL W = -2 E^0.5
    dfdil = pd.DataFrame({'field': np.arange(xmin, xmax+1, 1)})
    dfdil['W'] = -2*np.sqrt(dfdil['field']*au['mVcm'])/au['GHz']
    mask = (dfdil['field'] >= max(dftt10['field']))
    dfdil = dfdil[mask]
    dfdil.plot(x='field', y='W', label='DIL', color='k', lw=3, ax=ax)
    # fill between
    # Above binding time = 20 ns, Immediately Ionizes
    ax.fill_between(dftbind['field'], ymax, dftbind['W'], color=colors[1])
    ax.fill_between(dfdil['field'], ymax, dfdil['W'], color=colors[1])
    # Below turning time = 10 ns, Chaotic
    ax.fill_between(dftt10['field'], dftt10['W'], ymin, color=colors[2])
    mask = (dftbind['field'] >= max(dftt10['field']))
    ax.fill_between(dftbind[mask]['field'], dftbind[mask]['W'], ymin,
                    color=colors[2])
    ax.fill_between(dfdil['field'], dfdil['W'], ymin, color=colors[2])
    # Between binding time = 20ns and turnign time = 10 ns, Goldylocks
    x, y1, y2 = interp_match(dftbind, dftt10, 'field', 'W')
    ax.fill_between(x, y1, y2, color=colors[0])
    # text labels
    props = dict(boxstyle='round', facecolor='white', alpha=1.0)
    align = {'verticalalignment': 'center',
             'horizontalalignment': 'left'}
    ax.text(2.5, -50, "(d)", **align, bbox=props)  # chaotic
    # ax.text(0, 50, "(c)", **align, bbox=props)  # Late Return and Ionize
    ax.text(10, 0, "(b)", **align, bbox=props)  # Goldylocks Zone
    ax.text(2.5, 50, "(a)", **align, bbox=props)  # Immediate Ionization
    # touch ups
    ax.set(xlabel="Pulsed Field (mV/cm)", xlim=(xmin, xmax),
           ylabel=r"$W_0 + \Delta W_{MW}(\phi)$ (GHz)", ylim=(ymin, ymax),
           title="Downhill Electrons")
    ax.legend().remove()
    # add lines
    for f in field_picks:
        ax.axvline(f, color='k', lw=3, linestyle='dashed')
    ax.axhline(W0_pick, color='k', lw=3, linestyle='dashed')
    # save
    fig.tight_layout()
    fname = os.path.join("simple_conv", "up_and_down_orbits_lined.pdf")
    plt.savefig(fname)
    return data_tot, picked_tot


# main script
dW = 43
W0 = -10
f = 30
dphi = np.pi/180
# dfprob_up = tt_conv_plot(dW, W0, f, dphi)
# params = params_bulk(dW, W0, dphi)
# params = pd.read_csv("tconv_params_0_up.csv", index_col=0)
# bulk_plot(params)

# dfmarks = build_marks_down()
# dfprob_down = tt_conv_plot_down(dW, W0, f, dphi)
# dfprob = prob_and_conv_down(dfmarks, dW, W0, f, dphi)
# print(dfmarks)

# dfprob_up, dfprob_down = tt_conv_plot_both(dW, W0, f, dphi)
# turning_time_figure([0, 15, 30], W0)

dfmarks = build_marks()
obs = dfmarks.loc[f]
dfprob, popt = prob_and_conv_dil(dfmarks, dW, W0, f, dphi)
ax = dfprob.plot(x='phi', y='W')
ax.axhline(obs['tt10'])
ax = dfprob.plot(x='phi', y='p')
ax.axvline(np.pi/6)
ax.axvline(7*np.pi/6)
