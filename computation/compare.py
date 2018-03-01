# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 00:29:44 2018

@author: edmag
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def atomic_units():
    """Return a dictionary of atomic units"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
    return au


def phase_amp_plot():
    au = atomic_units()
    params = pd.read_csv("params_sums.txt", index_col=0)
    masknan = np.isnan(params["dL"]) & np.isnan(params["th_LRL"])
    E0s = params[masknan]["E0"].unique()
    plt.close()
    fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col', sharey='row')
    # E0 = 0 GHz
    E0 = E0s[1]
    mask = masknan & (params["E0"] == E0)
    data = params[mask].copy(deep=True)
    data["a"] = data["a"]*2
    data["Ep"] = data["Ep"]/au["mVcm"]
    data.plot(x="Ep", y="x0", ax=ax[0, 1], kind="scatter")
    ax[0, 1].tick_params(which="minor", left="off")
    data.plot(x="Ep", y="a", ax=ax[1, 1], kind="scatter")
    data.plot(x="Ep", y="y0", ax=ax[2, 1], kind="scatter")
    ax[2, 1].set(xlabel="Pulsed Field (mV/cm)")
    ax[0, 1].set(title=(r"$W_0$ = {} GHz".format(np.round(E0/au["GHz"], 2))))
    # E0 = -20 GHz
    E0 = E0s[0]
    mask = masknan & (params["E0"] == E0)
    data = params[mask].copy(deep=True)
    data["Ep"] = data["Ep"]/au["mVcm"]
    data.plot(x="Ep", y="x0", ax=ax[0, 0], kind="scatter")
    ax[0, 0].set(yticks=[np.pi/6, 7*np.pi/6],
                 yticklabels=[r"$\pi/6$", "$7\pi/6$"])
    ax[0, 0].tick_params(which="minor", left="off")
    ax[0, 0].set(ylabel=r"Phase $\phi$ (rad)")
    data.plot(x="Ep", y="a", ax=ax[1, 0], kind="scatter")
    ax[1, 0].set(ylabel="Amp (pk-pk)")
    data.plot(x="Ep", y="y0", ax=ax[2, 0], kind="scatter")
    ax[2, 0].set(xlabel="Pulsed Field (mV/cm)", ylabel="Mean")
    ax[0, 0].set(title=(r"$W_0$ = {} GHz".format(np.round(E0/au["GHz"], 2))))
    # turn on grids
    for i, j in itertools.product([0, 1], [0, 1, 2]):
        ax[j, i].grid(True)
    return()


def dil_p2_expanded():
    """Selects "fits.txt" data with lasers at DIL +2GHz and Attn = 38.0
    (happens to all be 2016-09-22) and plots Static vs. fit parameters "a, phi"
    Uses massage_amp_phi() before plotting to fix "a, phi".
    Returns DataFrame "fsort" that is just the plotted observations."""
    # read in all fits
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "fits.txt")
    fits = pd.read_csv(fname, sep="\t", index_col=0)
    # mask out DIL + 2 GHz and Attn = 44.0
    mask = (fits["DL-Pro"] == 365872.6) & (fits["Attn"] == 44)
    fsort = fits[mask].sort_values(by=["Static"]).copy(deep=True)
    # unmassage amps and phases
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    fsort["phi"] = fsort["phi"] % (2*np.pi)
    # amplitude -> pk-pk
    fsort["a"] = 2*fsort["a"]
    # mV/cm
    fsort["Static"] = fsort["Static"]*0.72*0.1
    # manually exclude bad data runs
    excluded = ["2016-09-23\\3_delay.txt", "2016-09-23\\4_delay.txt"]
    for fname in excluded:
        fsort = fsort[fsort["Filename"] != fname]
    # plot
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    # data
    fsort.plot(x="Static", y="phi", kind="scatter", ax=axes[0])
    fsort.plot(x="Static", y="a", style="-o", ax=axes[1])
    fsort.plot(x="Static", y="y0", style="-o", ax=axes[2])
    axes[0].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes[0].set_yticklabels(["0", r"$\pi/2$", r"$\pi$",
                             r"$3\pi/2$", r"$2\pi$"])
    # make it pretty
    axes[0].set(ylabel="Phase (rad)", title="DIL + 2 GHz")
    axes[1].set(ylabel="Amp (pk-pk)")
    axes[2].set(xlabel="Pulsed Field (mV/cm)", ylabel="Mean")
    for i in [0, 1, 2]:
        axes[i].grid(True)
    for i in [1, 2]:
        axes[i].legend()
        axes[i].legend().remove()
    return fsort


def dil_m14_expanded():
    """Selects "fits.txt" data with lasers at DIL -14 GHz (from 2016-09-23to27)
    and plots Static vs. fit parameters "a, phi".
    Uses massage_amp_phi() before plotting to fix "a, phi".
    Manually excludes some bad data runs.
    Returns DataFrame "fsort" that is just the plotted observations."""
    # read in all fits
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "fits.txt")
    fits = pd.read_csv(fname, sep="\t", index_col=0)
    # mask out just DIL - 14 GHz
    mask = (fits["DL-Pro"] == 365856.7)
    fsort = fits[mask].sort_values(by=["Static"])
    # unmassage amps and phases
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    fsort["phi"] = fsort["phi"] % (2*np.pi)
    # amplitude -> pk-pk
    fsort["a"] = 2*fsort["a"]
    # mV/cm
    fsort["Static"] = fsort["Static"]*0.72*0.1
    # manually exclude bad data runs
    excluded = ["2016-09-23\\5_delay.txt", "2016-09-23\\11_delay.txt",
                "2016-09-23\\12_delay.txt", "2016-09-23\\16_delay.txt",
                "2016-09-23\\17_delay.txt", "2016-09-26\\8_delay.txt",
                "2016-09-26\\9_delay.txt"]
    for fname in excluded:
        fsort = fsort[fsort["Filename"] != fname]
    # plot
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    fsort.plot(x="Static", y="phi", kind="scatter", ax=axes[0])
    fsort.plot(x="Static", y="a", style="-o", ax=axes[1])
    fsort.plot(x="Static", y="y0", style="-o", ax=axes[2])
    axes[0].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    axes[0].set_yticklabels(["0", r"$\pi/2$", r"$\pi$",
                             r"$3\pi/2$", r"$2\pi$"])
    # make it pretty
    axes[0].set(ylabel=r"Phase $\phi$ (rad)", title="DIL - 14 GHz")
    axes[1].set(ylabel="Amp (pk-pk)")
    axes[2].set(xlabel="Pulsed Field (mV/cm)", ylabel="Mean")
    for i in [0, 1, 2]:
        axes[i].grid(True)
    for i in [1, 2]:
        axes[i].legend()
        axes[i].legend().remove()
    return fsort


def mod_p0_exp():
    au = atomic_units()
    params = pd.read_csv("params_sums.txt", index_col=0)
    masknan = np.isnan(params["dL"]) & np.isnan(params["th_LRL"])
    E0s = params[masknan]["E0"].unique()
    # E0 = 0 GHz
    E0 = E0s[1]
    mask = masknan & (params["E0"] == E0)
    data = params[mask].copy(deep=True)
    # interpret data
    data["a"] = data["a"]*2
    data["Ep"] = data["Ep"]/au["mVcm"]
    data = mirror_model(data)
    title = (r"Model $W_0$ = {} GHz".format(np.round(E0/au["GHz"], 2)))
    phase_amp_mean_plot(data, title)
    plt.savefig(os.path.join("compare", "mod_p0.pdf"))
    return data


def mod_m20_exp():
    au = atomic_units()
    params = pd.read_csv("params_sums.txt", index_col=0)
    masknan = np.isnan(params["dL"]) & np.isnan(params["th_LRL"])
    E0s = params[masknan]["E0"].unique()
    # E0 = -20 GHz
    E0 = E0s[0]
    mask = masknan & (params["E0"] == E0)
    data = params[mask].copy(deep=True)
    # interpret data
    data["a"] = data["a"]*2
    data["Ep"] = data["Ep"]/au["mVcm"]
    data = mirror_model(data)
    title = (r"Model: $W_0$ = {} GHz".format(np.round(E0/au["GHz"], 2)))
    phase_amp_mean_plot(data, title)
    plt.savefig(os.path.join("compare", "mod_m20.pdf"))
    return data


def exp_m14_exp(ph_thresh=True):
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "fits.txt")
    fits = pd.read_csv(fname, sep="\t", index_col=0)
    # mask out just DIL - 14 GHz
    mask = (fits["DL-Pro"] == 365856.7)
    fsort = fits[mask].sort_values(by=["Static"])
    # unmassage amps and phases
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    fsort["phi"] = fsort["phi"] % (2*np.pi)
    # amplitude -> pk-pk
    fsort["a"] = 2*fsort["a"]
    # mV/cm
    fsort["Static"] = fsort["Static"]*0.72*0.1
    # manually exclude bad data runs
    excluded = ["2016-09-23\\5_delay.txt", "2016-09-23\\11_delay.txt",
                "2016-09-23\\12_delay.txt", "2016-09-23\\16_delay.txt",
                "2016-09-23\\17_delay.txt", "2016-09-26\\8_delay.txt",
                "2016-09-26\\9_delay.txt"]
    for fname in excluded:
        fsort = fsort[fsort["Filename"] != fname]
    # translate
    data = pd.DataFrame()
    data["Ep"] = fsort["Static"]
    data["a"] = fsort["a"]
    data["x0"] = fsort["phi"]
    data["y0"] = fsort["y0"]
    # phase threshold
    if ph_thresh is True:
        ph_th = 5*np.pi/6
        # Amplitude
        mask = (data["x0"] >= (ph_th - np.pi)) & (data["x0"] < ph_th)
        data.loc[mask, "a"] = -data[mask]["a"]
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "a"] = -data[mask]["a"]
        # phase
        mask = (data["x0"] < (ph_th - np.pi))
        data.loc[mask, "x0"] = data["x0"] + 2*np.pi
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "x0"] = data["x0"] - 2*np.pi
    else:
        ph_th = None
    # plot
    title = (r"Experiment: $W_0$ = DIL - 14 GHz")
    phase_amp_mean_plot(data, title, ph_th)
    plt.savefig(os.path.join("compare", "exp_m14.pdf"))
    return data


def exp_p2_exp(ph_thresh=True):
    # read in all fits
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "fits.txt")
    fits = pd.read_csv(fname, sep="\t", index_col=0)
    # mask out DIL + 2 GHz and Attn = 44.0
    mask = (fits["DL-Pro"] == 365872.6) & (fits["Attn"] == 44)
    fsort = fits[mask].sort_values(by=["Static"]).copy(deep=True)
    # unmassage amps and phases
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    fsort["phi"] = fsort["phi"] % (2*np.pi)
    # amplitude -> pk-pk
    fsort["a"] = 2*fsort["a"]
    # mV/cm
    fsort["Static"] = fsort["Static"]*0.72*0.1
    # manually exclude bad data runs
    excluded = ["2016-09-23\\3_delay.txt", "2016-09-23\\4_delay.txt"]
    for fname in excluded:
        fsort = fsort[fsort["Filename"] != fname]
    # translate
    data = pd.DataFrame()
    data["Ep"] = fsort["Static"]
    data["a"] = fsort["a"]
    data["x0"] = fsort["phi"]
    data["y0"] = fsort["y0"]
    # phase threshold
    if ph_thresh is True:
        ph_th = 6*np.pi/6
        # Amplitude
        mask = (data["x0"] >= (ph_th - np.pi)) & (data["x0"] < ph_th)
        data.loc[mask, "a"] = -data[mask]["a"]
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "a"] = -data[mask]["a"]
        # phase
        mask = (data["x0"] < (ph_th - np.pi))
        data.loc[mask, "x0"] = data["x0"] + 2*np.pi
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "x0"] = data["x0"] - 2*np.pi
    else:
        ph_th = None
    # plot
    title = (r"Experiment: $W_0$ = DIL + 2 GHz")
    phase_amp_mean_plot(data, title, ph_th)
    plt.savefig(os.path.join("compare", "exp_p2.pdf"))
    return data


def mirror_model(data):
    """Mirror model data to get +/- Ep"""
    mirrored = data.copy(deep=True)
    mirrored["Ep"] = -mirrored["Ep"]
    mirrored["x0"] = data["x0"] + np.pi
    data = data.append(mirrored, ignore_index=True)
    data["x0"] = data["x0"] % (2*np.pi)
    data.sort_values(by="Ep", inplace=True)
    return data


def phase_amp_mean_plot(data, title, ph_th=None):
    """Standard plotting for computed or experimental data.
    data DataFrame must have "Ep", "x0", "a", and "y0" keys."""
    fig, ax = plt.subplots(nrows=3, sharex=True)
    # line
    if ph_th is not None:
        ax[0].axhline(ph_th, color="k", lw=1)
    # plot data
    data.plot(x="Ep", y="x0", ax=ax[0], style="-o")
    data.plot(x="Ep", y="a", ax=ax[1], style="-o")
    data.plot(x="Ep", y="y0", ax=ax[2], style="-o")
    # beautify
    ax[0].tick_params(which="minor", left="off")
    ax[0].set(  # xlim=(-200, 200),
              yticks=[np.pi/6, 7*np.pi/6],
              yticklabels=[r"$\pi/6$", "$7\pi/6$"],
              ylabel=r"Phase $\phi_0$ (rad)",
              ylim=(-np.pi*0.2, 2.2*np.pi))
    ax[0].set(title=title)
    ax[1].set(ylabel="Amp (pk-pk)")
    ax[2].set(xlabel="Pulsed Field (mV/cm)", ylabel="Mean")
    # turn on grids
    for i in [0, 1, 2]:
        ax[i].grid(True)
        ax[i].legend()
        ax[i].legend().remove()
    plt.tight_layout()
    return fig, ax

def fsort_prep(fsort, excluded, title, ph_th=None, figname=None):
    fsort.sort_values(by=["Static"], inplace=True)
    # unmassage amps and phases
    mask = (fsort["a"] < 0)
    fsort.loc[mask, "a"] = -fsort[mask]["a"]
    fsort.loc[mask, "phi"] = fsort[mask]["phi"] + np.pi
    fsort["phi"] = fsort["phi"] % (2*np.pi)
    # amplitude -> pk-pk
    fsort["a"] = 2*fsort["a"]
    # mV/cm
    fsort["Static"] = fsort["Static"]*0.72*0.1
    # manually exclude bad data runs
    for fname in excluded:
        fsort = fsort[fsort["Filename"] != fname]
    # translate
    data = pd.DataFrame()
    data["Ep"] = fsort["Static"]
    data["a"] = fsort["a"]
    data["x0"] = fsort["phi"]
    data["y0"] = fsort["y0"]
    # phase threshold
    if ph_th is not None:
        # ph_th = 6*np.pi/6
        # Amplitude
        mask = (data["x0"] >= (ph_th - np.pi)) & (data["x0"] < ph_th)
        data.loc[mask, "a"] = -data[mask]["a"]
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "a"] = -data[mask]["a"]
        # phase
        mask = (data["x0"] < (ph_th - np.pi))
        data.loc[mask, "x0"] = data["x0"] + 2*np.pi
        mask = (data["x0"] >= (ph_th + np.pi))
        data.loc[mask, "x0"] = data["x0"] - 2*np.pi
    # plot
    fig, ax = phase_amp_mean_plot(data, title, ph_th)
    # save
    if figname is not None:
        plt.savefig(os.path.join("compare", figname))
    return data, fig, ax


def comp_plots():
    # read in all fits
    fname = os.path.join("..", "..", "Data", "StaPD-Analysis", "fits.txt")
    fits = pd.read_csv(fname, sep="\t", index_col=0)
    # DIL + 18 GHz
    mask = (fits["DL-Pro"] == 365888.5) & (fits["Attn"] == 44.0)
    fsort = fits[mask].copy(deep=True)
    excluded = ["2016-10-01\\3_delay.txt", "2016-10-01\\4_delay.txt",
                "2016-10-01\\9_delay.txt", "2016-10-01\\22_delay.txt"]
    title = r"Experiment: $W_0$ = DIL + 18 GHz"
    ph_th = 2/6*np.pi
    figname = "exp_p18.pdf"
    data, fig, ax = fsort_prep(fsort=fsort, excluded=excluded, title=title,\
                               ph_th=ph_th, figname=figname)
    # DIL + 2 GHz
    mask = (fits["DL-Pro"] == 365872.6) & (fits["Attn"] == 44)
    fsort = fits[mask].copy(deep=True)
    excluded = ["2016-09-23\\3_delay.txt", "2016-09-23\\4_delay.txt"]
    title = r"Experiment: $W_0$ = DIL + 2 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_p2.pdf"
    data, fig, ax = fsort_prep(fsort=fsort, excluded=excluded, title=title,\
                               ph_th=ph_th, figname=figname)
    # DIL - 14 GHz
    mask = (fits["DL-Pro"] == 365856.7)
    fsort = fits[mask].copy(deep=True)
    excluded = ["2016-09-23\\5_delay.txt", "2016-09-23\\11_delay.txt",
                "2016-09-23\\12_delay.txt", "2016-09-23\\16_delay.txt",
                "2016-09-23\\17_delay.txt", "2016-09-26\\8_delay.txt",
                "2016-09-26\\9_delay.txt"]
    title = r"Experiment: $W_0$ = DIL - 14 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_m14.pdf"
    data, fig, ax = fsort_prep(fsort=fsort, excluded=excluded, title=title,
                               ph_th=ph_th, figname=figname)
    # DIL - 30 GHz
    mask = (fits["DL-Pro"] == 365840.7)
    fsort = fits[mask].sort_values(by=["Static"])
    excluded = ["2016-09-27\\7_delay.txt", "2016-09-27\\15_delay.txt"]
    title = r"Experiment: $W_0$ = DIL - 30 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_m30.pdf"
    data, fig, ax = fsort_prep(fsort=fsort, excluded=excluded, title=title,\
                               ph_th=ph_th, figname=figname)
    # DIL - 46 GHz
    mask = (fits["DL-Pro"] == 365824.8) & (fits["Attn"] == 44.0)
    fsort = fits[mask].copy(deep=True)
    excluded = ["2016-09-28\\2_delay.txt", "2016-09-28\\3_delay.txt",
                "2016-09-28\\4_delay.txt", "2016-09-28\\5_delay.txt",
                "2016-09-28\\6_delay.txt", "2016-09-28\\7_delay.txt",
                "2016-09-28\\8_delay.txt", "2016-09-28\\9_delay.txt",
                "2016-09-28\\10_delay.txt", "2016-09-28\\11_delay.txt",
                "2016-09-28\\27_delay.txt", "2016-10-01\\2_delay.txt"]
    title = r"Experiment: $W_0$ = DIL - 46 GHz"
    ph_th = 5.5/6*np.pi
    figname = "exp_m46.pdf"
    data, fig, ax = fsort_prep(fsort=fsort, excluded=excluded, title=title,\
                               ph_th=ph_th, figname=figname)
    return

comp_plots()
