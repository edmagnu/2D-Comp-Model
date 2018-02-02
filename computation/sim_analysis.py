# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:58:04 2018

@author: edmag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools


def atomic_units():
    """Return a dictionary of atomic units"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
    return au


def read_metadata(fname):
    """Read file metadata and return a dictionary"""
    meta = {}  # initialize metadata dict
    with open(fname) as file:
        for line in file:
            if line[0] == "#":  # metadata written in comment lines
                line = line[1:]
                para = line.split("\t")  # tabs betweeen key & value
                if len(para) == 2:  # format for actual metadata, not comment
                    meta[para[0].strip()] = para[1].strip()
    return meta


def read_tidy():
    """Read in every result data file with metadata. Add every file's data into
    a tidy DataFrame.
    Returns pd.DataFrame"""
    # specify file
    directory = ("C:\\Users\\edmag\\Documents\\Work\\" +
                 "2D-Comp-Model\\computation\\results")
    flist = os.listdir(directory)
    data_m = pd.DataFrame()  # initialize DataFrame
    for file in flist:
        fname = directory + "\\" + file  # build file
        # print(fname)
        # load metadata and data
        meta = read_metadata(fname)
        data = pd.read_csv(fname, sep="\t", comment="#", index_col=False)
        # add in metadata
        data["Filename"] = pd.Series([fname]*len(data["phi"]), dtype=str)
        data["E0"] = pd.Series([meta["E0"]]*len(data["phi"]), dtype=float)
        data["Ep"] = pd.Series([meta["Ep"]]*len(data["phi"]), dtype=float)
        # organize data
        data = data[["Filename", "E0", "Ep", "dL", "th_LRL", "phi", "enfinal"]]
        # enfinal_plot(data)
        data_m = data_m.append(data)  # append to master DataFrame
    data_m.to_csv("data_raw.txt")
    return data_m


def enfinal_plot(data):
    """Given simulation results, plot the final orbital energy against launch
    phase.
    Returns None."""
    au = atomic_units()
    fig, ax = plt.subplots(nrows=2, sharex=True)
    # mask only LRL and non-NaN
    th_LRL = data["th_LRL"].unique()
    labels = ["Uphill", "Downhill"]
    for i in [0, 1]:
        mask_u = data["th_LRL"] == th_LRL[i]
        mask_nan = np.isnan(data["enfinal"])
        mask = mask_u & np.logical_not(mask_nan)
        # plot final energies
        ax[i].plot(data[mask]["phi"], data[mask]["enfinal"]/au["GHz"], ".",
                   color="C0", label=labels[i])
        # plot NaN
        mask = mask_u & mask_nan
        ax[i].plot(data[mask]["phi"], data[mask]["phi"]*0, "X", color="C3",
                   label="missing")
        ax[i].set_ylabel(r"$ E_f = -\frac{1}{r} + \frac{1}{2}v^2$ (GHz)")
        ax[i].grid(True)
        ax[i].legend(loc=3)
    # label and tick axes
    ax[1].set_xlabel(r"MW Launch Phase $\phi_0$ (rad.)")
    ax[1].set_xticks(np.array([1/6, 4/6, 7/6, 10/6])*np.pi)
    ax[1].set_xticklabels([r"$\pi/6$", r"$4\pi/6$", r"$7\pi/6$", r"$10\pi/6$"])
    plt.tight_layout()
    return None


def bound_plot(data, key="bound"):
    """Given simulation results, plot whether the electron is bound against
    launch phase.
    Returns None."""
    fig, ax = plt.subplots(nrows=2, sharex=True)
    # mask only LRL and non-NaN
    th_LRL = data["th_LRL"].unique()
    labels = ["Uphill", "Downhill"]
    for i in [0, 1]:
        mask_u = (data["th_LRL"] == th_LRL[i])
        mask_nan = np.isnan(data[key])
        mask = mask_u & np.logical_not(mask_nan)
        # plot final energies
        ax[i].plot(data[mask]["phi"], data[mask][key], ".",
                   color="C0", label=labels[i])
        # plot NaN
        mask = mask_u & mask_nan
        ax[i].plot(data[mask]["phi"], data[mask]["phi"]*0, "X", color="C3",
                   label="missing")
        ax[i].set_ylabel("bound")
        ax[i].grid(True)
        ax[i].legend()
    # label and tick axes
    ax[1].set_xlabel(r"MW Launch Phase $\phi_0$ (rad.)")
    ax[1].set_xticks(np.array([1/6, 4/6, 7/6, 10/6])*np.pi)
    ax[1].set_xticklabels([r"$\pi/6$", r"$4\pi/6$", r"$7\pi/6$", r"$10\pi/6$"])
    plt.tight_layout()
    return None


def single_patch(data, i):
    """Given a known enfinal=NaN at location i, determine bound_p from average
    of nearest neighbors. data.iloc[i]["bound_p"] is fixed.
    Returns DataFrame"""
    obs = data.iloc[i]  # get the NaN observation
    # mask out each quality (E0, Ep, dL, th_LRL)
    mask = ((data["E0"] == obs["E0"]) & (data["Ep"] == obs["Ep"]) &
            (data["dL"] == obs["dL"]) & (data["th_LRL"] == obs["th_LRL"]))
    run = data[mask][["phi", "enfinal", "bound"]]
    run.sort_values(by="phi", inplace=True)
    # find NaN nearest neighbors in run.
    inan = run.index.get_loc(i)  # get physical location in run
    # nearest neighbor from below
    imin = np.NaN
    di = 1
    while np.isnan(imin):  # step down until "bound" is not NaN
        itemp = (inan-di) % len(run)  # wrap if itemp out of range
        if np.logical_not(np.isnan(run.iloc[itemp]["bound"])):
            imin = inan-di
        di = di + 1
    # nearest neighbor from above
    imax = np.NaN
    di = 1
    while np.isnan(imax):  # step up until "bound" is not NaN
        itemp = (inan+di) % len(run)  # wrap if itemp is out of range
        if np.logical_not(np.isnan(run.iloc[itemp]["bound"])):
            imax = inan+di
        di = di+1
    # print(imin, imax)
    # take the distance to the furtherst "nearest neighbor"
    di = max([imax-inan, inan-imin])
    # replace bad "bound" with patched "bound_p"
    # array mod len() handles indexes out of range
    iarray = np.arange(inan-di, inan+di+1) % len(run)
    data.loc[i, "bound_p"] = np.mean(run["bound"].iloc[iarray])
    return data


def bound_patch():
    """Reads raw data from read_tidy() output, and uses enfinal to add "bound"
    to the DataFrame. To replace NaN, "bound_p" averages nearest neighbors in
    "bound". Writes new data to "data_bound.txt"
    Returns data DataFrame"""
    data = pd.read_csv("data_raw.txt", index_col=0)
    data.reset_index(drop=True, inplace=True)
    mask_nan = np.isnan(data["enfinal"])
    # print(data[mask_nan].index.unique())
    data["bound"] = (data["enfinal"] < 0)*1.0
    data["bound_p"] = data["bound"]
    data.loc[mask_nan, "bound"] = np.NaN
    # replace NaN with average of surrounding.
    for i in data[mask_nan].index:
        print(i, "/", len(mask_nan), end="\r")  # progress
        data = single_patch(data, i)  # run for each value
    data.to_csv("data_bound.txt")
    return data


def bound_test_data(phi0, dphi):
    """Generate mock data to test convolution on.
    Returns DataFrame["th_LRL", "phi", "bound", "bound_p"]"""
    au = atomic_units()
    data = pd.DataFrame()  # initialize
    dft = pd.DataFrame()
    # Start with E0=0, Ep=0, dL=-1, th_LRL=0
    phi_series = pd.Series(np.arange(0, 2*np.pi, np.pi/100), dtype=float)
    ln = len(phi_series)
    dft["Filename"] = pd.Series(["fake\\data_test.txt"]*ln, dtype=str)
    dft["E0"] = pd.Series([0]*ln, dtype=float)
    dft["Ep"] = pd.Series([0]*ln, dtype=float)
    dft["dL"] = pd.Series([-1]*ln, dtype=float)
    dft["th_LRL"] = pd.Series([0]*ln, dtype=float)
    dft["phi"] = phi_series
    dft["enfinal"] = pd.Series([-1*au["GHz"]]*ln, dtype=float)
    dft["bound"] = pd.Series([0]*ln, dtype=float)
    dft["bound_p"] = pd.Series([0]*ln, dtype=float)
    # E0=0, Ep=0, dL=-1, th_LRL=0
    data = data.append(dft)
    # E0=0, Ep=0, dL=-1, th_LRL=pi
    dft.replace({"th_LRL": {0: np.pi}}, inplace=True)
    data = data.append(dft)
    # E0=0, Ep=0, dL=1, th_LRL = 0
    dft.replace({"th_LRL": {np.pi: 0}}, inplace=True)
    dft.replace({"dL": {-1: 1}}, inplace=True)
    data = data.append(dft)
    # E0=0, Ep=0, dL=1, th_LRL = np.pi
    dft.replace({"th_LRL": {0: np.pi}}, inplace=True)
    data = data.append(dft)
    # add bump to th_LRL = 0 at phi = phi0
    # build mask
    mask_th = (data["th_LRL"] == 0)
    mask_phi = ((data["phi"] > phi0-dphi) & (data["phi"] < phi0+dphi))
    if phi0-dphi < 0:
        mask_phi = np.logical_or(mask_phi,
                                 (data["phi"] > ((phi0-dphi) % (2*np.pi))))
    if phi0+dphi > 2*np.pi:
        mask_phi = np.logical_or(mask_phi,
                                 (data["phi"] < ((phi0-dphi) % (2*np.pi))))
    mask = mask_th & mask_phi
    # adjust data
    data.loc[mask, "enfinal"] = 1.0*au["GHz"]
    data.loc[mask, "bound"] = 1.0
    data.loc[mask, "bound_p"] = 1.0
    # add bump to th_LRL=pi at phi = (phi0+pi) % 2pi
    phi0 = ((phi0 + np.pi) % (2*np.pi))
    # build mask
    mask_th = (data["th_LRL"] == np.pi)
    mask_phi = ((data["phi"] > phi0-dphi) & (data["phi"] < phi0+dphi))
    if phi0-dphi < 0:
        mask_phi = np.logical_or(mask_phi,
                                 (data["phi"] > ((phi0-dphi) % (2*np.pi))))
    if phi0+dphi > 2*np.pi:
        mask_phi = np.logical_or(mask_phi,
                                 (data["phi"] < ((phi0-dphi) % (2*np.pi))))
    mask = mask_th & mask_phi
    # adjust data
    data.loc[mask, "enfinal"] = 1.0*au["GHz"]
    data.loc[mask, "bound"] = 1.0
    data.loc[mask, "bound_p"] = 1.0
    data.reset_index(drop=True, inplace=True)
    return data


def inventory():
    data = pd.read_csv("data_bound.txt", index_col=0)
    au = atomic_units()
    # build combinations list for E0, Ep, dL, th_LRL
    keys = ["E0", "Ep", "dL", "th_LRL"]
    header = (keys[0] + "\t" + keys[1] + "\t" + keys[2] + "\t" + keys[3]
              + "\t" + "NaN/Tot")
    vals = {}
    reports = [[header]]
    for key in keys:
        vals[key] = np.sort(data[key].unique())
    # build mask for each combination
    combos = itertools.product(
            vals["E0"], vals["Ep"], vals["dL"], vals["th_LRL"])
    for combo in combos:
        val = {}
        mask = [True]*len(data)  # start with every point
        for i in [0, 1, 2, 3]:
            mask = mask & (data[keys[i]] == combo[i])  # add conditions to mask
            val[keys[i]] = combo[i]
        mask_nan = np.isnan(data["enfinal"])  # also compare to NaN
        # print conditions and NaN/total observations.
        report = "{0:> 6.2f} \t {1:> 8.2f} \t {2: .0f} \t {3:.0f}"
        report = report + " \t {4:>3.0f}/{5:>3.0f}"
        report = report.format(
                val["E0"]/au["GHz"], val["Ep"]/au["mVcm"], val["dL"],
                val["th_LRL"]/np.pi, sum(mask & mask_nan), sum(mask))
        # print(report)
        reports = np.append(reports, [report])
    reports = pd.DataFrame(reports)
    reports.to_csv("inventory.txt", header=False, index=False)
    return reports


def conv_model(x, x0):
    """Model of AM laser envelope to convolve over data["bound"].
    Returns np.array of 0.5 + np.cos(x + x0)"""
    return 0.5*(1 + np.cos(x - x0))


def data_triple(data):
    """Given a DataFrame with "phi" from [0, 2pi], make copies of all
    observations at [-2pi, 0] and [2pi, 4pi] as well.
    Returns DataFrame"""
    temp = data.copy(deep=True)  # df1 = df2 makes pointers equal
    # below
    temp["phi"] = temp["phi"] - 2*np.pi
    temp.set_index(temp.index - len(temp), inplace=True)
    data = data.append(temp)
    # above
    temp["phi"] = temp["phi"] + 4*np.pi
    temp.set_index(temp.index + 2*len(temp), inplace=True)
    data = data.append(temp)
    data.sort_values(by="phi", inplace=True)
    return data


def convolution(data, mask):
    """Run a convolution on data[mask] based on conv_model().
    Set the "conv" key in data to the result.
    Returns DataFrame"""
    # amlaser.plot(x="phi", y="I", kind="scatter")
    amlaser = pd.DataFrame()
    amlaser["phi"] = data[mask]["phi"]
    amlaser["I"] = conv_model(amlaser["phi"], np.pi)/(200*0.5)
    # triple data
    data3 = data_triple(data[mask])
    # mask particular run with E0, Ep, dL, th_LRL
    # convolve
    conv3 = pd.DataFrame()
    conv3["phi"] = data3["phi"]
    # conv3["conv"] =
    conv3["conv"] = np.convolve(data3["bound"], amlaser["I"], mode="same")
    conv = conv3.loc[0:199]
    data.loc[mask, "conv"] = conv3.loc[0:199, "conv"]
    return data, conv, mask


def test_convolve():
    phi0 = np.pi/6
    dphi = np.pi/12
    data = bound_test_data(phi0=phi0, dphi=dphi)
    # build dict of parameters
    vals = {}
    keys = ["E0", "Ep", "dL", "th_LRL"]
    for key in keys:
        vals[key] = np.sort(data[key].unique())
    # build convolution array
    data["conv"] = pd.Series([np.NaN]*len(data), dtype=float)
    # mask particular run
    combos = itertools.product(
            vals["E0"], vals["Ep"], vals["dL"], vals["th_LRL"])
    for combo in combos:
        val = {}  # store current values
        mask = [True]*len(data)  # start with every point
        for i in [0, 1, 2, 3]:
            mask = mask & (data[keys[i]] == combo[i])  # add conditions to mask
            val[keys[i]] = combo[i]
        print(val)
        print(data[mask].index)
        data, conv, mask = convolution(data, mask)
        # conv.plot(x="phi", y="conv")
    # ax = conv.plot(x="phi", y="conv", kind="scatter", color="C0")
    fig, ax = plt.subplots()
    ax.plot(data[mask]["phi"], data[mask]["bound_p"], ".", color="C0",
            label="bound")
    # data[mask].plot(x="phi", y="bound_p", kind="scatter", color="C0", ax=ax)
    ax.plot(data[mask]["phi"], data[mask]["conv"], ".", color="C1",
            label="conv")
    # data[mask].plot(x="phi", y="conv", kind="scatter", color="C1", ax=ax)
    ax.axvline(2*np.pi, color="black")
    ax.axvline(0, color="black")
    ax.axvline(phi0, color="black", linestyle="dashed")
    ax.legend()
    return data, conv, mask


data, conv, mask = test_convolve()