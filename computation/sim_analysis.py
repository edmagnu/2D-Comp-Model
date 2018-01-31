# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:58:04 2018

@author: edmag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


def bound_plot(data, key):
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
        ax[i].legend(loc=3)
    # label and tick axes
    ax[1].set_xlabel(r"MW Launch Phase $\phi_0$ (rad.)")
    ax[1].set_xticks(np.array([1/6, 4/6, 7/6, 10/6])*np.pi)
    ax[1].set_xticklabels([r"$\pi/6$", r"$4\pi/6$", r"$7\pi/6$", r"$10\pi/6$"])
    plt.tight_layout()
    return None


def bound_patch():
    data = pd.read_csv("data_raw.txt", index_col=0)
    data.reset_index(drop=True, inplace=True)
    mask_nan = np.isnan(data["enfinal"])
    # print(data[mask_nan].index.unique())
    data["bound"] = (data["enfinal"] < 0)*1.0
    data["bound_p"] = data["bound"]
    data.loc[mask_nan, "bound"] = np.NaN
    # replace NaN with average of surrounding.
    for i in data[mask_nan].index:
        print(i, "/", len(mask_nan))  # progress
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
        data.loc[i, "bound_p"] = np.mean(run["bound"].iloc[inan-di:inan+di+1])
    data.to_csv("data_bound.txt")
    return data


# data = bound_patch()
# bound_plot(data, "bound")
# bound_plot(data, "bound_p")
# data = read_tidy()
# data = bound_patch()
data = pd.read_csv("data_bound.txt", index_col=0)
