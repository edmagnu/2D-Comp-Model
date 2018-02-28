# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:58:04 2018

@author: edmag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import itertools
import scipy.optimize
from PyPDF2 import PdfFileMerger


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


def read_tidy():
    """Read in every result data file with metadata. Add every file's data into
    a tidy DataFrame and writes to "data_raw.txt"
    Returns pd.DataFrame"""
    # specify file
    directory = ("results")
    flist = os.listdir(directory)
    data_m = pd.DataFrame()  # initialize DataFrame
    fnumber = len(flist)  # how many times to we need to run
    funcname = "read_tidy()"
    for i, file in enumerate(flist):
        # progress
        progress(funcname, i, fnumber)
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
    fnumber = sum(mask_nan)  # how many times do we need to run?
    funcname = "bound_patch()"
    for i, j in enumerate(data[mask_nan].index):
        # progress
        progress(funcname, i, fnumber)
        data = single_patch(data, j)  # run for each value
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


def xticks_2p():
    """Return ticks and ticklabels starting at pi/6 separated by pi/2"""
    ticklabels = [r"$\pi/6$", r"$4\pi/6$", r"$7\pi/6$", r"$10\pi/6$"]
    ticks = [np.pi/6, 4*np.pi/6, 7*np.pi/6, 10*np.pi/6]
    return ticks, ticklabels


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
    amlaser["I"] = conv_model(amlaser["phi"], np.pi)/(200*0.5)
    return amlaser


def combinations(data, keys, drop_nan=True):
    """Given the DataFrame data, and keys in data, produce a list of all the
    combinations of unique values data has for those keys. If drop_nan is True,
    all np.NaN values will be removed from the unique values and
    the combinations.
    Returns combos: list of unique combinations,
            vals: dict of {key: unique values}"""
    vals = {}
    vallist = []
    for key in keys:
        val = np.sort(data[key].unique())
        if drop_nan is True:
            val = val[~np.isnan(val)]
        vals[key] = val
        vallist.append(vals[key])
    combos = list(itertools.product(*vallist))
    return combos, vals


def convolution(data, E0, Ep, dL, th_LRL):
    """Given DataFrame data, select only obs with the specified E0, Ep, dL,
    th_LRL, and convolve the "phi" vs "bound_p" data with the array from
    laser_envelope(). Store the convolution in data["conv"]
    Returns data: DataFrame, mask: Boolean mask used,
            amlaser: Convolution envelope"""
    # unpack combination mask
    mask = pd.Series([True]*len(data))
    mask = mask & (data["E0"] == E0)
    mask = mask & (data["Ep"] == Ep)
    mask = mask & (data["dL"] == dL)
    mask = mask & (data["th_LRL"] == th_LRL)
    # convolve
    amlaser = laser_envelope(data[mask])
    conv = np.convolve(data[mask]["bound_p"], amlaser["I"], mode="same")
    # insert convolution into data
    data.loc[mask, "conv"] = conv[range(sum(mask), 2*sum(mask))]
    return data, mask, amlaser


def plot_conv(data, ax, E0, Ep, dL, th_LRL):
    """Given DataFrame data, select the obs with the specified E0, Ep, dL,
    th_LRL, and plot the convolution on the provided axes ax.
    Returns mask: Boolean mask used, ax: matplotlib axes object."""
    au = atomic_units()
    # unpack combination mask
    mask = pd.Series([True]*len(data))
    mask = mask & (data["E0"] == E0)
    mask = mask & (data["Ep"] == Ep)
    mask = mask & (data["dL"] == dL)
    mask = mask & (data["th_LRL"] == th_LRL)
    # convolve
    amlaser = laser_envelope(data[mask])
    # plot bound_p
    data[mask].plot.scatter(x="phi", y="bound_p", ax=ax, c="C0",
                            label="Bound")
    ax.plot(amlaser.loc[0:199, "phi"], amlaser.loc[0:199, "I"]*100,
            c="C2", linewidth=3, label=r"Laser I")
    data[mask].plot(x="phi", y="conv", c="C3", lw=3, label="Conv.",
                    ax=ax)
    # make it pretty
    xticks, xticklabels = xticks_2p()
    titlestring = ("E0 = " + str(E0/au["GHz"]) + "\tEp = " +
                   str(Ep/au["mVcm"]) + "\tdL = " + str(dL) +
                   "\tth_LRL = " + str(th_LRL/np.pi) + r"$\pi$")
    ax.set(xticks=xticks, xticklabels=xticklabels,
           xlabel=r"Phase $\phi$", ylabel="", title=titlestring)
    ax.legend()
    return mask, ax


def test_convolve(phi0=np.pi/6, dphi=np.pi/12, plot=True):
    """Produces DataFrame from bound_test_data(). For each E0, Ep, dL, and
    th_LRL combination from combinations(), uses convolution() to convolve with
    the laser_envelope(). Plots each combination using plot_conv().
    Returns DataFrame data.
    """
    # build test data
    # phi0 = 1*np.pi/6
    # dphi = np.pi/12
    data = bound_test_data(phi0=phi0, dphi=dphi)
    data["conv"] = pd.Series([np.NaN]*len(data))
    # build dict of parameters
    keys = ["E0", "Ep", "dL", "th_LRL"]
    combos, vals = combinations(data, keys)
    for combo in combos:
        data, mask, amlaser = convolution(data, *combo)
    # plots
    if plot is True:
        fig, ax = plt.subplots(nrows=len(combos), figsize=(6, 3*len(combos)))
        for i, combo in enumerate(combos):
            plot_conv(data, ax[i], *combo)
            # plot marker lines
            ax[i].axvline(phi0 % (2*np.pi), linestyle="solid", c="silver")
            ax[i].axvline((phi0+np.pi) % (2*np.pi), linestyle="dashed",
                          c="silver")
        plt.tight_layout()
    return data


def build_convolve():
    """Reads DataFrame from data_bound.txt. Looks at every combination of E0,
    Ep, dL, and th_LRL with combinations(). Masks out each combination and
    convolves with convolution(), producing a new data["conv"] key. Writes to
    data_conv.txt
    Returns data DataFrame.
    """
    data = pd.read_csv("data_bound.txt", index_col=0)
    data["conv"] = pd.Series([np.NaN]*len(data))
    # build dict of parameters
    keys = ["E0", "Ep", "dL", "th_LRL"]
    combos, vals = combinations(data, keys)
    funcname = "build_convolve()"
    total = len(combos)
    for i, combo in enumerate(combos):
        # progress
        progress(funcname, i, total)
        data, mask, amlaser = convolution(data, *combo)
        if (sum(mask) != 200):
            print(sum(mask))
    print()
    data.to_csv("data_conv.txt")
    return data


def combo_mask(data, E0, Ep, dL, th_LRL):
    """Given values in combo = [E0, Ep, dL, th_LRL], return a mask of just
    those obs in data DataFrame
    Returns boolean array mask"""
    mask = pd.Series([True]*len(data))
    mask = mask & (data["E0"] == E0)
    mask = mask & (data["Ep"] == Ep)
    mask = mask & (data["dL"] == dL)
    mask = mask & (data["th_LRL"] == th_LRL)
    return mask


def model_func(x, a, x0, y0):
    """Model for fitting cosine to convolved data"""
    return y0 + a*np.cos(x - x0)


def test_fits(phi0=np.pi/6, dphi=np.pi/12, plot=True):
    """Load fake data from test_convolve(), and fit the "conv" to the
    model_func y0 + a*cos(x-x0). Add the popt, pconv and fitted data to the
    DataFrame.
    return DataFrame data with "popt", "pconv", "fitconv" keys added."""
    au = atomic_units()
    # load test data
    # phi0 = 4*np.pi/6
    # dphi = np.pi/12
    data = test_convolve(phi0, dphi, plot=False)
    # get combination list
    keys = ["E0", "Ep", "dL", "th_LRL"]
    combos, vals = combinations(data, keys)
    # build placeholders
    data["a"] = pd.Series([np.NaN]*len(data))
    data["x0"] = pd.Series([np.NaN]*len(data))
    data["y0"] = pd.Series([np.NaN]*len(data))
    data["a_sigma"] = pd.Series([np.NaN]*len(data))
    data["x0_sigma"] = pd.Series([np.NaN]*len(data))
    data["y0_sigma"] = pd.Series([np.NaN]*len(data))
    data["fitconv"] = pd.Series([np.NaN]*len(data))
    # get fit parameters and data
    print()
    for i, combo in enumerate(combos):
        print("\r {0}/{1}".format(i+1, len(combos)), end="\r")
        mask = combo_mask(data, *combo)
        phis = data[mask]["phi"]
        conv = data[mask]["conv"]
        p0 = [0.5, 1, np.pi/6]  # best zero-info guess for model_func
        popt, pconv = scipy.optimize.curve_fit(model_func, phis, conv, p0)
        data.loc[mask, "a"] = [popt[0]]*sum(mask)
        data.loc[mask, "a_sigma"] = [pconv[0, 0]]
        data.loc[mask, "x0"] = [popt[1]]*sum(mask)
        data.loc[mask, "x0_sigma"] = [pconv[1, 1]]
        data.loc[mask, "y0"] = [popt[2]]*sum(mask)
        data.loc[mask, "y0_sigma"] = [pconv[2, 2]]
        data.loc[mask, "fitconv"] = model_func(phis, *popt)
    if plot is True:
        fig, ax = plt.subplots(nrows=len(combos), figsize=(6, 3*len(combos)),
                               sharex=True)
        for i, combo in enumerate(combos):
            (E0, Ep, dL, th_LRL) = combo
            mask = combo_mask(data, *combo)
            phis = data[mask]["phi"]
            ax[i].axvline(phi0, linestyle="solid", c="gray")
            ax[i].axvline((phi0 + np.pi) % (2*np.pi), linestyle="dashed",
                          c="gray")
            data[mask].plot(x="phi", y="bound_p", label="bound",
                            kind="scatter", c="C0", ax=ax[i])
            data[mask].plot(x="phi", y="conv", label="conv", c="C1", lw=2,
                            ax=ax[i])
            data[mask].plot(x="phi", y="fitconv", label="fit", c="C2", lw=2,
                            ax=ax[i])
            ax[i].plot(data[mask]["phi"],
                       data[mask]["fitconv"] - data[mask]["conv"],
                       label="diff", c="C3", lw=2)
            titlestring = ("E0 = " + str(E0/au["GHz"]) + "\tEp = " +
                           str(Ep/au["mVcm"]) + "\tdL = " + str(dL) +
                           "\tth_LRL = " + str(th_LRL/np.pi) + r"$\pi$")
            xticks, xticklabels = xticks_2p()
            ax[i].set(xticks=xticks, xticklabels=xticklabels,
                      xlabel=r"Phase $\phi$", ylabel="",
                      title=titlestring)
            ax[i].legend()
            plt.tight_layout()
    return data


def build_fits():
    """Read DataFrame from data_conv.txt". Fit convolved data to model_func()
    of the form y = y0 + a*cos(x - x0). Store these fit parametes, and the
    fitted data in new keys in the DataFrame. Write to "data_fit.txt".
    Returns data DataFrame"""
    # load convolved data
    data = pd.read_csv("data_conv.txt", index_col=0)
    # get combination list
    keys = ["E0", "Ep", "dL", "th_LRL"]
    combos, vals = combinations(data, keys)
    # build placeholders
    data["a"] = pd.Series([np.NaN]*len(data))
    data["x0"] = pd.Series([np.NaN]*len(data))
    data["y0"] = pd.Series([np.NaN]*len(data))
    data["a_sigma"] = pd.Series([np.NaN]*len(data))
    data["x0_sigma"] = pd.Series([np.NaN]*len(data))
    data["y0_sigma"] = pd.Series([np.NaN]*len(data))
    data["fitconv"] = pd.Series([np.NaN]*len(data))
    # get fit parameters and data
    funcname = "build_fits()"
    total = len(combos)
    for i, combo in enumerate(combos):
        # progress
        progress(funcname, i, total)
        # get masked phi and convolution data
        mask = combo_mask(data, *combo)
        phis = data[mask]["phi"]
        conv = data[mask]["conv"]
        # run fit
        p0 = [0.5, 1, np.pi/6]  # best zero-info guess for model_func
        popt, pconv = scipy.optimize.curve_fit(model_func, phis, conv, p0)
        # coerce fit parameters
        if popt[0] < 0:
            popt[0] = -popt[0]
            popt[1] = (popt[1] - np.pi)
        popt[1] = popt[1] % (2*np.pi)
        # add to DataFrame
        data.loc[mask, "a"] = [popt[0]]*sum(mask)
        data.loc[mask, "a_sigma"] = [pconv[0, 0]]
        data.loc[mask, "x0"] = [popt[1]]*sum(mask)
        data.loc[mask, "x0_sigma"] = [pconv[1, 1]]
        data.loc[mask, "y0"] = [popt[2]]*sum(mask)
        data.loc[mask, "y0_sigma"] = [pconv[2, 2]]
        data.loc[mask, "fitconv"] = model_func(phis, *popt)
    # save
    data.to_csv("data_fit.txt")
    return data


def HarAddTheCos(a1, phi1, a2, phi2):
    """Harmonic Addition Theorem for 2 cosines. Returns a, phi such that
    a1*Cos[x - phi1] + a2*cos(x - phi2) = a*cos(x - phi)"""
    # convert to sines so a1*sin(t + d1) + a2*sin(t + d2) = a*sin(t + d)
    d1 = -phi1 + np.pi/2
    d2 = -phi2 + np.pi/2
    # find a and d
    a = np.sqrt(abs(a1**2 + a2**2 + 2*a1*a2*np.cos(d2 - d1)))
    d = np.arctan2(a1*np.sin(d1) + a2*np.sin(d2),
                   a1*np.cos(d1) + a2*np.cos(d2))
    # convert back to cosines
    phi = -(d - np.pi/2)
    return a, phi


def plot_sums(a1, a2, g1, g2, c1, c2, a, g):
    # plot a check.
    fig, ax = plt.subplots()
    xs = np.arange(0, 2*np.pi, np.pi/100)
    y1s = model_func(xs, a1, g1, c1)
    y2s = model_func(xs, a2, g2, c2)
    ys = model_func(xs, a, g, c1+c2)
    ax.plot(xs, y1s, label="y1", lw=3)
    ax.plot(xs, y2s, label="y2", lw=1)
    ax.plot(xs, ys, label="sol", lw=3)
    ax.plot(xs, y1s + y2s, label="sum", lw=1)
    ax.axvline(np.pi/6, c="k")
    ax.axvline(g, c="grey")
    ax.legend()
    return


def build_params():
    """Build a DataFrame that doesn't have every phase as an observation.
    Instead, only stores parameters and fits. Read DataFrame from
    "data_fit.txt", pull out individual datasets using E0, Ep, dL, th_LRL,
    and store each in one obs with fit info. Write to "params.txt"
    Returns data, params DataFrames"""
    # load fit data
    data = pd.read_csv("data_fit.txt", index_col=0)
    # get combination list
    keys = ["E0", "Ep", "dL", "th_LRL"]
    combos, vals = combinations(data, keys)
    # initialize params DataFrame
    # will have Filename, E0, Ep, dL, th_LRL, a, x0, y0, a_sigma, x0_sigma,
    #   y0_sigma
    params = pd.DataFrame()
    pkeys = ["Filename", "E0", "Ep", "dL", "th_LRL", "a", "x0", "y0",
             "a_sigma", "x0_sigma", "y0_sigma"]
    funcname = "build_params()"
    total = len(combos)
    for i, combo in enumerate(combos):
        # progress
        progress(funcname, i, total)
        # get mask
        mask = combo_mask(data, *combo)
        # pick one observation to fill params entry
        obs = data[mask].iloc[0][pkeys]
        params = params.append(obs)
    params.to_csv("params.txt")
    return data, params


def dL_sums(params):
    # load data
    # data = pd.read_csv("data_fit.txt", index_col=0)
    # params = pd.read_csv("params.txt", index_col=0)
    # get combos of dL +/- pairs
    keys = ["E0", "Ep", "th_LRL"]
    combos, vals = combinations(params, keys)
    for i, combo in enumerate(combos):
        # build mask for particular combo
        mask = [True]*len(params)
        for i in range(len(keys)):
            mask = mask & (params[keys[i]] == combo[i])
        if sum(mask) != 2:  # check that mask is just 2
            print("params mask error")
        # unpack fit values from masked params
        a1 = params[mask].iloc[0]["a"]
        a2 = params[mask].iloc[1]["a"]
        g1 = params[mask].iloc[0]["x0"]
        g2 = params[mask].iloc[1]["x0"]
        c1 = params[mask].iloc[0]["y0"]
        c2 = params[mask].iloc[1]["y0"]
        # use Harmonic Addition Theorem
        a, g = HarAddTheCos(a1, g1, a2, g2)
        # build an observation to append to params
        obs = params[mask].iloc[0]
        obs["dL"] = np.NaN
        obs["a"] = a/2
        obs["x0"] = g
        obs["y0"] = (c1 + c2)/2
        obs.name = obs.name-1
        # append to params, ignore the index
        params = params.append(obs, ignore_index=True)
    return params


def th_LRL_sums(params):
    keys = ["E0", "Ep"]
    combos, vals = combinations(params, keys)
    for i, combo in enumerate(combos):
        # progress
        # build mask for particular combo
        mask = [True]*len(params)
        for i in range(len(keys)):
            mask = mask & (params[keys[i]] == combo[i])
        # add that we only want to look at dL = np.NaN
        mask = mask & np.isnan(params["dL"])
        if sum(mask) != 2:  # check tha tmask is just 2
            print("params mask error")
        # unpack fit values from masked params
        a1 = params[mask].iloc[0]["a"]
        a2 = params[mask].iloc[1]["a"]
        g1 = params[mask].iloc[0]["x0"]
        g2 = params[mask].iloc[1]["x0"]
        c1 = params[mask].iloc[0]["y0"]
        c2 = params[mask].iloc[1]["y0"]
        # print(a1, a2, g1, g2, c1, c2)
        # print(a1**2 + a2**2 + 2*a1*a2*np.cos((-g2+np.pi/2) - (-g1+np.pi/2)))
        # use Harmonic Addition Theorem
        a, g = HarAddTheCos(a1, g1, a2, g2)
        obs = params[mask].iloc[0]
        obs["dL"] = np.NaN
        obs["th_LRL"] = np.NaN
        obs["a"] = a/2
        obs["x0"] = g
        obs["y0"] = (c1 + c2)/2
        obs.name = obs.name-1
        # append to params, ignore the index
        params = params.append(obs, ignore_index=True)
    return params


def build_params_sums():
    """Use params.txt and the Harmonic Addition Theorem to combine runs of the
    same dL, and then of the same th_LRL into what experiment results should
    look like. Results of combined dL store dL = NaN, and results of the same
    th_LRL store dL = NaN and th_LRL = NaN. Writes to "params_sums.txt".
    returns DataFrame params"""
    funcname = "build_params_sums()"
    print()
    params = pd.read_csv("params.txt", index_col=0)
    print("\r{}: dL_sums()".format(funcname), end="\r")
    params = dL_sums(params)
    print("\r{}: th_LRL_sums()".format(funcname), end="\r")
    params = th_LRL_sums(params)
    print()
    params.to_csv("params_sums.txt")
    return params


def check_analysis(keys, combo, ax):
    au = atomic_units()
    [E0, Ep, dL, th_LRL] = combo
    # read data
    data = pd.read_csv("data_fit.txt", index_col=0)
    # mask out the combo
    mask = combo_mask(data, *combo)
    dmask = data[mask].copy(deep=True)
    dmask["enfinal"] = dmask["enfinal"]/au["GHz"]
    # get nans
    masknan = np.isnan(dmask["enfinal"])
    # plot
    # rows = 3
    # fig, ax = plt.subplots(nrows=rows, sharex=True, figsize=(6, 3*rows))
    xticks, xticklabels = xticks_2p()
    # titlestring = ("E0 = " + str(np.round(E0/au["GHz"], 2)) + " GHz" +
    #                "    " +
    #                "Ep = " + str(np.round(Ep/au["mVcm"], 2)) + " mV/cm" +
    #                "    " +
    #                "dL = " + str(dL) +
    #                "   " +
    #                "th_LRL = " + str(np.round(th_LRL/np.pi, 2)) + r"$\pi$")
    titlestring = ("dL = " + str(dL) + "\t" +
                   "th_LRL = " + str(np.round(th_LRL/np.pi, 2)) + r"$\pi$")
    # fig.suptitle(titlestring)
    # ax[0] : phi vs enfinal
    n = 0
    ax[n].axvline(np.pi/6, c="k")
    ax[n].axvline(7*np.pi/6, c="k")
    ax[n].axhline(0, c="k")
    dmask.plot(x="phi", y="enfinal", kind="scatter", ax=ax[0])
    ax[n].plot(dmask[masknan]["phi"], [0]*sum(masknan), 'X', c="C3")
    # beautify
    ax[n].set(xticks=xticks, xticklabels=xticklabels, title=titlestring)
    #           xlabel=r"MW Phase $\phi$", ylabel="Final Energy (GHz)",
    ax[n].tick_params(which="minor", bottom="off")
    # ax[1] : phi vs bound_p
    n = 1
    ax[n].axvline(np.pi/6, c="k")
    ax[n].axvline(7*np.pi/6, c="k")
    dmask.plot(x="phi", y="bound_p", kind="scatter", ax=ax[n])
    ax[n].plot(dmask[masknan]["phi"], dmask[masknan]["bound_p"], 'X', c="C3")
    # beautify
    ax[n].set(xticks=xticks, xticklabels=xticklabels)
    #           xlabel=r"MW Phase $\phi$", ylabel="Bound Patch")
    ax[n].tick_params(which="minor", bottom="off")
    # ax[2] : phi vs convolution
    n = 2
    ax[n].axvline(np.pi/6, c="k")
    ax[n].axvline(7*np.pi/6, c="k")
    ax[n].axhline(0, c="k")
    dmask.plot(x="phi", y="conv", ax=ax[n])
    # beautify
    ax[n].set(xticks=xticks, xticklabels=xticklabels, ylim=(0, 1),
              xlabel=r"MW Phase $\phi$ (rad.)")
    #           ylabel="Convolution")
    ax[n].legend().remove()
    ax[n].tick_params(which="minor", bottom="off")
    plt.tight_layout()
    return data


def params_plot(params, keys, combo, ax):
    # au = atomic_units()
    # unpack
    [E0, Ep, dL, th_LRL] = combo
    # build mask
    mask = [True]*len(params)
    for i in range(len(keys)):
        if np.isnan(combo[i]):
            mask = mask & np.isnan(params[keys[i]])
        else:
            mask = mask & (params[keys[i]] == combo[i])
    if sum(mask) != 1:
        print("Mask Error")
    # pull out just this observation from params
    obs = params[mask]
    a = float(obs["a"])
    x0 = float(obs["x0"])
    y0 = float(obs["y0"])
    # build data from fits
    xs = np.arange(0, 2*np.pi, np.pi/100)
    ys = model_func(xs, a, x0, y0)
    # plot
    ax.axvline(np.pi/6, c="k")
    ax.axvline(7*np.pi/6, c="k")
    ax.axhline(0, c="k")
    ax.plot(xs, ys, color="C0")
    # beautify
    xticks, xticklabels = xticks_2p()
    if np.isnan(dL):
        dLstring = "Both"
    else:
        dLstring = str(np.round(dL, 2))
    if np.isnan(th_LRL):
        thstring = "Both"
    else:
        thstring = str(np.round(th_LRL/np.pi, 2)) + r"$\pi$"
    titlestring = ("dL = " + dLstring + "        " + "th_LRL = " + thstring)
    ax.set(xticks=xticks, xticklabels=xticklabels, ylim=(0, 1),
           xlabel=r"MW Phase $\phi$ (rad.)", ylabel="Signal",
           title=titlestring)
    return


def build_fitdata_from_params(obs):
    """Given an observation from params DataFrame, pull out a, x0, y0 and use
    model_func to the fit data.
    Returns xs, ys: np.array"""
    a = float(obs["a"])
    x0 = float(obs["x0"])
    y0 = float(obs["y0"])
    xs = np.arange(0, 2*np.pi, np.pi/100)
    ys = model_func(xs, a, x0, y0)
    return xs, ys


def up_down_plot(params, E0, Ep, ax):
    # build mask for E0, Ep, dL = NaN (combined +/-1)
    mask = [True]*len(params)
    mask = mask & (params["E0"] == E0)
    mask = mask & (params["Ep"] == Ep)
    mask = mask & np.isnan(params["dL"])
    # add th_LRL = 0, pi, NaN (combined 0,pi)
    th_LRLs = np.sort(params["th_LRL"].unique())
    mask0 = mask & (params["th_LRL"] == th_LRLs[0])
    maskp = mask & (params["th_LRL"] == th_LRLs[1])
    maskn = mask & np.isnan(params["th_LRL"])
    if (sum(mask0) > 1) or (sum(maskp) > 1) or (sum(maskn) > 1):
        print("Mask Error")
    # plot 1/2 th_LRL = 0
    obs = params[mask0]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label=r"$\theta_{LRL} = 0$", lw=3)
    # plot 1/2 th_LRL = pi
    obs = params[maskp]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys/2, label=r"$\theta_{LRL} = \pi$", lw=3)
    # plot th_LRL = NaN (sum signal)
    obs = params[maskn]
    xs, ys = build_fitdata_from_params(obs)
    ax.plot(xs, ys, label=r"Signal", lw=3)
    return


def analysis_array_plot(data, E0, Ep, vals, keys, ax):
    """For a given E0, Ep, plots the data analysis pipeline onto a 3x4 array
    (ax) of axes objects. Each column are Efinal -> Bound_p -> Convolution
    vs phase plots for a particular dL = +/-1, th_LRL = 0,pi combination.
    Returns None"""
    # dL = -1, th_LRL = 0
    combo = [E0, Ep, vals["dL"][0], vals["th_LRL"][0]]
    axes = [ax[0, 0], ax[1, 0], ax[2, 0]]
    check_analysis(keys, combo, axes)
    # dL = +1, th_LRL = 0
    combo[2] = vals["dL"][1]
    combo[3] = vals["th_LRL"][0]
    axes = [ax[0, 1], ax[1, 1], ax[2, 1]]
    check_analysis(keys, combo, axes)
    # dL = -1, th_LRL = pi
    combo[2] = vals["dL"][0]
    combo[3] = vals["th_LRL"][1]
    axes = [ax[0, 2], ax[1, 2], ax[2, 2]]
    check_analysis(keys, combo, axes)
    # dL = +1, th_LRL = pi
    combo[2] = vals["dL"][1]
    combo[3] = vals["th_LRL"][1]
    axes = [ax[0, 3], ax[1, 3], ax[2, 3]]
    check_analysis(keys, combo, axes)
    return


def build_report_pdf(data, params, E0, Ep, vals, keys):
    au = atomic_units()
    # Build figure and Inner and Outer GridSpecs
    fig = plt.figure(figsize=(11, 8.5))
    gso = gridspec.GridSpec(2, 1)
    gsi = gridspec.GridSpecFromSubplotSpec(
            3, 4, subplot_spec=gso[0])
    ax = np.array([[None]*4]*3)  # axes array for inner GridSpec
    xticks, xticklabels = xticks_2p()  # retrieve custom xticks
    # Add subplots from ax array with appropriate shared axes and labels.
    ax[2, 0] = fig.add_subplot(gsi[2, 0])
    ax[1, 0] = fig.add_subplot(gsi[1, 0], sharex=ax[2, 0])
    ax[0, 0] = fig.add_subplot(gsi[0, 0], sharex=ax[2, 0])
    for i in [1, 2, 3]:
        ax[2, i] = fig.add_subplot(gsi[2, i], sharey=ax[2, 0])
        ax[1, i] = fig.add_subplot(gsi[1, i], sharex=ax[2, i], sharey=ax[1, 0])
        ax[0, i] = fig.add_subplot(gsi[0, i], sharex=ax[2, i], sharey=ax[0, 0])
    ax[2, 0].set(xticks=xticks, xticklabels=xticklabels)
    axl = fig.add_subplot(gso[1])
    # Upper Plot Grid
    analysis_array_plot(data, E0, Ep, vals, keys, ax)
    ax[0, 0].set(ylabel=r"$E_{f}$ (GHz)")
    ax[1, 0].set(ylabel="Bound")
    ax[2, 0].set(ylabel="Conv")
    # Lower Plot
    axl.axvline(np.pi/6, c="k")
    axl.axvline(7*np.pi/6, c="k")
    axl.axhline(0, c="k")
    up_down_plot(params, E0, Ep, axl)
    axl.set(xticks=xticks, xticklabels=xticklabels)
    # beautify
    axl.legend()
    titlestring = ("E0 = " + str(np.round(E0/au["GHz"], 2)) + " GHz" +
                   "    " + "    " +
                   "Ep = " + str(np.round(Ep/au["mVcm"], 2)) + " mV/cm")
    plt.suptitle(titlestring, size=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # make room for title
    fname = "E0{0}_Ep{1}.pdf".format(int(round(E0/au["GHz"])),
                                     int(round(Ep/au["mVcm"])))
    fname = os.path.join("analysis_reports", fname)
    # print(fname)
    plt.savefig(fname)
    # plt.savefig("analysis_reports\check.pdf")
    plt.close(fig)
    return fname


def build_all_reports():
    """For each E0, Ep combination in params_sums.txt, make a .pdf showing
    the analysis pipeline. Top half is a grid of each dL, th_LRL combination
    going from enfinal -> bound_p -> conv. Bottom half is the dL sum showing
    th_LRL = 0, pi on the same plot as the th_LRL sum. Outputs 'record.txt'
    of the output file, E0, and Ep for each report. Reports are named as
    E0<E0>_Ep<E0>.pdf"""
    au = atomic_units()
    data = pd.read_csv("data_fit.txt", index_col=0)
    params = pd.read_csv("params_sums.txt", index_col=0)
    keys = ["E0", "Ep", "dL", "th_LRL"]
    combos, vals = combinations(params, keys)
    E0Eps = list(itertools.product(vals["E0"], vals["Ep"]))
    record = pd.DataFrame()
    funcname = "build_all_reports()"
    total = len(E0Eps)
    for i, [E0, Ep] in enumerate(E0Eps):
        # progress
        progress(funcname, i, total)
        # print("{0}/{1}: \t E0 = {2} GHz \t Ep = {3} mV/cm".format(
        #         i+1, len(E0Eps), round(E0/au["GHz"], 3),
        #         round(Ep/au["mVcm"])))
        fname = build_report_pdf(data, params, E0, Ep, vals, keys)
        fname = "E0{0}_Ep{1}.pdf".format(int(round(E0/au["GHz"])),
                                         int(round(Ep/au["mVcm"])))
        fname = os.path.join("analysis_reports", fname)
        obs = pd.DataFrame({"E0": E0, "Ep": Ep, "fname": fname}, index=[i])
        record = record.append(obs)
    record = record.sort_values(by=["E0", "Ep"])
    record.to_csv(os.path.join("analysis_reports", "record.txt"))
    return


def stitch_reports():
    """Using record.txt, find all reports for each unique energy and stitch
    them into one document, with each report it's own page and ordered by
    increasing pulsed field Ep. Compliled reports are named as
    E0<E0>_master.pdf"""
    au = atomic_units()
    fname = os.path.join("analysis_reports", "record.txt")
    record = pd.read_csv(fname, index_col=0)
    funcname = "stitch_reports()"
    E0s = record["E0"].unique()
    for j, E0 in enumerate(E0s):
        print("{0}: E0s: {1} / {2}".format(funcname, j+1, len(E0s)))
        fname = "E0{0}_master.pdf".format(int(round(E0/au["GHz"])))
        fname = os.path.join("analysis_reports", fname)
        mask = (record["E0"] == E0)
        pdfs = list(record[mask]["fname"])
        merger = PdfFileMerger()
        total = len(pdfs)
        for i, pdf in enumerate(pdfs):
            # print(fname, "\t", pdf)
            # progress
            progress(funcname, i, total)
            merger.append(pdf)
        merger.write(fname)
    return record


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


def nanplot():
    au = atomic_units()
    data = pd.read_csv("data_fit.txt", index_col=0)
    E0s = data["E0"].unique()
    E0s = np.sort(E0s)
    Eps = data["Ep"].unique()
    Eps = np.sort(Eps)
    mask = (data["th_LRL"] == 0)
    nan0 = np.array([np.NaN]*len(Eps))
    mask0 = mask & (data["E0"] == E0s[0])
    nan1 = np.array([np.NaN]*len(Eps))
    mask1 = mask & (data["E0"] == E0s[1])
    for i, Ep in enumerate(Eps):
        maskep = (data["Ep"] == Ep)
        mask = mask0 & maskep
        nan0[i] = sum(np.isnan(data[mask]["enfinal"]))
        mask = mask1 & maskep
        nan1[i] = sum(np.isnan(data[mask]["enfinal"]))
    plt.figure()
    plt.plot(Eps/au["mVcm"], 100*nan0/400,
             label=r"$W_0$ = {} GHz".format(np.round(E0s[0]/au["GHz"], 2)))
    plt.plot(Eps/au["mVcm"], 100*nan1/400,
             label=r"$W_0$ = {} GHz".format(np.round(E0s[1]/au["GHz"], 2)))
    plt.xlabel("Pulsed Field (mV/cm)")
    plt.ylabel("% of Bad Uphill Runs")
    bad = sum(np.isnan(data["enfinal"]))
    plt.title("Total Bad Runs {0} / {1} or {2}%".format(
            bad, len(data), round(100*bad/len(data))))
    plt.legend()
    plt.tight_layout()
    plt.savefig("nanplot.pdf")
    return


def assimilate_new_data():
    """Run through every data, analysis, report building step in order to
    update all relevant files with new raw simulation data sets from the
    results folder.
    read_tidy() -> bound_pathc() -> build_convolve() -> build_fits =>
    build_params() -> build_params_sums() =>
    build_all_reports -> stitch_reports()
    """
    # data files from "results" -> data_raw.txt
    read_tidy()
    # data_raw.txt -> data_bound.txt
    bound_patch()
    # data_bound.txt -> data_conv.txt"
    build_convolve()
    # data_conv.txt -> data_fit.txt"
    build_fits()
    # data_fit.txt -> params.txt
    build_params()
    # params.txt -> params_sum.txt
    build_params_sums()
    # data_fit.txt & params_sum.txt -> reports/*.pdf
    build_all_reports()
    # reports/*.pdf -> reports/*_master.pdf
    stitch_reports()
    return


def main():
    au = atomic_units()
    data = pd.read_csv("data_fit.txt")
    mask = np.isnan(data["enfinal"])
    obs = data.loc[data[mask].index[1]]
    # print(obs)
    # print()
    E0 = np.round(obs["E0"]/au["GHz"], 1)
    Ep = np.round(obs["Ep"]/au["mVcm"], 2)
    dL = np.round(obs["dL"], 1)
    th_LRL = np.round(obs["th_LRL"]/np.pi, 1)
    phi = np.round(obs["phi"]*100/np.pi, 1)
    rstring = ("E0 = {0} GHz\nEp = {1} mV/cm\ndL = {2}\nth_LRL = {3} pi" +
               "\nphi = ({4} / 100) pi")
    print(rstring.format(E0, Ep, dL, th_LRL, phi))
    return data


data = main()

# build_all_reports()
# stitch_reports()
# print(record)
# phase_amp_plot()
# nanplot()
# assimilate_new_data()
