# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 06:39:51 2018

@author: edmag
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


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
    directory = ("results")
    flist = os.listdir(directory)
    data_m = pd.DataFrame()  # initialize DataFrame
    for i, file in enumerate(flist):
        print("{0} / {1} \t".format(i+1, len(flist)), end="\r")
        # print(i, end="\r")
        fname = directory + "\\" + file  # build file
        # print(fname)
        # load metadata and data
        meta = read_metadata(fname)
        data = pd.read_csv(fname, sep="\t", comment="#", index_col=False)
        # add in metadata
        data["Filename"] = pd.Series([fname]*len(data), dtype=str)
        data["Dir"] = pd.Series([meta["Dir"]]*len(data), dtype=float)
        data["zi"] = pd.Series([meta["zi"]]*len(data), dtype=float)
        # data["field"] = pd.Series([meta["field"]]*len(data), dtype=float)
        # organize data
        data = data[["Filename", "Dir", "zi", "W", "field", "zt", "t", "argt"]]
        data_m = data_m.append(data)  # append to master DataFrame
    data_m.sort_values(by=["field", "W"], inplace=True)
    data_m.reset_index(drop=True, inplace=True)
    data_m.to_csv("data_raw.txt")
    return data_m


def unique_values(data, keys):
    """Produce a dictionary of unique values for "keys" in "data"
    returns dict: vals"""
    vals = {}
    for key in keys:
        vals[key] = np.sort(data[key].unique())
    return vals

def field_tt_figure(fig, ax):
    """Plot turnign times vs orbit energy for various fields
    Returns fig, ax"""
    au = atomic_units()  # atomic units
    data = pd.read_csv("data_raw.txt", index_col=0)  # read data
    # build unique values
    keys = ["Dir", "zi", "field"]
    vals = unique_values(data, keys)
    mask = pd.Series([True]*len(data))
    # mask Dir and zi (not really necessary, all the same for now)
    # must be explicit about key, value, etc to prevent memory problems.
    for i in [0, 1]:
        key = keys[i]
        value = vals[key][0]
        # print(key, value)
        mask_t = (data[key] == value)
        mask = mask & mask_t
    # fig, ax = plt.subplots()
    # mark where field turns off
    ax.axhline(20, color="grey", linestyle="dashed")
    # mark where electron returns before field off
    ax.axhline(10, color="grey", linestyle="dashed")
    # plot for several fields, i-1 = field in mV/cm
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i,f in enumerate(np.array([10, 25, 50, 100, 200]) + 1):
        # build field mask and plot
        key = "field"
        value = vals[key][f]
        mask_t = (data[key] == value)
        mask_plot = mask & mask_t
        ax.plot(data[mask_plot]["W"]/au["GHz"], data[mask_plot]["t"]/au["ns"],
                "-", color=colors[i], linewidth=3)
    ax.set_xlabel("Launch Energy $v^2/2 - 1/r$ (GHz)")
    ax.set_ylabel("Turning Time (ns)")
    return fig, ax


def field_tt_full():
    """Full plot with labels using field_tt_figure()
    Returns None"""
    fig, ax = plt.subplots()
    labels = ["10 mV/cm", "25 mV/cm", "50 mV/cm", "100 mV/cm", "200 mV/cm"]
    fig, ax = field_tt_figure(fig, ax)
    ax.set_xlim(-200, 250)
    ax.set_ylim(0, 40)
    # writing
    # 20 ns line
    ax.text(-190, 21, "Field Off",
            verticalalignment="bottom", horizontalalignment="left")
    ax.text(-190, 11, "Returns before Field Off",
            verticalalignment="bottom", horizontalalignment="left")
    # Line Label LOCations
    lllocs = [[60, 38], [185, 32], [185,15], [180, 7], [170, 2]]
    for i in [0, 1, 2, 3, 4]:
        ax.text(lllocs[i][0], lllocs[i][1], labels[i],
                verticalalignment="bottom", horizontalalignment="left")
    plt.tight_layout()
    return

def field_tt_close():
    """Close plot at range accessible do to MW energy exchange, with labels,
    using field_tt_figure()
    Returns None"""
    fig, ax = plt.subplots()
    ax.axvline(-62, color="grey", linestyle="dashed")
    ax.axvline(-42, color="grey", linestyle="dashed")
    ax.axvline(22, color="grey", linestyle="dashed")
    ax.axvline(42, color="grey", linestyle="dashed")
    fig, ax = field_tt_figure(fig, ax)
    labels = ["10 mV/cm", "25 mV/cm", "50 mV/cm", "100 mV/cm", "200 mV/cm"]
    ax.set_xlim(-90, 80)
    ax.set_ylim(0, 50)
    # writing
    props = props = dict(boxstyle='round', color="white", alpha=1.0)
    # 20 ns line
    ax.text(-88, 20, "Field Off",
            verticalalignment="center", horizontalalignment="left",
            bbox=props)
    # 10 ns line
    ax.text(-88, 10, "Returns \nbefore \nField Off",
            verticalalignment="center", horizontalalignment="left",
            bbox=props)
    # -62 GHz line
    ax.text(-62, 44, r"$W = -20$ GHz" + "\n" + "Minimum",
            verticalalignment="top", horizontalalignment="center",
            bbox=props)
    # -42 GHz line
    ax.text(-42, 49, r"$W = 0$ GHz" + "\n" + "Minimum",
            verticalalignment="top", horizontalalignment="center",
            bbox=props)
    # 22 GHz line
    ax.text(22, 44, r"$W = -20$ GHz" + "\n" + "Maximum",
            verticalalignment="top", horizontalalignment="center",
            bbox=props)
    # 42 GHz line
    ax.text(42, 49, r"$W = 0$ GHz" + "\n" + "Maximum",
            verticalalignment="top", horizontalalignment="center",
            bbox=props)
    # Line Label LOCations
    lllocs = [[55, 42], [55, 20], [55,10], [55, 6], [55, 1]]
    for i in [0, 1, 2, 3, 4]:
        ax.text(lllocs[i][0], lllocs[i][1], labels[i],
                verticalalignment="bottom", horizontalalignment="left",
                bbox=props)
    plt.tight_layout()
    plt.savefig("fields.pdf")
    return


def heatmap():
    """Produce a heatmap of Energy & Field vs Turnign time.
    returns the DataFrame holding the data"""
    au = atomic_units()
    data = pd.read_csv("data_raw.txt", index_col=0)  # read data
    data.loc[:, "field"] = data["field"]/au["mVcm"]
    data.loc[:, "W"] = data["W"]/au["GHz"]
    data.loc[:, "t"] = data["t"]/au["ns"]
    fig, ax = plt.subplots(ncols=1, figsize=(11.5, 8))
    data.plot(kind='hexbin', x='W', y='field', C='t',
              reduce_C_function=np.max, vmin=0, vmax=30,
              colormap="brg",
              xlim=(-100, 100), ylim=(0, 125), title="Return Time (ns)",
              ax=ax)
    #           ax=ax[0])
    # data.plot(kind='hexbin', x='W', y='field', C='t',
    #           reduce_C_function=np.max, vmin=0, vmax=20, colormap="bwr",
    #           xlim=(-100, 100), ylim=(0,125), title="Return by 20ns (10ns)",
    #           ax=ax[1])
    # plt.savefig("heatmap.pdf")
    return data


def ns_picker(t=20):
    """Find the field for which a particular launch energy will have a
    particular return time in ns.
    Return None"""
    au = atomic_units()
    data = pd.read_csv("data_raw.txt", index_col=0)  # read data
    picked = pd.DataFrame()  # hold selected observations
    keys = ["Dir", "field"]  # keys to consider
    vals = unique_values(data, keys)  # get unique values for keys
    mask = pd.Series([True]*len(data))
    # mask out Dir
    key = keys[0]
    mask_t = (data[key] == vals[key][0])
    mask = mask & mask_t
    # mask out a field
    key = keys[1]
    for field in vals[key]:
        print(field/au["mVcm"], "\t", end="\r")
        mask_t = (data[key] == field)
        mask_f = mask & mask_t
        i = np.argmin(np.abs(data[mask_f]["t"] - t*au["ns"]))
        obs = data[mask_f].loc[i]
        picked = picked.append(obs)
    return data, mask_f, picked


def main():
    au = atomic_units()
    data, mask_f, picked = ns_picker(20)
    picked.sort_values(by=["W"])
    picked["W"] = picked["W"]/au["GHz"]
    picked["field"] = picked["field"]/au["mVcm"]
    picked.plot(x="W", y="field", kind="line")
    return

au = atomic_units()
# heatmap()
# data = pd.read_csv("data_raw.txt", index_col=0)
main()
# field_tt_full()
# field_tt_close()

