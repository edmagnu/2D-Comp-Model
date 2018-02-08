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
    """Return a dictionary of atomic units, ["GHz"], ["mVcm"], and ["ns"]"""
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


def read_tidy_turning():
    """Read in every result data file with metadata. Add every file's data into
    a tidy DataFrame.
    Returns pd.DataFrame
    """
    # specify file
    directory = ("turning")
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


def read_tidy_binding():
    """Read in every result data file with metadata. Add every file's data into
    a tidy DataFrame.
    Returns pd.DataFrame
    """
    # specify file
    directory = ("binding")
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
        data = data[["Filename", "Dir", "zi", "W", "field", "zb", "tb", "argtb",
                     "zt", "tt", "argtt"]]
        data_m = data_m.append(data)  # append to master DataFrame
    data_m.sort_values(by=["field", "W"], inplace=True)
    data_m.reset_index(drop=True, inplace=True)
    data_m.to_csv("data_raw.txt")
    return data_m


def unique_values(data, keys):
    """Produce a dictionary of unique values for "keys" in "data"
    returns dict: vals
    """
    vals = {}
    for key in keys:
        vals[key] = np.sort(data[key].unique())
    return vals

def field_tt_figure(fig, ax):
    """Plot turnign times vs orbit energy for various fields
    Returns fig, ax
    """
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
    Returns None
    """
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
    Returns None
    """
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
    returns the DataFrame holding the data
    """
    au = atomic_units()
    data = pd.read_csv("data_raw.txt", index_col=0)  # read data
    data.loc[:, "field"] = data["field"]/au["mVcm"]
    data.loc[:, "W"] = data["W"]/au["GHz"]
    data.loc[:, "t"] = data["t"]/au["ns"]
    fig, ax = plt.subplots(ncols=1, figsize=(11.5, 8))
    data.plot(kind='hexbin', x='W', y='field', C='t',
              reduce_C_function=np.max, vmin=0, vmax=30,
              colormap="cool", ax=ax)
    picked = pd.read_csv("picked.txt", index_col=0)  # import data
    picked.sort_values(by=["W"])
    picked["W"] = picked["W"]/au["GHz"]
    picked["field"] = picked["field"]/au["mVcm"]
    # Energy extent of MW transfer for 0 and -20 GHz E0
    for i in [-62, -42, 22, 42]:
        ax.axvline(i, color="black", linestyle="dashed", linewidth=2)
    # fig, ax = plt.subplots()
    keys = ["t"]
    vals = unique_values(picked, keys)
    colors = ["firebrick", "navy"]
    for i, t in enumerate(vals["t"]):
        mask = (picked["t"] == t)
        picked[mask].plot(x="W", y="field", kind="line",
                          label=r"$t_R$ = " + str(t/au["ns"]) + " ns",
                          linewidth=3, color=colors[i], ax=ax)
    ax.set(xlabel=r"$E_{orbit} = W + \Delta E_{MW}$ (GHz)",
           ylabel="Field (mV/cm)",
           xlim=(-80, 60), ylim=(0, 100), title=r"Return Time $t_R$ (ns)")
    return data


def ns_picker(times=[10,20]):
    """From data_raw.txt, for each W find the observation where t is closest to
    provide values. Then interpolate t vs field to guess field for t exactly.
    Replace t and field the observation and append it to a DataFrame picked.
    Returns picked, DataFrame
    """
    au = atomic_units()
    data = pd.read_csv("data_raw.txt", index_col=0)  # read data
    picked = pd.DataFrame()  # hold selected observations
    keys = ["Dir", "W"]  # keys to consider
    vals = unique_values(data, keys)  # get unique values for keys
    mask = pd.Series([True]*len(data))
    # mask out Dir
    key = keys[0]
    mask_t = (data[key] == vals[key][0])
    mask = mask & mask_t
    key = keys[1]
    for W in vals[key]:
        # mask out W
        print("\r", W/au["GHz"], "\t", end="\r")
        mask_t = (data[key] == W)
        mask_f = mask & mask_t
        # find index where data["t"] is closest to t
        for t in times:
            i = np.argmin(np.abs(data[mask_f]["t"] - t*au["ns"]))
            obs = data[mask_f].loc[i]
            # interpolate to find E(W=Wi, t)
            data_int = pd.DataFrame()
            data_int[["t", "field"]] = data[mask_f][["t", "field"]]
            data_int.sort_values(by="t", inplace=True)
            x = t*au["ns"]
            y = np.interp(x, xp=data_int["t"], fp=data_int["field"],
                          left=np.NaN, right=np.NaN)
            obs["field"] = y
            obs["t"] = x
            picked = picked.append(obs)
    picked.reset_index(drop=True, inplace=True)
    picked.to_csv("picked.txt")
    return picked


def picked_plot():
    """Read DataFrame from "picked.txt" to plot E(W) where
    t_T = <some specific time>. Each unique time (up to 4) is automatically
    plotted with a unique color and labeled.
    Returns DataFrame picked
    """
    au = atomic_units()
    picked = pd.read_csv("picked.txt", index_col=0)  # import data
    picked.sort_values(by=["W"])
    picked["W"] = picked["W"]/au["GHz"]
    picked["field"] = picked["field"]/au["mVcm"]
    fig, ax = plt.subplots()
    keys = ["t"]
    vals = unique_values(picked, keys)
    colors = ["C0", "C1", "C2", "C3"]
    for i, t in enumerate(vals["t"]):
        mask = (picked["t"] == t)
        picked[mask].plot(x="W", y="field", kind="line",
                          label=r"$t_R$ = " + str(t/au["ns"]),
                          color=colors[i], ax=ax)
    ax.set(xlabel=r"$E_{orbit} = W + \Delta E_{MW}$ (GHz)",
           ylabel="Field (mV/cm)")
    return picked


def field_picker(data, picks = [["tt", 20]]):
    au = atomic_units()
    # data = pd.read_csv("data_raw.txt", index_col=0)  # read data
    picked = pd.DataFrame()  # hold selected observations
    keys = ["Dir", "W"]  # keys to consider
    vals = unique_values(data, keys)  # get unique values for keys
    mask = pd.Series([True]*len(data))
    # mask out Dir
    key = keys[0]
    mask_t = (data[key] == vals[key][0])
    mask = mask & mask_t
    key = "W"
    for W in vals[key]:
        # mask out W
        print("\r W = ", np.round(W/au["GHz"],2), "\t", end="\r")
        mask_t = (data[key] == W)
        mask_w = mask & mask_t
        # find index where data["t"] is closest to t
        for pick in picks:
            kind = pick[0]
            t = pick[1]
            # kind = list(picks.keys())[0]
            mask_t = np.logical_not(np.isnan(data[kind]))
            mask_k = mask_w & mask_t
            # print("mask length ", sum(mask_k))
            if np.sum(mask_k) != 0:
                i = np.argmin(np.abs(data[mask_k][kind] - t*au["ns"]))
                # print("index = ", i)
                obs = data[mask_k].loc[i]
                # interpolate to find E(W=Wi, t)
                data_int = pd.DataFrame()
                data_int[[kind, "field"]] = data[mask_k][[kind, "field"]]
                data_int.sort_values(by=kind, inplace=True)
                x = t*au["ns"]
                y = np.interp(x, xp=data_int[kind], fp=data_int["field"],
                              left=np.NaN, right=np.NaN)
                # "correct" observation data
                obs["field"] = y
                obs["t"] = x
                obs["kind"] = kind + "=" + str(t)
                obs = obs[["Filename", "Dir", "zi", "kind", "W", "field"]]
                picked = picked.append(obs)
    picked.reset_index(drop=True, inplace=True)
    picked = picked[["Filename", "Dir", "zi", "kind", "W", "field"]]
    picked.to_csv("picked_f.txt")
    return picked


def goldylocks():
    au = atomic_units()
    data = read_tidy_binding()
    # convert to lab units
    data["field"] = data["field"]/au["mVcm"]
    data["W"] = data["W"]/au["GHz"]
    data["tt"] = data["tt"]/au["ns"]
    data["tb"] = data["tb"]/au["ns"]
    data["tplus"] = 2*data["tt"] - data["tb"]  # upper limit on bound energy
    # if tb < 20ns < tplus, electron is bound when field is turned off
    data["gdlx"] = (data["tplus"] > 20) & (data["tb"] < 20)
    # plot goldylocks zone
    fig, ax = plt.subplots()
    # data.plot.hexbin(x="W", y="field", C="gdlx", cmap="Greens", vmin=0, vmax=1,
    #                  ax = ax)
    data[data["gdlx"]].plot.scatter(x="W", y="field", color="green", ax=ax)
    # picked plot
    picked = pd.read_csv("picked.txt", index_col=0)
    picked.sort_values(by=["W"])
    picked["W"] = picked["W"]/au["GHz"]
    picked["field"] = picked["field"]/au["mVcm"]
    keys = ["t"]
    vals = unique_values(picked, keys)
    colors = ["C0", "C1", "C2", "C3"]
    for i, t in enumerate(vals["t"]):
        mask = (picked["t"] == t)
        picked[mask].plot(x="W", y="field", kind="line",
                          label=r"$t_R$ = " + str(t/au["ns"]),
                          color=colors[i], ax=ax)
    # marker lines
    ax.axvline(0, color="black")
    ax.axhline(0, color="black")
    ax.set(xlim=(-20,40), ylim=(-1,21))
    return data


def main():
    au = atomic_units()
    # data = read_tidy_binding()
    data = pd.read_csv("data_raw.txt", index_col=0)
    # data = pd.read_csv("data_raw.txt", index_col=0)
    data["tplus"] = 2*data["tt"] - data["tb"]
    # picks = {"tb": 20}
    # picks = [["tt", 20], ["tt", 10], ["tb", 20], ["tplus", 20]]
    # picked = field_picker(data, picks)
    picked = pd.read_csv("picked_f.txt", index_col=0)
    picked["W"] = picked["W"]/au["GHz"]
    picked["field"] = picked["field"]/au["mVcm"]
    fig, ax = plt.subplots()
    # ----------
    # fill between tb & tplus
    mask = (picked["kind"]=="tb=20")
    mask = mask & np.logical_not(np.isnan(picked["field"]))
    # print("tb=20 \t min \t", min(picked[mask]["W"]))
    mintb = min(picked[mask]["W"])
    # print("tb=20 \t max \t", max(picked[mask]["W"]))
    maxtb = max(picked[mask]["W"])
    mask = (picked["kind"]=="tplus=20")
    mask = mask & np.logical_not(np.isnan(picked["field"]))
    # print("t+=20 \t min \t", min(picked[mask]["W"]))
    mintp = min(picked[mask]["W"])
    # print("t+=20 \t max \t", max(picked[mask]["W"]))
    maxtp = max(picked[mask]["W"])
    minw = max(mintb, mintp)
    maxw = min(maxtb, maxtp)
    masktb = ((picked["kind"]=="tb=20") & (picked["W"] <= maxw) &
              (picked["W"] >= minw))
    masktp = ((picked["kind"]=="tplus=20") & (picked["W"] <= maxw) &
              (picked["W"] >= minw))
    # print(picked[masktp])
    # print(picked[masktb])
    ax.fill_between(picked[masktb]["W"], picked[masktb]["field"],
                    picked[masktp]["field"], color="C2")
    # ----------
    # fill below tb
    mask = (picked["kind"]=="tb=20")
    ax.fill_between(picked[mask]["W"], picked[mask]["field"], 0, color="C3")
    # ----------
    # fill below tt=20 and W<=0
    mask = (picked["kind"] == "tt=10") & (picked["W"] <= minw)
    ax.fill_between(picked[mask]["W"], picked[mask]["field"], 0,
                    color="C2")
    # ----------
    # fill between tt=10 and tplus
    mask_nan = np.logical_not(np.isnan(picked["field"]))
    mask_tt10 = mask_nan & (picked["kind"]=="tt=10")
    mask_tplus = mask_nan & (picked["kind"]=="tplus=20")
    intrp = pd.DataFrame()
    intrp["x"] = pd.Series(
            np.intersect1d(picked[mask_tt10]["W"], picked[mask_tplus]["W"]))
    intrp["tplus=20"] = np.interp(intrp["x"], picked[mask_tplus]["W"],
          picked[mask_tplus]["field"])
    intrp["tt=10"] = np.interp(intrp["x"], picked[mask_tt10]["W"],
          picked[mask_tt10]["field"])
    ax.fill_between(intrp["x"], intrp["tt=10"], intrp["tplus=20"], color="C3")
    # ----------
    # fill above tt=10 and to the left of lowest energy of tt=10
    mask = np.logical_not(np.isnan(picked["field"]))
    mask = mask & (picked["kind"]=="tt=10")
    mask_u = mask & (picked["W"]<=0)
    ax.fill_between(picked[mask_u]["W"], 100, picked[mask_u]["field"],
                    color="C9")
    mask_o = mask & (picked["W"]>=0)
    ax.fill_between(picked[mask_o]["W"], 100, picked[mask_o]["field"],
                    color="C8")
    min_tt10 = min(picked[mask]["W"])
    ax.fill_between([-100,min_tt10], 100, 0, color="C9")
    # ----------
    # plot lines
    colors = ["C2", "k", "k", "k", "k"]
    for i, kind in enumerate(picked["kind"].unique()):
        mask = (picked["kind"]==kind)
        # print(kind + "\t" + str(sum(mask)))
        test = mask & np.logical_not(np.isnan(picked["field"]))
        print(kind + "\t" + str(sum(test)))
        picked[mask].plot(x="W", y="field", c=colors[i],
                          linewidth=2, ax=ax)
    # axeslines
    ax.axvline(0, color="k", linewidth=1)
    ax.axhline(0, color="k", linewidth=1)
    # make it pretty
    ax.set(xlabel=r"$E_{orbit} = E_0 + \Delta E_{MW}$ (GHz)",
           ylabel="Field (mV/cm)", title="Uphill Electrons", xlim=(-100,100),
           ylim=(0,100))
    ax.legend().remove()
    # text boxes
    props = dict(boxstyle='round', facecolor="white", alpha=1.0)
    ax.text(-95, 95, "(a)", bbox=props)
    ax.text(5, 95, "(b)", bbox=props)
    ax.text(90, 55, "(c)", bbox=props)
    ax.text(90, 10, "(c)", bbox=props)
    ax.text(90, 32, "(d)", bbox=props)
    return picked


picked = main()