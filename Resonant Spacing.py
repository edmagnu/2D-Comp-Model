# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:55:55 2018

@author: labuser
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def atomic_units():
    """Return a dictionary of atomic units, ["GHz"], ["mVcm"], and ["ns"]"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
    return au


def my_fields(ax):
    au = atomic_units()
    # populate dataframe
    spacing = pd.DataFrame()
    spacing["field"] = pd.Series(np.arange(0, 300+0.1, 0.1)*au["mVcm"])
    spacing["dW"] = 3.7*np.power(spacing["field"], 3/4)
    # convert to lab units
    spacing["field"] = spacing["field"]/au["mVcm"]
    spacing["dW"] = spacing["dW"]/au["GHz"]
    # plot
    spacing.plot(x="field", y="dW", label=r"3.7 $E^{3/4}$", ax=ax)
    # ax.legend_.remove()
    ax.legend()
    ax.set(xlabel="Field (mV/cm)", ylabel="Resonance Spacing (GHz)",
           title="Our Fields")
    ax.grid(True)
    return spacing, ax


def rottke_fields(ax):
    au = atomic_units()
    # populate dataframe
    spacing = pd.DataFrame()
    spacing["field"] = pd.Series(np.arange(0, 10000, 1)*1000*au["mVcm"])
    spacing["dW"] = 3.7*np.power(spacing["field"], 3/4)
    # convert to lab units
    spacing["field"] = spacing["field"]/(1000*au["mVcm"])  # V/cm
    spacing["dW"] = spacing["dW"]/(30*au["GHz"])  # cm^-1
    # plot
    spacing.plot(x="field", y="dW", label=r"3.7 $E^{3/4}$", ax=ax)
    plt.plot(5714, 25, 'xk', label="est. Rottke 1986")
    # ax.axvline(5714, color="black")
    # ax.axhline(25, color="black")
    # ax.legend_.remove()
    ax.legend()
    ax.set(xlabel="Field (V/cm)", ylabel=r"Resonance Spacing ($cm^-1$)",
           title="Rottke 1986 Fields")
    ax.grid(True)
    return spacing, ax


def FieldResFig():
    fig, ax = plt.subplots(ncols=2)
    spacing, ax[0] = my_fields(ax[0])
    spacing, ax[1] = rottke_fields(ax[1])
    plt.tight_layout()
    plt.savefig("Blue State Spacing.pdf")
    return fig, ax


def ResRetComp():
    au = atomic_units()
    # my data
    data_master = pd.read_csv("computation\\Turning Time\\data_raw.txt",
                              index_col=0)
    data_master = data_master[(data_master["Dir"] == -1.0)].copy(deep=True)
    fig, ax = plt.subplots()
    Wlist = data_master["W"].unique()[[100, 150, 200, 250, 300]]
    # Wlist = data_master["W"].unique()[[200]]
    for W in Wlist:
        data = data_master[(data_master["W"] == W)].copy(deep=True)
        data.sort_values(by="field", inplace=True)
        data["field"] = data["field"]/au["mVcm"]
        data["tt"] = data["tt"]/au["ns"]
        ax.plot(data["field"], data["tt"], label="_nolegend_")
        # data.plot(x="field", y="tt", ax=ax,
        #           label=(r"$t_T$, W = " + str(np.round(W/au["GHz"],2))
        #                  + " GHz"))
    # WKB
    # populate dataframe
    spacing = pd.DataFrame()
    spacing["field"] = pd.Series(np.arange(0, 300+1, 1)*au["mVcm"])
    spacing["dW"] = 3.7*np.power(spacing["field"], 3/4)
    spacing["field"] = spacing["field"]/au["mVcm"]  # V/cm
    spacing["dW"] = spacing["dW"]/au["GHz"]  # cm^-1
    spacing["tt"] = 1/(2*spacing["dW"])
    # plot
    spacing.plot(x="field", y="tt", linestyle="--", linewidth=3, ax=ax,
                 # label=r"$1/2 \cdot dn_1/dW$")
                 label=r"$\frac{1}{2} \cdot (3.7 E^{3/4})^{-1}$")
    # text boxes
    props = dict(boxstyle='round', color="white", alpha=1.0)
    ax.text(290, 2.4, "100 GHz", verticalalignment="center",
            horizontalalignment="center", bbox=props)
    ax.text(290, 1.5, " 50 GHz", verticalalignment="center",
            horizontalalignment="center", bbox=props)
    ax.text(290, 1.0, "  0 GHz", verticalalignment="center",
            horizontalalignment="center", bbox=props)
    ax.text(290, 0.55, "- 50 GHz", verticalalignment="center",
            horizontalalignment="center", bbox=props)
    ax.text(290, 0.33, "-100 GHz", verticalalignment="center",
            horizontalalignment="center", bbox=props)
    # pretty
    ax.set(yscale="log", xlabel="Field (mV/cm)", ylabel=r"Turning Time (ns)")
    ax.grid(True)
    ax.legend(fontsize=16)
    # plt.tight_layout()
    plt.savefig("ResRetComp.pdf")
    return


def NonZero_W():
    au = atomic_units()
    # my data
    data_master = pd.read_csv("computation\\Turning Time\\data_raw.txt",
                              index_col=0)
    data_master = data_master[(data_master["Dir"] == -1.0)].copy(deep=True)
    fig, ax = plt.subplots()
    colors = ["C0", "C1", "C2", "C3", "C4"]
    factors = [1.05, 1.23, 2]
    Wlist = data_master["W"].unique()[[[180, 200, 220]]]
    # Wlist = data_master["W"].unique()[[200]]
    for i, W in enumerate(Wlist):
        data = data_master[(data_master["W"] == W)].copy(deep=True)
        data.sort_values(by="field", inplace=True)
        data["field"] = data["field"]/au["mVcm"]
        data["tt"] = data["tt"]/au["ns"]
        ax.plot(data["field"], data["tt"], label="_nolegend_", c=colors[i])
    # WKB
    # populate dataframe
    # W = Wlist[0]
    for i, W in enumerate(Wlist):
        spacing = pd.DataFrame()
        spacing["field"] = pd.Series(np.arange(1, 300+1, 1)*au["mVcm"])
        spacing["n"] = np.power(np.power(W**2 + 3*spacing["field"], 0.5) - W,
                                -0.5)
        spacing["dW"] = (np.power(spacing["n"], -3) +
                         3*spacing["n"]*spacing["field"])
        spacing["field"] = spacing["field"]/au["mVcm"]  # V/cm
        spacing["dW"] = spacing["dW"]/au["GHz"]  # cm^-1
        spacing["tt"] = factors[i]*1/(2*spacing["dW"])
        # plot
        spacing.plot(x="field", y="tt", linestyle="--", linewidth=3, ax=ax,
                     # label=r"$1/2 \cdot dn_1/dW$")
                     label=(str(factors[i]) + r"$\cdot dW/dn$" + " at " +
                            str(np.round(W/au["GHz"], 2)) + " GHz"),
                     c=colors[i])
    ax.set(xscale="log", yscale="log", xlabel="Field (mV/cm)",
           ylabel="Turning Time (ns)")
    plt.tight_layout()
    plt.savefig("NonZeroW.pdf")
    return


ResRetComp()
NonZero_W()
