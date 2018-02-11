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
    spacing["dW"] = 3.7*np.power(spacing["field"],3/4)
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
    spacing["dW"] = 3.7*np.power(spacing["field"],3/4)
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


fig, ax = plt.subplots(ncols=2)
spacing, ax[0] = my_fields(ax[0])
spacing, ax[1] = rottke_fields(ax[1])
plt.tight_layout()
plt.savefig("Blue State Spacing.pdf")
