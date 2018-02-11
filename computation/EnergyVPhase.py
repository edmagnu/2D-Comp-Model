# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 00:08:57 2018

@author: edmag
"""

import pandas as pd
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt


def atomic_units():
    """Return a dictionary of atomic units"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
    return au


def model_func(x, a, phi):
    """Function to fit E_final vs phi.
    Returns a*cos(x-phi)"""
    return a*np.cos(x - phi)


def Efinal_phase():
    """Load data from model computation of E_final vs phi. Fits the data to
    model_func() and plots for select data files.
    Returns DataFrame of "a", "phi"."""
    fits = pd.DataFrame()
#    flist = ["1_Efinal_phase.txt", "2_Efinal_phase.txt", "3_Efinal_phase.txt",
#             "4_Efinal_phase.txt", "5_Efinal_phase.txt", "6_Efinal_phase.txt",
#             "7_Efinal_phase.txt", "8_Efinal_phase.txt", "9_Efinal_phase.txt"]
    flist = ["7_Efinal_phase.txt", "1_Efinal_phase.txt", "4_Efinal_phase.txt"]
    labels = ["6 V/cm", "4 V/cm", "2 V/cm"]
    colors = ["C0", "C1", "C2"]
    fields = np.array([6, 4, 2])*1000*1.94469e-13
    omega = 2.41653e-6
    folder = "C:\\Users\\edmag\\Documents\\Work\\Electron Motion\\testing\\"
    flist = [folder + file for file in flist]
    fig, ax = plt.subplots()
    for i, file in enumerate(flist):
        print(i, file)
        data = pd.read_csv(file, sep="\t", index_col=None, comment="#")
        data["efinal"] = -data["efinal"]
        p0 = [0, 0]
        pT = [3/2*(fields[i]/omega**(2/3))/1.51983e-7, np.pi/6]
        popt, pcov = scipy.optimize.curve_fit(
                model_func, data["phi"], data["efinal"], p0)
        data["fit"] = model_func(data["phi"], *pT)
        data["error"] = data["fit"] - data["efinal"]
        fit = pd.DataFrame(index=[i],
                           data={"a": popt[0], "phi": popt[1]})
        fits = fits.append(fit)
        data.plot(x="phi", y="efinal", kind="scatter", label=labels[i],
                  color=colors[i], ax=ax)
        ax.plot(data["phi"], data["fit"], linestyle="-", label='_nolegend_',
                color="black")
        # data.plot(x="phi", y="error", kind="scatter")
    ax.set_xlabel(r'$\phi_0$ (rad)')
    ax.set_xticks([np.pi/6, 4*np.pi/6, 7*np.pi/6, 10*np.pi/6])
    ax.set_xticklabels([r"$\pi$/6", r"$4\pi$/6", r"$7\pi$/6", r"$10\pi$/6"])
    ax.set_ylabel(r'$\Delta E$ (GHz)')
    plt.savefig("EvP.pdf")
    return fits


def DIL():
    """Produce a plot of the potential seen by an electron in a static field
    and the depressed ioniztionlimit as a function of static field.
    Returns DataFrames potential, dlimit
    """
    # Atomic units
    fAU1mVcm = 1.94469e-13
    enAU1GHz = 1.51983e-7
    # set up figure
    fig, ax = plt.subplots(figsize=(6, 3), ncols=2)
    # Potential DataFrame
    f = 10*fAU1mVcm
    lim = -2*(np.abs(f))**(0.5)/enAU1GHz
    rbound = (1/np.abs(f))**0.5
    zmax = 10000000
    dz = zmax/10000
    potential = pd.DataFrame({"z": np.arange(-zmax, 0, dz)})
    potential = potential.append(
            pd.DataFrame({"z": np.arange(dz, zmax + dz, dz)}))
    potential["C"] = -1/(np.abs(potential["z"]))/enAU1GHz
    potential["E"] = -f*potential["z"]/enAU1GHz
    potential["V"] = potential["C"] + potential["E"]
    # plot potential
    potential.plot(x="z", y="V", linewidth="3", ax=ax[0])
    potential.plot(x="z", y="E", linewidth="3",
                   label=r"$-E \cdot z$", ax=ax[0])
    # add Field arrow
    acent = 0
    awidth = 0.1*zmax
    ax[0].arrow(x=acent-awidth/2, y=15, dx=awidth, dy=0, width=1,
                length_includes_head=True, head_width=5,
                head_length=0.2*awidth, fc="k", ec="k")
    props = props = dict(boxstyle='round', color="white", alpha=1.0)
    ax[0].text(0.8, 0.7, r"$\vec{E}$", transform=ax[0].transAxes, fontsize=14,
               verticalalignment="top", horizontalalignment="right",
               bbox=props)
    # add indicator lines
    ax[0].axhline(0, linestyle="dashed", color="gray")
    ax[0].axhline(lim, linestyle="dashed", color="gray")
    ax[0].axvline(rbound, linestyle="dashed", color="gray")
    # make it pretty
    ax[0].set_xlim(-0.15*zmax, 0.15*zmax)
    ax[0].set_ylim(-50, 50)
    ax[0].set_xticks([rbound, 0])
    ax[0].set_xticklabels([r"$\sqrt{1/E}$", 0])
    ax[0].set_yticks([lim, 0])
    ax[0].set_yticklabels([r"$-2\sqrt{E}$", 0])
    ax[0].set_xlabel("z")
    ax[0].set_ylabel("Energy")
    ax[0].legend(loc=2, framealpha=1)
    # add (a) box
    props = props = dict(boxstyle='round', facecolor="white", alpha=1.0)
    ax[0].text(0.95, 0.95, "(a)", transform=ax[0].transAxes, fontsize=14,
               verticalalignment="top", horizontalalignment="right",
               bbox=props)
    # Limit DataFrame
    fmin = 0
    fmax = 300
    df = 0.1
    dlimit = pd.DataFrame({"field": np.arange(fmin, fmax+df, df)})
    dlimit["limit"] = -2/enAU1GHz*(dlimit["field"]*fAU1mVcm)**(0.5)
    # plot limit
    dlimit.plot(x="field", y="limit", linewidth="3", legend=None,
                ax=ax[1])
    # make it pretty
    ax[1].set_xlabel("Field (mV/cm)")
    ax[1].set_xlim(-10, 110)
    ax[1].set_ylabel("DIL (GHz)")
    ax[1].set_ylim(-70, 10)
    ax[1].set_yticks(np.arange(-70, 20, 10))
    ax[1].grid(b=True, which="major", color="black")
    ax[1].grid(b=True, which="minor")
    # add (b) box
    props = props = dict(boxstyle='round', facecolor="white", alpha=1)
    ax[1].text(0.95, 0.95, "(b)", transform=ax[1].transAxes, fontsize=14,
               verticalalignment="top", horizontalalignment="right",
               bbox=props)
    # finalize figure
    plt.tight_layout()
    plt.savefig("DIL.pdf")
    return potential, dlimit


def Potential_Plot(ax):
    """Plot of just the potential in static field.
    Returns matplotlib axes object"""
    # Atomic units
    fAU1mVcm = 1.94469e-13
    enAU1GHz = 1.51983e-7
    # set up figure
    # fig, ax = plt.subplots(figsize=(6, 3), ncols=2)
    # Potential DataFrame
    f = 10*fAU1mVcm
    lim = -2*(np.abs(f))**(0.5)/enAU1GHz
    rbound = (1/np.abs(f))**0.5
    zmax = 10000000
    dz = zmax/10000
    potential = pd.DataFrame({"z": np.arange(-zmax, 0, dz)})
    potential = potential.append(
            pd.DataFrame({"z": np.arange(dz, zmax + dz, dz)}))
    potential["C"] = -1/(np.abs(potential["z"]))/enAU1GHz
    potential["E"] = -f*potential["z"]/enAU1GHz
    potential["V"] = potential["C"] + potential["E"]
    # plot potential
    potential.plot(x="z", y="V", linewidth="3", ax=ax)
    potential.plot(x="z", y="E", linewidth="3",
                   label=r"$-E \cdot z$", ax=ax)
    # add Field arrow
    acent = 0
    awidth = 0.1*zmax
    ax.arrow(x=acent-awidth/2, y=15, dx=awidth, dy=0, width=1,
                length_includes_head=True, head_width=5,
                head_length=0.2*awidth, fc="k", ec="k")
    props = props = dict(boxstyle='round', color="white", alpha=1.0)
    ax.text(0.8, 0.7, r"$\vec{E}$", transform=ax.transAxes, fontsize=14,
               verticalalignment="top", horizontalalignment="right",
               bbox=props)
    # add indicator lines
    ax.axhline(0, linestyle="dashed", color="gray")
    ax.axhline(lim, linestyle="dashed", color="gray")
    ax.axvline(rbound, linestyle="dashed", color="gray")
    # make it pretty
    ax.set_xlim(-0.15*zmax, 0.15*zmax)
    ax.set_ylim(-50, 50)
    ax.set_xticks([rbound, 0])
    ax.set_xticklabels([r"$\sqrt{1/E}$", 0])
    ax.set_yticks([lim, 0])
    ax.set_yticklabels([r"$-2\sqrt{E}$", 0])
    ax.set_xlabel("z")
    ax.set_ylabel("Energy")
    ax.legend(loc=2, framealpha=1)
    # add (a) box
    # props = props = dict(boxstyle='round', facecolor="white", alpha=1.0)
    # ax.text(0.95, 0.95, "(a)", transform=ax.transAxes, fontsize=14,
    #         verticalalignment="top", horizontalalignment="right", bbox=props)
    return ax


def DIL_Plot(ax):
    """Plot of DIL as a function of static field.
    Returns matplotlib axes object."""
    # Atomic units
    fAU1mVcm = 1.94469e-13
    enAU1GHz = 1.51983e-7
    # Limit DataFrame
    fmin = 0
    fmax = 300
    df = 0.1
    dlimit = pd.DataFrame({"field": np.arange(fmin, fmax+df, df)})
    dlimit["limit"] = -2/enAU1GHz*(dlimit["field"]*fAU1mVcm)**(0.5)
    # plot limit
    dlimit.plot(x="field", y="limit", linewidth="3", legend=None, ax=ax)
    # make it pretty
    ax.set_xlabel("Field (mV/cm)")
    ax.set_xlim(-10, 110)
    ax.set_ylabel("DIL (GHz)")
    ax.set_ylim(-70, 10)
    ax.set_yticks(np.arange(-70, 20, 10))
    ax.grid(b=True, which="major", color="black")
    ax.grid(b=True, which="minor")
    # add (b) box
    # props = props = dict(boxstyle='round', facecolor="white", alpha=1)
    # ax.text(0.95, 0.95, "(b)", transform=ax.transAxes, fontsize=14,
    #         verticalalignment="top", horizontalalignment="right", bbox=props)
    return ax


def DIL_table():
    au = atomic_units()
    dil = pd.DataFrame()
    dil["fields"] = pd.Series(
            np.array([1, 2, 5, 10, 20, 30, 40, 60, 80, 100, 125, 150, 200,
                      300]))
    dil["dil"] = pd.Series(2*np.sqrt(dil["fields"]*au["mVcm"])/au["GHz"])
    dil["n"] = np.sqrt(1/(2*dil["dil"]*au["GHz"]))
    return dil


def DIL_and_Potential_Plot():
    fig, ax = plt.subplots(figsize=(6, 3), ncols=2)
    ax[0] = Potential_Plot(ax[0])
    ax[1] = DIL_Plot(ax[1])
    # add (a) box
    props = props = dict(boxstyle='round', color="white", alpha=1.0)
    ax[0].text(0.8, 0.7, r"$\vec{E}$", transform=ax[0].transAxes, fontsize=14,
               verticalalignment="top", horizontalalignment="right",
               bbox=props)
    # add (b) box
    props = props = dict(boxstyle='round', facecolor="white", alpha=1)
    ax[1].text(0.95, 0.95, "(b)", transform=ax[1].transAxes, fontsize=14,
               verticalalignment="top", horizontalalignment="right",
               bbox=props)
    # finalize figure
    plt.tight_layout()
    plt.savefig("DIL.pdf")
    return

# fits = Efinal_phase()
# potential, dlimit = DIL()
# dil = DIL_table()
# print(dil)
DIL_and_Potential_Plot()
