# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 17:01:42 2018

@author: labuser
"""

# A simple 1D model for the phase dependent ionization and recombination in
# static fields

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def atomic_units():
    """Return a dictionary of atomic units"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
    return au


def dWi(phi, Emw, fmw):
    """Energy exchange from leaving the core for the first time."""
    return (3./2.) * (Emw / (fmw**(2./3.)) ) * np.cos(phi - np.pi/6)


def dWs(phi, Emw, fmw):
    """Energy exchange from slingshotting the core."""
    return np.sqrt(3) * (3./2.) * (Emw / (fmw**(2./3.)) ) * np.cos(phi)


au = atomic_units()
fname = os.path.join("..", "Turning Time", "data_raw.txt")
data = pd.read_csv(fname, index_col=0)
