# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:58:04 2018

@author: edmag
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


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


def atomic_units():
    """Return a dictionary of atomic units"""
    au = {"GHz": 1.51983e-7, "mVcm": 1.94469e-13, "ns": 4.13414e7}
    return au


def main():
    """Read in results data file with metadata"""
    directory = ("C:\\Users\\edmag\\Documents\\Work\\" +
                 "2D-Comp-Model\\computation\\results")
    flist = os.listdir(directory)
    meta = read_metadata(directory + "\\" + flist[0])
    au = atomic_units()
    print(float(meta["Emw"])/au["mVcm"])
    return meta


meta = main()
