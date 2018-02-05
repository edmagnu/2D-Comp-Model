# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 06:39:51 2018

@author: edmag
"""


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
