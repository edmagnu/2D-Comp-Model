# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:37:15 2018

@author: edmag
"""

import numba


@numba.vectorize
def double_every_value(x):
    return x*2


df = pd.DataFrame()
