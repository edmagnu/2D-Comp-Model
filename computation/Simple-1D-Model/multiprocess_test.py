# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:38:00 2018

@author: edmag
"""

import multiprocessing
import time
# import os
import pandas as pd
import numpy as np

data = (['a', '2'], ['b', '4'], ['c', '6'], ['d', '8'], ['e', '1'], ['f', '3'],
        ['g', '5'], ['h', '7'])


def mp_handler(var1):
    for indata in var1:
        p = multiprocessing.Process(target=mp_worker,
                                    args=(indata[0], indata[1]))
        p.start()
    return


def mp_worker(inputs, the_time):
    print(" Processs %s\tWaiting %s seconds" % (inputs, the_time))
    time.sleep(int(the_time))
    print(" Process %s\tDONE" % inputs)
    return


def process_data(x):
    print(x[0], x[1])
    return


def func(a, b, c):
    return a**3 + b**2 + c**1


def que_test(i, a, b, c, q):
    q.put((i, a, b, c, func(a, b, c)))
    return


if __name__ == '__main__':
    c1 = time.clock()
    # freeze support
    multiprocessing.freeze_support()
    # queue
    q = multiprocessing.Queue()
    # parameters
    a = range(0, 10)
    b = range(10, 20)
    c = range(20, 30)
    i = range(0, 10)
    args = zip(i, a, b, c)
    # dataframe
    df = pd.DataFrame({'a': np.zeros(10)*np.NaN, 'b': np.zeros(10)*np.NaN,
                       'c': np.zeros(10)*np.NaN, 'r': np.zeros(10)*np.NaN})
    # workers
    workers = []
    for arg in args:
        # print(arg)
        workers = (workers +
                   [multiprocessing.Process(target=que_test, args=(*arg, q))])
    # run
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    while not q.empty():
        i, a, b, c, r = q.get()
        print(i, a, b, c, r)
        df.loc[i, ['a', 'b', 'c', 'r']] = [a, b, c, r]
    df.sort_index(inplace=True)
    c2 = time.clock()
    print(c2-c1)
