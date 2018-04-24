# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 00:01:20 2018

@author: edmag
"""

import turning_and_binding as tab


def glob_test():
    print(lut_up[-10])
    return


def main():
    global lut_up
    global lut_down
    lut_up, lut_down = tab.import_lookup_table()
    glob_test()
    return


if __name__ == '__main__':  # run if script is called directly
    result = main()
