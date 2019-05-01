#!/usr/bin/env python

"""
Loop over the sites and check for years when the CO2 does something funny, i.e.
drops by a significant amount which would indicate something bogus in the CO2
data.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (17.11.2015)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import datetime as dt
from rmse import rmse

def main():

    fname = 'data/processed/g1_fluxnet_screened.csv'
    df = pd.read_csv(fname)
    sites = pd.unique(df.site)

    for i, site in enumerate(sites):
        site_df = df[df.site == site]
        site_mean = site_df.CO2.mean()
        site_co2 = site_df.CO2.values
        site_global_co2 = site_df.global_CO2.values


        site_var = []
        for j in xrange(len(site_co2)):
            site_var.append( np.abs(site_co2[j] - site_global_co2[j]) / \
                                    site_global_co2[j] * 100. )

        # Note this prints a few duplicate years, e.g. UK-Gri, this is because
        # we've stored double the info in the TropRF class
        for j in xrange(len(site_co2)):
            if site_var[j] > 15.0:
                print "'%s': '%d'," % (site, site_df.year.values[j])#, site_var[j], j
                #print site_co2
                #print site_df.global_CO2.values
                #print site_df.year.values
                #print



if __name__ == "__main__":

    main()
