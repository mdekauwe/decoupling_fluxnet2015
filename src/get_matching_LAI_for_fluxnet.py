#!/usr/bin/env python

"""
Get the LAI for the EBF sites

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (14.03.2017)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import brewer2mpl
import codecs
import matplotlib
from getSingleModisPixel import grabMODISdata
#from suds.client import *

def main():

    fdir = "data/processed"
    f = codecs.open(os.path.join(fdir, "omega_fluxnet_screened_PM.csv"), "r",
                    encoding='utf-8', errors='ignore')
    df = pd.read_csv(f)
    df = df[df.omega >= 0]
    #df = df[(df.PFT == "EBF") | (df.PFT == "DBF") | (df.PFT == "TRF")]

    fdir = "data/raw_data/anna_meta"
    f = codecs.open(os.path.join(fdir, "site_metadata.csv"), "r",
                    encoding='utf-8', errors='ignore')
    df_m = pd.read_csv(f)


    df["LAI"] = np.zeros(len(df))


    sites = np.unique(df.site)
    for s in sites:
        lat = df[df.site == s].latitude.values[0]
        try:
            ht = df_m[df_m.SiteCode == s].TowerHeight.values[0]
            p = df[df.site == s].PFT.values[0]
            if ~np.isnan(ht):
                print(s, p, np.mean(df[df.site == s].omega.values), ht)
        except IndexError:
            pass
    sys.exit()

    year = 2006
    km_ab = 0
    km_lr = 0
    product = "MOD15A2"
    band = "Lai_1km"
    band_qa = "FparLai_QC"
    good_QA = np.array([0]).astype(np.int16)

    #sites = np.unique(df.site)
    #sites = ['AU-Tum' 'AU-Wac' 'CA-Oas' 'CN-Bed' 'DE-Hai' 'DK-Sor' 'FR-Fon' 'FR-Hes'
    #         'FR-Pue' 'GF-Guy' 'ID-Pag' 'IS-Gun' 'IT-Col' 'IT-Cpz' 'IT-Non' 'IT-PT1'
    #         'IT-Ro1' 'IT-Ro2' 'PT-Esp' 'PT-Mi1' 'UK-Ham' 'UK-PL3' 'US-Bar' 'US-Bn2'
    #         'US-Dk2' 'US-Ha1' 'US-MMS' 'US-MOz' 'US-UMB' 'US-WCr' 'VU-Coc']
    sites = ['US-Bn2','US-Dk2','US-Ha1','US-MMS','US-MOz','US-UMB','US-WCr','VU-Coc']

    for s in sites:

        # 46 8-days
        out = np.zeros(len(range(1, 365, 8)))
        count = np.zeros(len(range(1, 365, 8)))

        lat = df[df.site == s].latitude.values[0]
        lon = df[df.site == s].longitude.values[0]

        #for year in xrange(2001, 2003):
        year = 2000
        wsdlurl = 'https://daacmodis.ornl.gov/cgi-bin/MODIS/GLBVIZ_1_Glb_subset/MODIS_webservice.wsdl'
        client  = Client(wsdlurl)
        (doy, data,
         data_stdev, sds_name) = grabMODISdata(lat, lon, product, band,
                                               band_qa, year, km_ab, km_lr,
                                               good_QA, client)

        #doy = doy[~np.isnan(data)]
        #data = data[~np.isnan(data)]
        for i in range(len(doy)):
            if ~np.isnan(data[i]):
                out[i] += np.mean(data[i]) # remove weird extra bracket
                count[i] += 1.0

        count = count[~np.isnan(out)]
        out = out[~np.isnan(out)]

        out = np.where(count > 0, out/count, out)

        # to deal with noise take the mean of the highest 3 values averaged
        # across years
        mean_lai = np.mean(np.sort(out)[::-1][0:3])
        print(s, mean_lai, np.mean(df[df.site == s].omega.values))
        #df.loc[(df.site == s), "LAI"] = mean_lai


if __name__ == "__main__":

    main()
