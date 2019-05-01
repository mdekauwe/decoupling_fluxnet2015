#!/usr/bin/env python

"""
Run a filter over omega fits and remove the "bad" sites, see definitions
below

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (05.11.2015)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pandas as pd
import numpy as np
import calendar
import datetime as dt
import gdal


def main(infname, ofname):

    df = pd.read_csv(infname, sep=",")

    # No precip data at all in the summer
    # end up with a crazy g1 - 43.96
    bad_sites = {'UK-Gri': '2005'}

    for site,yr in bad_sites.iteritems():
        df = df.drop( df[(df.site == site) & (df.year == int(yr))].index )

    #drop sites which have been clearcut / burnt / planted recently
    df = df.drop( df[(df.site == 'CA-Ca2')].index )  # British Columbia- Campbell River - Clearcut Site
    df = df.drop( df[(df.site == 'CA-NS7')].index )  # UCI-1998 burn site
    df = df.drop( df[(df.site == 'CA-Qcu')].index )  # Quebec Boreal Cutover Site
    df = df.drop( df[(df.site == 'CA-SF3')].index )  # Sask. - Fire 1998
    df = df.drop( df[(df.site == 'CA-TP1')].index )  # Ontario- Turkey Point Seedling White Pine
    df = df.drop( df[(df.site == 'US-Bn3')].index )  # AK - Bonanza Creek 1999 Burn site near Delta Junction
    df = df.drop( df[(df.site == 'US-Me1')].index )  # OR - Metolius - Eyerly burn
    df = df.drop( df[(df.site == 'US-NC1')].index )  # NC - NC_Clearcut
    df = df.drop( df[(df.site == 'US-SO4')].index )  # CA - Sky Oaks- New Stand
    df = df.drop( df[(df.site == 'US-SP2')].index )  # FL - Slashpine-Mize-clearcut-3yrregen
    df = df.drop( df[(df.site == 'US-Wi7')].index )  # WI - Red pine clearcut (RPCC)
    df = df.drop( df[(df.site == 'US-Wi8')].index )  # WI - Young hardwood clearcut (YHW)

    # The Chinese poplar site is deciduous not evergreen.
    # It is also a relatively young plantation: LAI = 0.23 - best to drop?
    df = df.drop( df[(df.site == 'CN-Ku1')].index )  # CN-Ku1 Kubuqi_populus forest (K04)

    # Fix missing C4 PFTs in crop fits.
    df.loc[(df.site == "NL-Lan") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "NL-Lan") & (df.year == 2006), "PFT"] = "C4C"
    df.loc[(df.site == "US-ARM") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo1") & (df.year == 1997), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo1") & (df.year == 1999), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo1") & (df.year == 2001), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo1") & (df.year == 2003), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo1") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo1") & (df.year == 2007), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo2") & (df.year == 2004), "PFT"] = "C4C"
    df.loc[(df.site == "US-Bo2") & (df.year == 2006), "PFT"] = "C4C"
    df.loc[(df.site == "US-IB1") & (df.year == 2006), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne1") & (df.year == 2001), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne1") & (df.year == 2002), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne1") & (df.year == 2003), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne1") & (df.year == 2004), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne1") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne2") & (df.year == 2001), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne2") & (df.year == 2003), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne2") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne3") & (df.year == 2001), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne3") & (df.year == 2003), "PFT"] = "C4C"
    df.loc[(df.site == "US-Ne3") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "DK-Fou") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "FR-Gri") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "IT-BCi") & (df.year == 2004), "PFT"] = "C4C"
    df.loc[(df.site == "IT-BCi") & (df.year == 2005), "PFT"] = "C4C"
    df.loc[(df.site == "IT-BCi") & (df.year == 2006), "PFT"] = "C4C"

    # fix names
    df.loc[df.PFT == "CSH", "PFT"] = "SHB"
    df.loc[df.PFT == "OSH", "PFT"] = "SHB"
    df.loc[df.PFT == "WSA", "PFT"] = "SAV"
    df.loc[df.PFT == "GRA", "PFT"] = "C3G"
    df.loc[df.PFT == "CRO", "PFT"] = "C3C"
    df.loc[df.PFT == "TropRF", "PFT"] = "TRF"

    #df = df.sort_values(by=['latitude','year'], ascending=[False,True])
    df = df.sort_values(by=['site','year'], ascending=[True,True])

    # load C3/C4 fracs
    fdir = "data/raw_data/C4_global_map/"
    ds = gdal.Open(os.path.join(fdir, 'wcs.tiff'))
    c4_lookup = np.array(ds.GetRasterBand(1).ReadAsArray())

    #plt.imshow(c3_c4)
    #plt.show()
    #sys.exit()
    ncols = 720
    nrows = 360
    (lllon, lllat, urlon, urlat) = -180.0, -90.0, 180.0, 90.0
    dlon = (urlon - lllon) / ncols
    dLat = (urlat - lllat) / nrows
    lons = np.arange(lllon, urlon, dlon)
    lats = np.arange(lllat, urlat, dLat)
    lons, lats = np.meshgrid(lons, lats)

    # Can't immediately see why, but my lats are upside down, so flip it
    # the other way to correct it.
    lats = np.flipud(lats)

    #plt.imshow(lats)
    #plt.imshow(lons)
    #plt.colorbar()
    #plt.show()
    #sys.exit()

    df['c4_frac'] = np.ones(len(df)) * -999.9
    for i, row in df.iterrows():

        # find closest match
        distance = (np.abs(lats - row.latitude) +
                    np.abs(lons - row.longitude))
        idx = np.where(distance == distance.min())

        df.loc[i, 'c4_frac'] = c4_lookup[idx][0]

    df = df.reset_index() # need to do this to get the correct indexes

    #"""
    fdir = "data/raw_data/free_and_fair_use/"
    df_free = pd.read_csv(os.path.join(fdir, "free_sites.csv"), skiprows=0)

    # Screen just for free-use sites

    idx = []
    for i, row_all in df.iterrows():
        for j, row_free in df_free.iterrows():

            if (row_all.site == row_free.site and
                int(row_all.year) == row_free.yr):
                idx.append(i)


    df.to_csv(ofname, index=False)


if __name__ == "__main__":


    infname = 'data/processed/omega_fluxnet_PM.csv'
    ofname = 'data/processed/omega_fluxnet_screened_PM.csv'
    main(infname, ofname)
