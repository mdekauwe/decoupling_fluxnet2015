#!/usr/bin/env python
"""
Estimate omega from FLUXNET2015 data.

Data from:

/srv/ccrc/data04/z3509830/Fluxnet_data/FLUXNET2016/Original_data/Halfhourly_qc_fixed
/srv/ccrc/data04/z3509830/Fluxnet_data/FLUXNET2016/Original_data/Hourly_qc_fixed

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (01.05.2019)"
__email__ = "mdekauwe@gmail.com"

#import matplotlib
#matplotlib.use('agg') # stop windows popping up

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import datetime as dt
from scipy.stats import pearsonr
from rmse import rmse
import scipy.stats as stats
import re

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import constants as c
from penman_monteith import PenmanMonteith
from estimate_pressure import estimate_pressure


class FitOmega(object):
    """
    Fit Jarvis & McNaughton decoupling factor by inverting PM against
    fluxnet data using the lmfit package
    """

    def __init__(self, fdir, adir, co2dir, ofdir, site_fname,
                 global_co2_fname, ofname):
        print(fdir)
        self.flist = glob.glob(os.path.join(fdir, "*.csv"))
        self.site_fname = os.path.join(adir, site_fname)
        self.global_co2_fname = os.path.join(co2dir, global_co2_fname)
        self.ofname = os.path.join(ofdir, ofname)

        self.out_cols = ["site","name","country","year","latitude",\
                         "longitude","PFT","n","ebr","omega","wind",\
                         "ga","gs","lai","canopy_ht","tower_ht","map",
                         "sprecip"]
        self.oplots = "plots"
        self.outputs = "outputs"

        if not os.path.exists(self.oplots):
            os.mkdir(self.oplots)
        if not os.path.exists(self.outputs):
            os.mkdir(self.ofname)

    def main(self, hour=False):

        df_site = pd.read_csv(self.site_fname)
        #df_out = pd.DataFrame(columns=self.out_cols)

        bad_sites = ["US-PFa", "NO-Blv", "CH-Fru", "JP-SMF", "CH-Oe2", \
                     "ES-Ln2", "CH-Lae", "JP-MBF"]

        for i, fname in enumerate(self.flist):

            s = os.path.basename(fname).split(".")[0].split("_")[1].strip()
            if s in bad_sites:
                continue

            d = self.get_site_info(df_site, fname)
            print("%s %s" % (d['site'], d['yrs']))



            df = self.read_file(fname)

            (df, d, no_G) = self.filter_dataframe(df, d, hour)


            # Estimate gs from inverting the penman-monteith
            (df, error) = self.penman_montieth_wrapper(d, df, no_G)

            if error == False:
                dfx = df[['omega']]
                df_m = dfx.resample('M').mean()
                df_s = dfx.resample('M').std()
                df_c = dfx.resample('M').count()
                df_m = df_m.rename(columns={'omega':'omega_mu'})
                df_s = df_s.rename(columns={'omega':'omega_sigma'})
                df_c = df_c.rename(columns={'omega':'omega_count'})


                dfx = df.copy()
                dfx = dfx[['GPP', 'ET', 'ga', 'gs']]
                #dfx = dfx[['GPP', 'ET']]

                if hour:
                    # mol m-2 s-1 to kg m-2 hr-1
                    conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HR
                    dfx.loc[:, 'ga'] *= conv
                    dfx.loc[:, 'gs'] *= conv

                    # mol m-2 s-1 to kg m-2 s-1  (mm hr-1)
                    conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HR
                    dfx.loc[:, 'ET'] *= conv
                else:
                    # mol m-2 s-1 to mol m-2 hr-1
                    conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HLFHR
                    dfx.loc[:, 'ga'] *= conv
                    dfx.loc[:, 'gs'] *= conv

                    # mol m-2 s-1 to kg m-2 s-1  (mm hr-1)
                    conv = c.MOL_WATER_2_G_WATER * c.G_TO_KG * c.SEC_TO_HLFHR
                    dfx.loc[:, 'ET'] *= conv

                df_sum = dfx.resample('M').sum()
                df_sum = df_sum.rename(columns={'GPP':'GPP_g_c_m2_month',
                                                'ET':'ET_mm_month',
                                                'gs':'gs_mm_month',
                                                'ga':'gs_mm_month'})

                dfx = df.copy()
                dfx = dfx[['VPD', 'gs', 'ga']]
                conv = c.PA_TO_KPA
                dfx.loc[:, 'VPD'] *= conv
                df_mean = dfx.resample('M').mean()
                df_mean = df_mean.rename(columns={'VPD':'VPD_kPa',
                                                  'gs':'gs_mol_m2_s1',
                                                  'ga':'ga_mol_m2_s'})

                df_out = pd.concat([df_m,df_s,df_c,df_sum,df_mean],axis=1)
                #df_out = df_out.dropna()

                self.make_plot(d, df_out)

                ofname = os.path.join(self.outputs, \
                                      "%s_omega.csv" % (d['site']))
                if os.path.exists(ofname):
                    os.remove(ofname)
                df_out.to_csv(ofname, index=True)

    def read_file(self, fname):

        date_parse = lambda x: pd.datetime.strptime(x, '%Y%m%d%H%M%S')
        df = pd.read_csv(fname, index_col='TIMESTAMP_START',
                         parse_dates=['TIMESTAMP_START'],
                         date_parser=date_parse)
        df.index.names = ['date']

        # Using ERA interim filled met vars ... _F
        df = df.rename(columns={'LE_F_MDS': 'Qle', 'H_F_MDS': 'Qh',
                                'VPD_F_MDS': 'VPD', 'TA_F': 'Tair',
                                'NETRAD': 'Rnet', 'G_F_MDS': 'Qg',
                                'WS_F': 'Wind', 'P_F': 'Precip',
                                'USTAR': 'ustar', 'LE_CORR': 'Qle_cor',
                                'H_CORR': 'Qh_cor', 'CO2_F_MDS': 'CO2air',
                                'CO2_F_MDS_QC': 'CO2air_qc', 'PA_F': 'Psurf',
                                'G_F_MDS_QC': 'Qg_qc',
                                'LE_F_MDS_QC': 'Qle_qc', 'H_F_MDS_QC': 'Qh_qc',
                                'LE_CORR_JOINTUNC': 'Qle_cor_uc',
                                'H_CORR_JOINTUNC': 'Qh_cor_uc',
                                'GPP_NT_VUT_REF': 'GPP'})

        df = df[['Qle', 'Qh', 'VPD', 'Tair', 'Rnet', 'Qg', 'Wind', \
                 'Precip', 'ustar', 'Qle_cor', 'Qh_cor', 'Psurf',\
                 'CO2air', 'CO2air_qc', 'Qg_qc', 'Qle_qc', 'Qh_qc', \
                 'Qle_cor_uc', 'Qh_cor_uc', 'GPP']]

        # Convert units ...

        # hPa -> Pa
        df.loc[:, 'VPD'] *= c.HPA_TO_KPA * c.KPA_TO_PA

        # kPa -> Pa
        df.loc[:, 'Psurf'] *= c.KPA_TO_PA

        # W m-2 to kg m-2 s-1
        lhv = self.latent_heat_vapourisation(df['Tair'])
        df.loc[:, 'ET'] = df['Qle'] / lhv
        df.loc[:, 'ET_cor'] = df['Qle_cor'] / lhv

        # kg m-2 s-1 to mol m-2 s-1
        conv = c.KG_TO_G * c.G_WATER_TO_MOL_WATER
        df.loc[:, 'ET'] *= conv
        df.loc[:, 'ET_cor'] *= conv

        # screen by low u*, i.e. conditions which are often indicative of
        # poorly developed turbulence, after Sanchez et al. 2010, HESS, 14,
        # 1487-1497. Some authors use 0.3 m s-1 (Oliphant et al. 2004) or
        # 0.35 m s-1 (Barr et al. 2006) as a threshold for u*
        df = df[df.ustar >= 0.25]

        # screen for bad data
        df = df[df['Rnet'] > -900.0]

        return (df)

    def get_site_info(self, df_site, fname):

        d = {}
        s = os.path.basename(fname).split(".")[0].split("_")[1].strip()
        d['site'] = s
        d['yrs'] = os.path.basename(fname).split(".")[0].split("_")[5]
        d['lat'] = df_site.loc[df_site.SiteCode == s,'SiteLatitude'].values[0]
        d['lon'] = df_site.loc[df_site.SiteCode == s,'SiteLongitude'].values[0]
        d['pft'] = df_site.loc[df_site.SiteCode == s,\
                               'IGBP_vegetation_short'].values[0]
        d['pft_long'] = df_site.loc[df_site.SiteCode == s,\
                                    'IGBP_vegetation_long'].values[0]

        # remove commas from country tag as it messes out csv output
        name = df_site.loc[df_site.SiteCode == s,'Fullname'].values[0]
        d['name'] = name.replace("," ,"")
        d['country'] = df_site.loc[df_site.SiteCode == s,'Country'].values[0]
        d['elev'] = df_site.loc[df_site.SiteCode == s,'SiteElevation'].values[0]

        d['Vegetation_description'] = df_site.loc[df_site.SiteCode == s,\
                                            'VegetationDescription'].values[0]
        d['soil_type'] = df_site.loc[df_site.SiteCode == s,\
                                            'SoilType'].values[0]
        d['disturbance'] = df_site.loc[df_site.SiteCode == s,\
                                            'Disturbance'].values[0]
        d['crop_description'] = df_site.loc[df_site.SiteCode == s,\
                                            'CropDescription'].values[0]
        d['irrigation'] = df_site.loc[df_site.SiteCode == s,\
                                            'Irrigation'].values[0]

        d['measurement_ht'] = -999.9
        try:
            ht = float(df_site.loc[df_site.SiteCode == s, \
                       'MeasurementHeight'].values[0])
            if ~np.isnan(ht):
                d['measurement_ht'] = ht
        except IndexError:
            pass

        d['tower_ht'] = -999.9
        try:
            ht = float(df_site.loc[df_site.SiteCode == s, \
                       'TowerHeight'].values[0])
            if ~np.isnan(ht):
                d['tower_ht'] = ht
        except IndexError:
            pass

        d['canopy_ht'] = -999.9
        try:
            ht = float(df_site.loc[df_site.SiteCode == s, \
                       'CanopyHeight'].values[0])
            if ~np.isnan(ht):
                d['canopy_ht'] = ht
        except IndexError:
            pass

        return (d)

    def filter_dataframe(self, df, d, hour):
        """
        Filter data only using QA=0 (obs) and QA=1 (good)
        """
        no_G = False

        df_p = df.copy()
        total_length = len(df_p)
        df_y = df_p.groupby(df_p.index.year).sum()

        d['ppt'] = np.mean(df_y.Precip.values)

        # filter daylight hours, good LE data, GPP, CO2
        #
        # If we have no ground heat flux, just use Rn
        if len(df[(df['Qg_qc'] == 0) | (df['Qg_qc'] == 1)]) == 0:
            df = df[(df.index.hour >= 7) &
                    (df.index.hour <= 18) &
                    ( (df['Qle_qc'] == 0) | (df['Qle_qc'] == 1) ) &
                    (df['ET'] > 0.01 / 1000.) & # check in mmol, but units are mol
                    (df['VPD'] > 0.05)]
            no_G = True
        else:
            df = df[(df.index.hour >= 7) &
                    (df.index.hour <= 18) &
                    ( (df['Qle_qc'] == 0) | (df['Qle_qc'] == 1) ) &
                    ( (df['Qg_qc'] == 0) | (df['Qg_qc'] == 1) ) &
                    (df['ET'] > 0.01 / 1000.) & # check in mmol, but units are mol
                    (df['VPD'] > 0.05)]

        # Filter events after rain ...
        idx = df[df.Precip > 0.0].index.tolist()

        if hour:
            # hour gap i.e. Tumba
            bad_dates = []
            for rain_idx in idx:
                bad_dates.append(rain_idx)
                for i in range(24):
                    new_idx = rain_idx + dt.timedelta(minutes=60)
                    bad_dates.append(new_idx)
                    rain_idx = new_idx

            df2 = df.copy()
            df2.loc[:, 'GPP'] *= c.MOL_C_TO_GRAMS_C * c.UMOL_TO_MOL * \
                                 c.SEC_TO_HR


            df = df2
        else:

            # 30 min gap
            bad_dates = []
            for rain_idx in idx:
                bad_dates.append(rain_idx)
                for i in range(48):
                    new_idx = rain_idx + dt.timedelta(minutes=30)
                    bad_dates.append(new_idx)
                    rain_idx = new_idx

            df2 = df.copy()
            df2.loc[:, 'GPP'] *= c.MOL_C_TO_GRAMS_C * c.UMOL_TO_MOL * \
                                c.SEC_TO_HLFHR

            df = df2

        # There will be duplicate dates most likely so remove these.
        bad_dates = np.unique(bad_dates)

        # remove rain days...
        df = df[~df.index.isin(bad_dates)]

        return (df, d, no_G)

    def penman_montieth_wrapper(self, d, df, no_G):

        error = False

        if no_G:
            G = None
        elif len(df.Qg) > 0:
            G = df['Qg'] # W m-2
        else:
            G = None

        PM = PenmanMonteith(use_ustar=True)
        df['gs_est']  = PM.invert_penman(df['VPD'], df['Wind'], df['Rnet'],
                                         df['Tair'], df['Psurf'], df['ET'],
                                         ustar=df["ustar"], G=G)

        (omega, ga, gs) = PM.calc_decoupling_coefficent(df['Wind'], df['Tair'],
                                                        df['Psurf'],
                                                        df["gs_est"],
                                                        ustar=df["ustar"])

        ga = np.where(np.logical_and(omega >= 0.0, omega <= 1.0), ga, np.nan)
        gs = np.where(np.logical_and(omega >= 0.0, omega <= 1.0), gs, np.nan)
        omega = np.where(np.logical_and(omega >= 0.0, omega <= 1.0),
                         omega, np.nan)

        df["ga"] = ga
        df["gs"] = gs
        df["omega"] = omega


        # screen for bad data, or data I've set to bad
        df = df[(df['gs_est'] > 0.0) & (np.isnan(df['gs_est']) == False)]

        # Filter extreme omega are ridiculous
        if df['omega'].count() != 0:    # count non Nans
            extreme = np.nanmean(df['omega']) + (3.0 * np.nanstd(df['omega']))
            df = df[df['omega'] < extreme]
        else:
            error = True # all bad data

        # Filter extreme gs values that are ridiculous
        if df['gs_est'].count() != 0:    # count non Nans
            df = df[df['gs_est'] < 1.5 * df['ga']]
        else:
            error = True # all bad data

        return (df, error)

    def make_plot(self, d, df):

        fig = plt.figure(figsize=(9,6))
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(wspace=0.05)
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = "sans-serif"
        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['font.size'] = 14
        plt.rcParams['legend.fontsize'] = 14
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14

        ax = fig.add_subplot(111)

        ax.plot(df.index, df.omega_mu, lw=2, color='blue')
        ax.fill_between(df.index, df.omega_mu+df.omega_sigma,
                        df.omega_mu-df.omega_sigma,
                        facecolor='blue', alpha=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("$\Omega$ (-)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        fig.savefig(os.path.join(self.oplots, "%s_omega.pdf" % (d['site'])),
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        return df, d

    def write_row(self, df_out, d):

        row = pd.Series([d['site'], d['name'], d['country'], d['yr'],
                         d['lat'], d['lon'], d['pft'], d['num_pts'],
                         d['EBR'], d['omega'], d['wind'], d['ga'],
                         d['gs'], d['tower_ht'], d['canopy_ht']],
                         index=self.out_cols)


        result = df_out.append(row, ignore_index=True)
        return result


    def latent_heat_vapourisation(self, tair):
        """
        Latent heat of vapourisation is approximated by a linear func of air
        temp (J kg-1)

        Reference:
        ----------
        * Stull, B., 1988: An Introduction to Boundary Layer Meteorology
          Boundary Conditions, pg 279.
        """
        return (2.501 - 0.00237 * tair) * 1E06

if __name__ == "__main__":
    """
    F = FitOmega(fdir="/Users/mdekauwe/Desktop/test_hrly",
                 #fdir="data/raw_data/fluxnet2015_tier_1",
                 adir="data/raw_data/anna_meta",
                 ofdir="data/processed/",
                 co2dir="data/raw_data/global_CO2_data/",
                 site_fname="site_metadata.csv",
                 global_co2_fname="Global_CO2_mean_NOAA.csv",
                 ofname="omega_fluxnet_PM.csv")
    F.main(hour=True)
    """

    #"""
    F = FitOmega(fdir="/Users/mdekauwe/Desktop/test_hfhrly",
                 #fdir="data/raw_data/fluxnet2015_tier_1",
                 adir="data/raw_data/anna_meta",
                 ofdir="data/processed/",
                 co2dir="data/raw_data/global_CO2_data/",
                 site_fname="site_metadata.csv",
                 global_co2_fname="Global_CO2_mean_NOAA.csv",
                 ofname="omega_fluxnet_PM.csv")
    F.main(hour=False)
    #"""
