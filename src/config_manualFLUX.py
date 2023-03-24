#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mika Korkiakoski (mika.korkiakoski@fmi.fi)

Config file for the ManualChamberFlux program.
"""
#===============================================================================================================#

#This is the name of the results file
#The result file will have the following columns: Starting time of the measurement, source chamber, length of the
#linear and exponential fits, headspace temperature, air pressure, linear and exponential fluxes of the gases,
#and possible extra columns inserted at the end of the times file.
#The selected file name has to be in hyphens
site_name='results_example'

#===============================================================================================================#

#Directory where fluxes are saved. A folder called 'plots' is created
#into this folder, if plots are made.
#The location needs to be in hyphens
#The folders can be separated by "/" (Linux and windows) or by "\" (Windows)
#But if the folders are separated by "\", one needs to add "r" before the first hyphen like this:
#result_loc=r'C:\m\flux_example\results\'
result_loc='/home/m/flux_example/results/'

#===============================================================================================================#

#The DIRECTORY where the daily data is located
#The data file should be comma delimited (.csv).
#Compressed (zip, tar.gz) csv files should also work.
#The location has to be in hyphens

#MANDATORY COLUMNS:
    #Datetime 

#Include the gas mixing ratio columns below of which you want to calculate the
#fluxes (these need to be DRY concentrations in PPM)
#You can also give headspace temperature, air pressure, RH and PAR in this file. 
#If headspace temperature is given here AND also in the TIMES file, then the T
#from the TIMES file will be used in the flux calculation.
    #CO2_dry (ppm)
    #CH4_dry (ppm)
    #N2O_dry (ppm)
    #CO_dry (ppm)
    #Headspace Temperature (C)
    #Air pressure (hpa)
    #RH (%)
    #PAR (umol m-2 s-1)

#Rest of the columns in the datafile will be ignored

#The folders can be separated by "/" (Linux and windows) or by "\" (Windows)
#But if the folders are separated by "\", one needs to add "r" before the first hyphen like this:
#result_loc=r'C:\m\flux_example\data_example.csv'
data_loc='/home/m/flux_example/data/data_example.csv'

#===============================================================================================================#

#This is the location of the "times" file (has to be an excel file ".xlsx")
#MANDATORY COLUMNS (if not explicitly stated, the columns have to include 
#sensible data, otherwise the program won't work):
    #Date
    #Start time
    #End time
    #Source
    #System height (m) (Chamber height+collar height or chamber height-chamber sinking into the soil)
    #Headspace temperature (C) (Leave empty if the T data is given in the chamber data file.)
    #Air pressure (hPa) (Not absolutely necessary, but the column has to exist. Leave empty if no data.)

#If the data in the air pressure column is empty, the program will search for
#air pressure data from the chamber data file, if no air pressure column is
#found from these two sources, the flux is calculated without taking into
#account the air pressure.
#If headspace temperature is given here AND also in the DATA file, then
#the T from the TIMES file will be used in the flux calculation.

#Any other columns included in the times file will be added to the result
#file. So, include environmental values like ST, WTD, SM measured during the
#closure here if you want them saved
#with the respective fluxes.
#The location needs to be in hyphens.
#The folders can be separated by "/" (Linux and windows) or by "\" (Windows)
#But if the folders are separated by "\", one needs to add "r" before the first hyphen like this:
#result_loc=r'C:\m\flux_example\times_example.csv'
times_loc='/home/m/flux_example/times_example.xlsx'

#===============================================================================================================#

#Time format of the DATA file
#Example: year-month-day hour:minutes:seconds.decimals = '%Y-%m-%d %H:%M:%S.%f'
#The time format needs to be in hyphens
time_format_data='%Y-%m-%d %H:%M:%S.%f'

#===============================================================================================================#

#Time format of the TIMES file
#Example: year-month-day hour:minutes:seconds = '%Y-%m-%d %H:%M:%S'
#Note, if the user wants to interactively select the starting and ending times
#by using this program, it is recommended to set the starting time earlier and
#the ending time later than expected in the TIMES file to make sure that the
#whole closure is shown in the plot, which is used to select the exact times.
#The time format needs to be in hyphens
time_format_times='%Y-%m-%d %H:%M:%S'

#===============================================================================================================#

#Select starting and ending times of the measurements interactively
#If True, then starting and ending times in the TIMES file are used to
#plot the timeseries of the gases for each closure. For each plot, the
#user has to left click the plot to select start and end times.
#Selections can be cancelled by right clicking the plot.
#This program assumes that the first click is the starting time and the second
#click is the ending time and the third click closes the plot.
#The selections are marked with red cross.
#It is recommended to click between the measurement points and not the points
#themselves to make sure the selected interval is what is wanted (all the points
#located between the red crosses are included for the fitting).
#If the parameter is False, then the starting and ending times in the
#times_file are used to make the fits and calculate the fluxes.
interactive_times=True

#===============================================================================================================#

#Replace the starting and ending times in the times file with the
#interactively selected times (True). The new TIMES file is saved
#in the folder indicated by the results_loc parameter below.
#The old times file is not replaced! The new file will have a suffix
#"_new" to avoid overwriting.
#If False, the interactively selected times will not be saved, but
#they will still be used in the flux calculation.
interactive_save=True

#===============================================================================================================#

#How many seconds of data will be discarded from the start of the measurement.
#This will be applied also when selecting the starting times interactively.
#If set to 0, the starting time will be exactly the one selected interactively 
#or the one given in the times-file.
skip_start=0

#===============================================================================================================#

#The minimum and the maximum allowed length of the fit in seconds.
#If the fit is longer than fit_maximum, datapoints from the end will
#be removed until the length is less than fit_maximum
#If the fit is shorter than fit_minimum, the closure will be ignored
fit_minimum=60
fit_maximum=330

#===============================================================================================================#

#Use time averaging in the mixing ratio data.
#In seconds. Set to zero if no time averaging is wanted.
time_avg=5

#===============================================================================================================#

#Plot closures? (True or False)
#If True, plot the mixing ratio time series of all the selected gases and
#the fits made into them and save the figures.
plot_closures=True

#===============================================================================================================#

#Save also the fitting statistics (True) or just the fluxes and the optional
#columns in the data file (False)
#The statistics will be calculated for all the gases and for both linear and exponential fluxes.
#The calculated statistics are: standard deviation of the slope of the fit (e.g. std_CO2_lin), normalized root
#mean square error of the fit (NRMSE), adjusted R2 value of the fit (R2adj), Akaike information criterion of the
#fit (AIC), autocorrelation of 1 point lag (autocorr)
save_statistics=True

#===============================================================================================================#
