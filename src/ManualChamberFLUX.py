#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, bisect, warnings, time, sys, ntpath
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from alive_progress import alive_bar

#Import the config file
try:
    import config_manualFLUX as config
except (SyntaxError, NameError):
    sys.exit("One of the required parameters in the config file is empty or has"
             " an incorrect value or the config file is completely messed up!"
             " Check that: none of the parameters are empty, folder/file paths"
             " have hyphens around them, and parameters that should have a True"
             " or False value are written capitalized. Closing the program.")

VERSION='v1.3.2 MAR 2023'
APPNAME='ManualChamberFlUX'

#Ignore warnings. I know what I'm doing.
warnings.simplefilter('ignore')
    
#Checks if the directory exists and creates one if it doesn't
def check_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
    return

#Check that the given parameter values make sense
def check_config():
    #Check that the site_name, locations for the data, times and result files exist
    
    #Create the folder for the results
    check_create_dir(config.result_loc)
    
    #Check the result_loc parameter, if the final character is not "/" or "\", add it
    if config.result_loc[-1]!='/':
        if config.result_loc[-1]!='\\':
            #Check the first character, if it is "/", assume linux system and add "/"
            if config.result_loc[0]=='/':
                config.result_loc=config.result_loc+'/'
            #Otherwise, it is a windows system, add "\"
            else:
                config.result_loc=config.result_loc+'\\'
    
    #If closure plots are made, create folder for them
    if config.plot_closures==True:
        check_create_dir(config.result_loc+'plots')
    
    if type(config.fit_minimum)!=int and type(config.fit_minimum)!=float:
        sys.exit("Config-file: fit_minimum has to have a numerical value! Closing the program.")
        
    if type(config.fit_maximum)!=int and type(config.fit_maximum)!=float:
        sys.exit("Config-file: fit_maximum has to have a numerical value! Closing the program.")
    
    #Check that the minimum fit length is smaller than the maximum fit length
    if config.fit_minimum>config.fit_maximum:
        sys.exit("Config-file: fit_minimum can't be larger than fit_maximum! Closing the program.")
        
    #Check that save_statistics parameter has either True of False value
    if config.save_statistics!=True and config.save_statistics!=False:
        sys.exit("Config-file: save_statistic has to be either True or False! Closing the program.")
    
    #Check that the plot_closures paramter has either True or False value
    if config.plot_closures!=True and config.plot_closures!=False:
        sys.exit("Config-file: plot_closures has to be either True or False! Closing the program.")
    
    #Check that the plot_closures paramter has either True or False value
    if config.interactive_times!=True and config.interactive_times!=False:
        sys.exit("Config-file: interactive_times has to be either True or False! Closing the program.")
        
    #Check that the plot_closures paramter has either True or False value
    if config.interactive_save!=True and config.interactive_save!=False:
        sys.exit("Config-file: interactive_save has to be either True or False! Closing the program.")
    
    #Check that the time_avg parameter is zero or larger and has a numerical value.
    try:
        if config.time_avg<0:
            sys.exit("Config-file: time_avg has to be zero or larger! Closing the program.")
    except TypeError:
        sys.exit("Config-file: time_avg has to have a numerical value! Closing the program.")
    
    #Check that the skip_start parameter is zero or larger and has a numerical value
    try:
        if config.skip_start<0:
            sys.exit("Config-file: skip_start cannot be smaller than zero! Closing the program.")
    except TypeError:
        sys.exit("Config-file: skip_start has to have a numerical value! Closing the program.")
        
    print("Config file checked.")
    
    return 

#Load the chamber data file
def load_data_file():
    data=pd.read_csv(config.data_loc,sep=',',index_col=0) #Load chamber data
        
    data.index=pd.to_datetime(data.index,format=config.time_format_data) #Timestamps to datetime
    data=data.sort_index() #Sort data by index
    
    #Check that at least one gas column exists
    test_cols=data.columns.isin(['CO2_dry (ppm)','CH4_dry (ppm)','N2O_dry (ppm)','CO_dry (ppm)'])
    if not True in test_cols:
        sys.exit('The datafile does not have any gas columns in it! Closing the program.')
    
    return data

def load_times_file():
    #Load the times file
    data_times=pd.read_excel(config.times_loc,dtype={'Date':str,'Start time':str,'End time':str})
    
    #Check that the mandatory columns exist
    test_cols=data_times.columns.isin(['Date','Start time','End time','Source',
                                       'System height (m)','Headspace temperature (C)',
                                       'Air pressure (hPa)'])
    test_cols=test_cols[test_cols==True]
    if len(test_cols)<7:
        sys.exit("One of the following columns is missing from the times file: "
                 "'Date','Start time','End time','Source','System height (m)',"
                 "'Headspace temperature (C)', 'Air pressure (hPa)'! Closing the program.")
    
    #Times to datetime
    start_times=pd.Series()
    end_times=pd.Series()
    try:
        for k in range(len(data_times)):
            start_times=start_times.append(
                pd.Series(index=[pd.to_datetime(data_times.Date[k][0:10]+' '+data_times['Start time'][k],
                                                format=config.time_format_times)]))
            end_times=end_times.append(
                pd.Series(index=[pd.to_datetime(data_times.Date[k][0:10]+' '+data_times['End time'][k],
                                                format=config.time_format_times)]))
    except:
        sys.exit('There is an invalid value in either "Start time" or "End time" column in the times file!')
    
    data_times['Start time']=start_times.index
    data_times['End time']=end_times.index
    
    #Check that the start and end times in the times file make sense
    #If the measurements spans over midnight, adjust the end time accordingly
    st_et_diff=data_times['End time']-data_times['Start time']
    if len(st_et_diff[st_et_diff<dt.timedelta(0)])>0:
        #Get rows where the start time is bigger than the end time
        st_et_diff=st_et_diff[st_et_diff<-dt.timedelta(seconds=1)].index
        
        #Check the rows with a bigger starting time than an end time if they occur
        #during midnight, if they do, add 1 day to the end time and check that the
        #measurement is shorter than 2 hours
        bad_rows=np.array([])
        for row in st_et_diff:
            cur_end=data_times.iloc[row]['End time'] #Old end time
            new_end=cur_end+dt.timedelta(days=1) #New end time (+1 day)
            cur_length=new_end-data_times.iloc[row]['Start time']
            cur_length=cur_length.total_seconds() #Length of the measurement in seconds
            if cur_length<7200: #If the measurement is shorter than 2 hours, it is ok
                data_times.at[row,'End time']=new_end
            else: #Otherwise there is a typo in the starting or ending time
                bad_rows=np.append(bad_rows,row)
        
        if len(bad_rows)>0:
            print("There are rows where the end time is smaller than the start time.")
            print("Check the following rows in the times file:")
            for row in bad_rows:
                print(row)
            sys.exit("Closing the program.")
    
    #Check that air pressure values make sense. This is mainly for checking that the units are ok.
    bad_P=data_times[(data_times['Air pressure (hPa)']<850) | (data_times['Air pressure (hPa)']>1200)].index
    if len(bad_P)>0:
        print("There are unrealistic air pressures in the times file on the following rows:")
        for row in bad_P:
            print(row)
        sys.exit("Closing the program.")
        
    #Check that headspace temperature values make sense. This is mainly for checking that the units are ok.
    bad_T=data_times[(data_times['Headspace temperature (C)']<-40) | (data_times['Headspace temperature (C)']>50)].index
    if len(bad_T)>0:
        print("There are unrealistic headspace temperatures in the times file on the following rows:")
        for row in bad_T:
            print(row)
        sys.exit("Closing the program.")
        
    return data_times

#Check which gas columns exist in the data file
#The column names include "(ppm)", remove it to make code easier to write
def check_columns(data):
    if ('CO2_dry (ppm)' or 'CO2_dry') in data.columns:
        CO2_dry=True
        if 'CO2_dry (ppm)' in data.columns:
            data=data.rename({'CO2_dry (ppm)':'CO2_dry'}, axis=1)
    else:
        CO2_dry=False
        
    if ('CH4_dry (ppm)' or 'CH4_dry') in data.columns:
        CH4_dry=True
        if 'CH4_dry (ppm)' in data.columns:
            data=data.rename({'CH4_dry (ppm)':'CH4_dry'}, axis=1)
    else:
        CH4_dry=False
        
    if ('N2O_dry (ppm)' or 'N2O_dry') in data.columns:
        N2O_dry=True
        if 'N2O_dry (ppm)' in data.columns:
            data=data.rename({'N2O_dry (ppm)':'N2O_dry'}, axis=1)
    else:
        N2O_dry=False
        
    if ('CO_dry (ppm)' or 'CO_dry') in data.columns:
        CO_dry=True
        if 'CO_dry (ppm)' in data.columns:
            data=data.rename({'CO_dry (ppm)':'CO_dry'}, axis=1)
    else:
        CO_dry=False
        
    cols_to_calc=pd.DataFrame([np.array([CO2_dry, CH4_dry, N2O_dry, CO_dry])],
                              columns=['CO2_dry', 'CH4_dry', 'N2O_dry', 'CO_dry'],index=[0])
    
    return cols_to_calc, data

#Initialize the result dataframe. 
#Create columns to the result file for gases which were found from the datafile in check_columns
def init_results(gas):
    if gas=='co2':
        results=pd.DataFrame(columns=['Lin_Flux_CO2 [mg CO2 m-2 h-1]',
                                      'Exp_Flux_CO2 [mg CO2 m-2 h-1]','Start_CO2 [ppm]','End_CO2 [ppm]'])
        if config.save_statistics==True:
            temp=pd.DataFrame(columns=['std_CO2_lin', 'std_CO2_exp', 'NRMSE_CO2_lin',
                                       'NRMSE_CO2_exp','R2adj_CO2_lin','R2adj_CO2_exp',
                                       'AIC_CO2_lin','AIC_CO2_exp','Autocorr_CO2_lin','Autocorr_CO2_exp'])
            results=results.join(temp)
            
    if gas=='ch4':
        results=pd.DataFrame(columns=['Lin_Flux_CH4 [ug CH4 m-2 h-1]',
                                      'Exp_Flux_CH4 [ug CH4 m-2 h-1]','Start_CH4 [ppm]','End_CH4 [ppm]'])
        if config.save_statistics==True:
            temp=pd.DataFrame(columns=['std_CH4_lin', 'std_CH4_exp', 'NRMSE_CH4_lin',
                                       'NRMSE_CH4_exp','R2adj_CH4_lin','R2adj_CH4_exp',
                                       'AIC_CH4_lin','AIC_CH4_exp','Autocorr_CH4_lin','Autocorr_CH4_exp'])
            results=results.join(temp)

    if gas=='n2o':
        results=pd.DataFrame(columns=['Lin_Flux_N2O [ug N2O m-2 h-1]',
                                      'Exp_Flux_N2O [ug N2O m-2 h-1]','Start_N2O [ppm]','End_N2O [ppm]'])
        if config.save_statistics==True:
            temp=pd.DataFrame(columns=['std_N2O_lin', 'std_N2O_exp', 'NRMSE_N2O_lin',
                                       'NRMSE_N2O_exp','R2adj_N2O_lin','R2adj_N2O_exp',
                                       'AIC_N2O_lin','AIC_N2O_exp','Autocorr_N2O_lin','Autocorr_N2O_exp'])
            results=results.join(temp)
            
    if gas=='co':
        results=pd.DataFrame(columns=['Lin_Flux_CO [ug CO m-2 h-1]',
                                      'Exp_Flux_CO [ug CO m-2 h-1]','Start_CO [ppm]','End_CO [ppm]'])
        if config.save_statistics==True:
            temp=pd.DataFrame(columns=['std_CO_lin', 'std_CO_exp', 'NRMSE_CO_lin',
                                       'NRMSE_CO_exp','R2adj_CO_lin','R2adj_CO_exp',
                                       'AIC_CO_lin','AIC_CO_exp','Autocorr_CO_lin','Autocorr_CO_exp'])
            results=results.join(temp)
            
    return results

#Initialize the result array for the optional columns in the times file
def init_results_rest(times_file):
    #Drop the mandatory columns
    results_rest=times_file.drop(['Date','Start time','End time','Source','System height (m)',
                                  'Headspace temperature (C)','Air pressure (hPa)'],axis=1)
    
    #Add an extra columns
    results_rest['Fit length [s]']=''
      
    return results_rest

#Time averaging for the gas mixing ratio data
def data_time_avg(data):
    secs=str(config.time_avg)+'S' #formatting for the resample function
    data=data.apply(pd.to_numeric,errors='coerce') #Transform all values to numeric values
    data=data.resample(secs).mean() #Do the time averaging
    data=data.dropna(how='all') #Drop rows if all values are nan
    
    return data

#Get the current closure data from the data file
def start_end_times_source_name(data_file, times_file, ind):
    skip_flag=0 #Used to flag the closure (0=ok, 1=skip)
    
    cur_source=times_file['Source'][ind] #Get the current source
    
    #Start and end index of the current closure
    start=data_file.index.searchsorted(times_file['Start time'][ind])
    try:
        end=data_file.index.searchsorted(times_file['End time'][ind])+1
    except IndexError: #If the system crashed before a closure ends
        end=len(data_file)-1
    
    data_closure=data_file.iloc[start:end] #Get the current closure
    
    #Check that the closure is shorter than two hours because no one does chamber
    #measurements that take longer than that, if they do, they are doing something wrong
    #If it is not, skip it
    if (data_closure.index[-1]-data_closure.index[0]).total_seconds()>120*60:
        skip_flag=1
        return np.nan, cur_source, times_file['Start time'][ind], skip_flag
        
    try:
        stime=data_closure.index[0] #Starting time
    except IndexError: #If the current closure data is empty, move to the next one
        skip_flag=1 #Skip this closure
        return np.nan, cur_source, times_file['Start time'][ind], skip_flag
    
    return data_closure, cur_source, stime, skip_flag

#Limit the current closure data according to minimum and maximum fit parameters
def limit_closure_time(data_closure, stime):
    skip_flag=0 #Used to flag the closure (0=ok, 1=skip)
    
    a=data_closure.index-stime #Time from the start of the closure for each datapoint
    x=np.array([])
    for k in range(len(a)): #Convert to seconds from the start of the closure
        x=np.append(x,a[k].total_seconds())
        
    #If closure is shorter than fit_minimum, ignore it
    if x[len(x)-1]<config.fit_minimum:
        skip_flag=1
        return np.nan, np.nan, skip_flag
    
    #If the closure duration is longer than the allowed (fit_maximum) remove
    #data from the end to reach the maximum allowed length
    time_dif=(data_closure.index-data_closure.index[0]).total_seconds()
    data_closure['Fit length [s]']=time_dif #Save seconds from the beginning
    end=bisect.bisect(time_dif,config.fit_maximum)
    data_closure=data_closure[0:end]
    
    if len(data_closure)<3:
        sys.exit('The closure starting on '+str(stime)+' includes less than 3 measurement points.'
                 'Try increasing fit_maximum parameter in the config file.')
    
    stime=data_closure.index[0] #Starting time of the closure
        
    return data_closure, stime, skip_flag

#Calculate means of ancillary variables given in the data file
def datafile_means(anc_means,data_closure):
    #Calculate the mean for par data
    try:
        anc_means['PAR [umol m-2 s-1]']=round(data_closure['PAR (umol m-2 s-1)'].mean())
        
    except: #if no par data, ignore it
        pass
    
    #Calculate the mean for RH data
    try:
        anc_means['RH [%]']=round(data_closure['RH (%)'].mean())
    except: #if no RH data, ignore it
        pass

    return anc_means

#2nd order polynomial    
def func_2nd(x, a, b, c):
	return a*x**2+b*x+c
    
#Exponential function, which will be fitted into the data
def func_exp(x, a, b, c):
	return a+b*np.exp(c*x)

#Linear fit    
def func_lin(x,a,b):
    return a+b*x

#17th order taylor power series for 2nd order equation. resulting coefficients
#will be used as initial values for the exponential fit.
def func_17th(x, c, b, a):
    return a+b*x+c*x**2+((2**2*c**2)/(6*b))*x**3+((2**3*c**3)/(24*b**2))*x**4 \
    +(2**4*c**4)/(120*b**3)*x**5+(2**5*c**5)/(720*b**4)*x**6+(2**6*c**6) \
    /(5040*b**5)*x**7+(2**7*c**7)/(40320*b**6)*x**8+(2**8*c**8)/(362880*b**7) \
    *x**9+(2**9*c**9)/(3628800*b**8)*x**10+(2**10*c**10)/(39916800*b**9)*x**11 \
    +(2**11*c**11)/(479001600*b**10)*x**12+(2**12*c**12)/(6227020800*b**11) \
    *x**13+(2**13*c**13)/(87178291200*b**12)*x**14+(2**14*c**14) \
    /(1307674368000*b**13)*x**15+(2**15*c**15)/(20922789888000*b**14)*x**16 \
    +(2**16*c**16)/(355687428096000*b**15)*x**17

#Exponential fitting
def exp_fit(x, y, coefs_2nd):
    #make 17th order fit by using 2nd order coefficients as initial values
    coefs_17=curve_fit(func_17th, x.values, y.values, coefs_2nd)
        
    #find the initial values for the exponential fit by using the coefficients
    #acquired from the 17th order fit
    init_exp=np.array([coefs_17[0][2]-(coefs_17[0][1]**2)/(2*coefs_17[0][0]), \
    (coefs_17[0][1]**2)/(2*coefs_17[0][0]), 2*coefs_17[0][0]/coefs_17[0][1]])
    
    #make exp fit
    coefs_exp1=curve_fit(func_exp, x.values, y.values, init_exp)    
    
    coefs_exp=coefs_exp1[0]
    errcoefs_exp=np.array([np.sqrt(coefs_exp1[1][0,0]),coefs_exp1[1][1,1]-2*coefs_exp1[1][1,2],\
    coefs_exp1[1][2,2]-2*coefs_exp1[1][1,2]])
    err_exp=np.abs(coefs_exp[1]*coefs_exp[2]*\
    np.sqrt((errcoefs_exp[1]/coefs_exp[1])**2+(errcoefs_exp[2]/coefs_exp[2])**2))
    
    #calculate SSE
    sse=np.array([np.mean((y.values-func_exp(x.values, *coefs_exp))**2)*len(x)])
    
    #calculate rmse and nrmse. assuming ydata is in ppm
    goods_exp=rmse_nrmse_calc(sse, y.values)
    
    return coefs_exp, goods_exp, err_exp

#SSE to RMSE and NRMSE calculator    
def rmse_nrmse_calc(sse, data):
    n=len(data)
    rmse=np.sqrt(sse/n)
    nrmse=rmse/(max(data)-min(data))
    results=np.array([rmse, nrmse])
    
    return results
    
#Calculates R^2 for data fits
def r_squared(data, yfit):   
    r,_=pearsonr(data, yfit) #calculate the correlation coefficient   
    rsq=r**2
    rsq=round(rsq, 5)
    return rsq

#Calculates the autocorrelation (one point lag) of the data
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    lag1=result[1]
    return result, lag1

#Calculates adjusted R2 and AIC for the fits
def stats_calc(n, sse_s, rsq_s):
    p_lin=2
    p_exp=3
    
    #R2_adj
    R2_lin=1-(((1-rsq_s[0])*(n-1))/(n-p_lin-1))
    R2_exp=1-(((1-rsq_s[1])*(n-1))/(n-p_exp-1))
    
    #AIC
    try:
        AIC_lin=n*np.log(sse_s[0]/n)+2*p_lin+((2.0*p_lin*(p_lin+1))/(n-p_lin-1.0))
    except ZeroDivisionError:
        AIC_lin=np.nan

    try:
        AIC_exp=n*np.log(sse_s[1]/n)+2*p_exp+((2.0*p_exp*(p_exp+1))/(n-p_exp-1.0))
    except ZeroDivisionError: #If too few measurement points, set AIC_exp to nan
        AIC_exp=np.nan
    
    #combine
    r2_aic=pd.DataFrame({'r2_lin':R2_lin,'r2_exp':R2_exp,'aic_lin':AIC_lin,'aic_exp':AIC_exp},index=[0])

    return r2_aic

#The main program handling the fitting and statistical parameters of a closure
def closure_fitting(data_closure, gas):
    #put x and y data into same array
    fit_data=pd.DataFrame({'x':data_closure['Fit length [s]'],'ydata':data_closure[gas]})
    
    #remove nans, but use this only for fitting. modelling will be made for all points
    fit_data_na=fit_data.dropna(how='any')
    
    #make linear fit
    coefs_lin=np.polyfit(fit_data_na.x, fit_data_na.ydata, 1, full=True)
    
    #calculate uncertainty of the slope
    try:
        _,cov=np.polyfit(fit_data_na.x, fit_data_na.ydata, 1, cov=True)
    except ValueError:
        coef=np.array([np.nan,np.nan,np.nan])
        coef=np.vstack([coef,coef])
        goods=np.array([np.nan,np.nan])
        goods=np.vstack([goods,goods])
        stdr=pd.DataFrame([np.nan,np.nan],index=['lin','exp'])
        acorrs=pd.DataFrame([np.nan,np.nan],index=['lin','exp'])
        exp_failed=True
        yfits=sse_s=rsq_s=np.array([np.nan,np.nan])
        err_slopes=pd.DataFrame([[np.nan,np.nan], [np.nan,np.nan]],index=['lin','exp'])
        return yfits, coef, goods, exp_failed, sse_s, rsq_s, err_slopes,stdr,acorrs
        
    err_lin=np.sqrt(np.diag(cov)) #std
    
    #model y-values from the linear fit
    data_lin=np.polyval(coefs_lin[0], fit_data.x)
    
    #calculate rmse and nrmse
    sse_lin=coefs_lin[1]
    goods_lin=rmse_nrmse_calc(sse_lin, data_closure[gas])
    
    #if goods_lin array is empty for some weird reason (SSE missing), insert nans
    try: #use try-except because if the array exists, the if sentence is not valid
        if not goods_lin:
            goods_lin=np.array([np.nan,np.nan])
    except ValueError:
        pass
    
    #make 2nd order fit
    try:
        coefs_2nd=np.polyfit(fit_data_na.x, fit_data_na.ydata, 2, full=True)
        sse_exp=coefs_2nd[1]
    except ValueError:
        coefs_2nd=np.array([np.nan,np.nan])
        sse_exp=np.nan
        
    #make the exponential fit
    exp_failed=False
    
    try:
        [coefs_exp, goods_exp, err_exp]=exp_fit(fit_data_na.x, fit_data_na.ydata, coefs_2nd[0])
        
    except (RuntimeError, TypeError):
        exp_failed=True
        coefs_exp=np.array([float('nan'), float('nan'), float('nan')])
        goods_exp=np.array([float('nan'), float('nan')])
        err_exp=np.nan
        data_exp=np.empty((len(data_lin),))
        data_exp[:]=np.nan
        
    #model the y-values from the fitting coefficients
    data_exp=coefs_exp[0]+coefs_exp[1]*np.exp(coefs_exp[2]*fit_data.x)
    
    #append linear coefficient lists with 0 to make merging with exp coefficient list possible
    coefs_lin=coefs_lin[0]
    coefs_lin=np.append(coefs_lin, 0)
    
    #merge all coefficients into one array
    coef=np.vstack([coefs_lin, coefs_exp])
    
    #merge all rmse & nrmse values into one array
    goods=np.vstack([goods_lin.T, goods_exp.T])
    
    #merge lin and exp fit values into one array
    yfits=np.vstack([data_lin, data_exp])
    yfits=yfits.T
    
    #calculate r2
    try:
        fit_rsq=fit_data.ydata[~np.isnan(fit_data.ydata)].values
        data_lin_rsq=data_lin[~np.isnan(fit_data.ydata).values]
        data_exp_rsq=data_exp[~np.isnan(fit_data.ydata).values]
        rsq_lin=r_squared(fit_rsq, data_lin_rsq)
        rsq_exp=r_squared(fit_rsq, data_exp_rsq)
    except ValueError:
        rsq_lin=rsq_exp=np.nan
    
    #merge sse
    sse_s=np.array([sse_lin,sse_exp])
    
    #merge rsq
    rsq_s=np.array([rsq_lin,rsq_exp])
    
    #merge std of the slopes
    err_slopes=pd.DataFrame([err_lin[0], err_exp],index=['lin','exp'])
    
    #autocorrelations
    _,lag1_lin=estimated_autocorrelation(fit_rsq-data_lin_rsq)
    _,lag1_exp=estimated_autocorrelation(fit_rsq-data_exp_rsq)
    acorrs=pd.DataFrame([lag1_lin,lag1_exp],index=['lin','exp'])
    
    return yfits, coef, goods, exp_failed, sse_s, rsq_s, err_slopes, acorrs

#Makes the closure plots
def flux_plotter(gas_col, data_closure, yfits_gas, exp_failed_gas): 
    #fix the fonts in plots when using TEX formatting, remove the italization
    mpl.rcParams['mathtext.default'] = 'regular'
    
    #plotplot
    fig=plt.figure(num=None, figsize=(10, 7), dpi=100) 
    ax=fig.add_subplot(111)
    #plot the data points
    org=ax.plot(data_closure['Fit length [s]'], data_closure[gas_col], 'o', color='k')
    #plot the lin fit
    lin,=ax.plot(data_closure['Fit length [s]'], yfits_gas[:,0], '-', color='r', linewidth=2) 
    
    #Remove the annoying exponent format from the yaxislabels that python likes to use sometimes
    ax.get_yaxis().get_major_formatter().set_useOffset(False) 
    ax.set_xlabel('Seconds from the start', fontsize='16') #xlabel
    
    #Set the ylabel for the gas
    if gas_col=='CO2_dry' or gas_col=='CO2_licor':
        ax.set_ylabel(r'$CO_2$ mixing ratio (ppm)', fontsize='16')
    if gas_col=='CH4_dry':
        ax.set_ylabel(r'$CH_4$ mixing ratio (ppm)', fontsize='16')
    if gas_col=='N2O_dry':
        ax.set_ylabel(r'$N_2O$ mixing ratio (ppm)', fontsize='16')
    if gas_col=='CO_dry':
        ax.set_ylabel(r'$CO$ mixing ratio (ppm)', fontsize='16')
        
    if exp_failed_gas==True: #If exp fit failed, modify the legend accordingly
        ax.legend([org, lin], ['Data', 'Lin fit'], numpoints=1, loc=0, title='Exp fit failed')
        
    else: #If exp fit succeded, plot also exp fit and modify the legend accordingly
        exp,=ax.plot(data_closure['Fit length [s]'], yfits_gas[:,1], '-', color='g', linewidth=2)
        ax.legend(['Data', 'Linear fit','Exponential fit'], numpoints=1, loc=0)   
    
    y_formatter=mpl.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    
    #Fine-tune x and y limits
    ax.set_ylim(np.min(data_closure[gas_col])-0.001, np.max(data_closure[gas_col])+0.001)
    ax.set_xlim([0,data_closure['Fit length [s]'][len(data_closure)-1]])

    return fig    

#Saves the plots
def fig_save(fig, pic_time, fig_path, source):
    start=dt.datetime.strftime(pic_time[0],'%H_%M_%S') #get starting time
    date=dt.datetime.strftime(pic_time[0],'%Y-%m-%d') #get the date
    folder=fig_path+'/'+date+'/' #folder where to save the figure
    check_create_dir(folder) #create the folder if it does not exist
    fig.savefig(folder+str(source)+'_'+start+'.png', transparent=False, dpi=100) #Save the figure
    plt.close(fig) #Close the figure
    
    return    

#Make fits to the gas data and make closure plots
def gas_fitting(data_closure,gas,cur_source):
    try:
        if gas=='co2':
           if np.isfinite(np.nanmean(data_closure['CO2_dry']))==True:
               gas_col='CO2_dry' #If CO2_dry data exists, use it
           else: #If no licor CO2 data, set everything to nan
               coef_gas=np.array([np.nan,np.nan,np.nan])
               coef_gas=np.vstack([coef_gas,coef_gas])
               fit_params=pd.DataFrame({'acorrs':[np.nan,np.nan],'slope_err':[np.nan,np.nan],\
                                 'aic':[np.nan,np.nan],'r2':[np.nan,np.nan],\
                                 'rmse':[np.nan,np.nan],'nrmse':[np.nan,np.nan]},index=['lin','exp'])
    
               return coef_gas, fit_params
           
        if gas=='ch4':
            if np.isfinite(np.nanmean(data_closure['CH4_dry']))==True:
               gas_col='CH4_dry' #If using picarro data and it exists
            else:
               coef_gas=np.array([np.nan,np.nan,np.nan])
               coef_gas=np.vstack([coef_gas,coef_gas])
               fit_params=pd.DataFrame({'acorrs':[np.nan,np.nan],'slope_err':[np.nan,np.nan],\
                                 'aic':[np.nan,np.nan],'r2':[np.nan,np.nan],\
                                 'rmse':[np.nan,np.nan],'nrmse':[np.nan,np.nan]},index=['lin','exp'])
               return coef_gas, fit_params
    
        if gas=='n2o':
            if np.isfinite(np.nanmean(data_closure['N2O_dry']))==True:
               gas_col='N2O_dry' #If using picarro data and it exists
            else:
               coef_gas=np.array([np.nan,np.nan,np.nan])
               coef_gas=np.vstack([coef_gas,coef_gas])
               fit_params=pd.DataFrame({'acorrs':[np.nan,np.nan],'slope_err':[np.nan,np.nan],\
                                 'aic':[np.nan,np.nan],'r2':[np.nan,np.nan],\
                                 'rmse':[np.nan,np.nan],'nrmse':[np.nan,np.nan]},index=['lin','exp'])
               return coef_gas, fit_params
    
        if gas=='co':
            if np.isfinite(np.nanmean(data_closure['CO_dry']))==True:
               gas_col='CO_dry' #If using picarro data and it exists
            else:
               coef_gas=np.array([np.nan,np.nan,np.nan])
               coef_gas=np.vstack([coef_gas,coef_gas])
               fit_params=pd.DataFrame({'acorrs':[np.nan,np.nan],'slope_err':[np.nan,np.nan],\
                                 'aic':[np.nan,np.nan],'r2':[np.nan,np.nan],\
                                 'rmse':[np.nan,np.nan],'nrmse':[np.nan,np.nan]},index=['lin','exp'])
               return coef_gas, fit_params
        
        #Make the fits        
        yfits_gas, coef_gas, goods_gas, exp_failed_gas, sse_s_gas, rsq_s_gas,\
        err_slopes_gas,acorrs_gas=closure_fitting(data_closure, gas_col)
        
        #Calculate statistics
        r2_aic_gas=stats_calc(len(data_closure), sse_s_gas, rsq_s_gas)
        fit_params=pd.DataFrame({'acorrs':acorrs_gas[0],'slope_err':err_slopes_gas[0],\
                                 'aic':[r2_aic_gas.aic_lin[0],r2_aic_gas.aic_exp[0]],\
                                 'r2':[r2_aic_gas.r2_lin[0],r2_aic_gas.r2_exp[0]],\
                                 'rmse':[goods_gas[0,0],goods_gas[0,1]],\
                                 'nrmse':[goods_gas[1,0],goods_gas[1,1]]},index=['lin','exp'])
        
        #Plot closure time series
        if config.plot_closures==True:
            try:
                fig=flux_plotter(gas_col, data_closure, yfits_gas, exp_failed_gas) #Plot
                fig_save(fig, data_closure.index, config.result_loc+'plots/'+gas, cur_source) #Save
            except IndexError:
                plt.close(fig)
                pass
    
    #If fitting fails, set everything to nan
    except (np.linalg.linalg.LinAlgError, ValueError, TypeError, AttributeError):
        coef_gas=np.array([np.nan,np.nan,np.nan])
        coef_gas=np.vstack([coef_gas,coef_gas])
        fit_params=pd.DataFrame({'acorrs':[np.nan,np.nan],'slope_err':[np.nan,np.nan],\
                         'aic':[np.nan,np.nan],'r2':[np.nan,np.nan],\
                         'rmse':[np.nan,np.nan],'nrmse':[np.nan,np.nan]},index=['lin','exp'])
        return coef_gas, fit_params
        
    return coef_gas, fit_params

#Calculate the flux    
def soil_flux(cols_to_calc, gas, coefs, cur_T, cur_P, cur_height):
    R=8.31446 #universal gas constant [J/mol*K]
    
    P_exists=~np.isnan(cur_P) #Check if the current air pressure is nan
      
    if P_exists==True: #Get air pressure, if it exists
        cur_P=cur_P*100 #hpa to pa
    
    #Set headspace temp to nan if it does not make sense
    if cur_T>60 or cur_T<-50:
        cur_T=np.nan
     
    #concentration changes in time [ppm/s]
    dcdt_lin=coefs[0,0]
    dcdt_exp=coefs[1,1]*coefs[1,2]
    
    #select correct molecular mass [g/mol]
    if gas=='co2':
        M=44.0095
    if gas=='ch4':
        M=16.0425
    if gas=='n2o':
        M=44.013        
    if gas=='co':
        M=28.010
    
    #calculate lin and exp flux
    if P_exists==True: #If air pressure data exists, use it to calculate the flux
        flux_lin=dcdt_lin*M*(cur_P/(R*(273.15+cur_T)))*(cur_height)*3600.0  #[ug (gas) m-2 h-1]
        flux_exp=dcdt_exp*M*(cur_P/(R*(273.15+cur_T)))*(cur_height)*3600.0  #[ug (gas) m-2 h-1]
    else: #If no pressure data, use ideal gas molar volume
        Vm=0.0224 #m3 mol-1
        flux_lin=dcdt_lin*(M/Vm)*(273.15/(273.15+cur_T))*(cur_height)*3600.0  #[ug (gas) m-2 h-1]
        flux_exp=dcdt_exp*(M/Vm)*(273.15/(273.15+cur_T))*(cur_height)*3600.0  #[ug (gas) m-2 h-1]

    #in case of co2, change units to [mg CO2 m-2 h-1]
    if gas=='co2':
        flux_lin=flux_lin/1000
        flux_exp=flux_exp/1000
    
    #put fluxes into the same df
    fluxes=pd.DataFrame({'lin':flux_lin,'exp':flux_exp},index=[0])
    
    #slopes to the same array
    dcdt=pd.DataFrame([dcdt_lin,dcdt_exp],index=['lin','exp'])
    
    return fluxes, dcdt

#Add the results of the current closure to the result dataframe
def result_array(results_df, gas, fluxes, concs, fit_params, data_closure):
    #Add also the statistic columns if wanted
    if config.save_statistics==True:
        temp=pd.DataFrame([[fluxes.lin[0],fluxes.exp[0],concs.init[0],concs.final[0],
                           fit_params.slope_err.lin,fit_params.slope_err.exp,
                           fit_params.nrmse.lin,fit_params.nrmse.exp,fit_params.r2.lin,
                           fit_params.r2.exp,fit_params.aic.lin,fit_params.aic.exp,
                           fit_params.acorrs.lin,fit_params.acorrs.exp]],
                           index=[data_closure.index[0]], columns=results_df.columns)
    else: #Only the fluxes (and mixing ratios)
        temp=pd.DataFrame([[fluxes.lin[0],fluxes.exp[0],concs.init,concs.final]],
                          index=[data_closure.index[0]], columns=results_df.columns)
    
    results_df=results_df.append(temp)
    
    return results_df

#Add the current values from the optional columns in the times file into the results dataframe
def result_array_rest(results_rest, results_extra, data_closure, anc_means, cur_source, ind):
    #If the results dataframe does not exist, create it
    if len(results_rest)==0:
        results_rest=pd.DataFrame(results_extra.iloc[0,:]).transpose()
        results_rest.index=[anc_means.index[0]] #Closure starting time as index
        results_rest=results_rest.join(anc_means) #Join the ancillary data of the closure
        #Add fit length column
        results_rest['Fit length [s]']=np.int(data_closure['Fit length [s]'][len(data_closure)-1]) 
        results_rest['Source']=cur_source #Add the current source
    
    else: #If the dataframe exists (not the first closure of the data), append it with the new values
        temp=pd.DataFrame(results_extra.iloc[ind,:]).transpose()
        temp.index=[anc_means.index[0]]
        temp=temp.join(anc_means)
        temp['Fit length [s]']=np.int(data_closure['Fit length [s]'][len(data_closure)-1])
        temp['Source']=cur_source
     
        results_rest=results_rest.append(temp) #Append the result dataframe
    
    return results_rest

#Select starting and ending times for the fits interactively
def interactive_times(data_file, times_file):
    #Check which gas columns exist in the data file
    cols_to_calc, data_file=check_columns(data_file)
    
    for ind in range(len(times_file)):
        #Get the current closure
        data_closure, cur_source, stime, skip_flag=start_end_times_source_name(data_file, times_file, ind)
        
        #Check if the data_closure table returned empty
        if skip_flag==1:
            timestr=times_file['Start time'][ind].strftime('%Y-%m-%d %H:%M:%S')
            sys.exit('No data for the measurement starting on: '+timestr+' was found.'+
                  ' Check the Start and End time in the times file for that measurement! '+
                  'If times file is ok, the data file does not contain the data for that measurement!')
            #Save the updated times_file?
            if config.interactive_save==True:
                #Get the name of the times file
                _,times_name=ntpath.split(config.times_loc)
                times_name=times_name[:-5] #Remove the extension of the filename, assuming xlsx
                #Save the times file as an excel file
                times_file.to_excel(config.result_loc+times_name+'_new.xlsx',index=False)
        
        #Loop the same plot until the clicks have been properly given.
        while True:
            try:
                fig, ax=plt.subplots(figsize=(12,10))
                
                #Plot the data and the legend of all the gases that have data, remove yticks
                if cols_to_calc.CO2_dry[0]==True and data_closure.CO2_dry.isnull().all()==False:
                    ax.plot(data_closure.CO2_dry,'o',color='k',label='CO2')
                    ax.set_yticks([])
                    ax.legend(loc='upper left',bbox_to_anchor=(0.3,1))
                if cols_to_calc.CH4_dry[0]==True and data_closure.CH4_dry.isnull().all()==False:
                    ax=ax.twinx()
                    ax.plot(data_closure.CH4_dry,'o',color='r',label='CH4')
                    ax.set_yticks([])
                    ax.legend(loc='upper left',bbox_to_anchor=(0.4,1))
                if cols_to_calc.N2O_dry[0]==True and data_closure.N2O_dry.isnull().all()==False:
                    ax=ax.twinx()
                    ax.plot(data_closure.N2O_dry,'o',color='g',label='N2O')
                    ax.set_yticks([])
                    ax.legend(loc='upper left',bbox_to_anchor=(0.5,1))
                if cols_to_calc.CO_dry[0]==True and data_closure.CO_dry.isnull().all()==False:
                    ax=ax.twinx()
                    ax.plot(data_closure.CO_dry,'o',color='m',label='CO')
                    ax.set_yticks([])
                    ax.legend(loc='upper left',bbox_to_anchor=(0.6,1))
                
                #Add title with the source and starting time to the plot
                title_str = str(cur_source) + ', ' + stime.strftime(format='%Y-%m-%d %H:%M:%S')
                ax.set_title(title_str)
                
                #tick format to %H:%M:%S
                fmt=mpl.dates.DateFormatter('%H:%M:%S')
                ax.xaxis.set_major_formatter(fmt)
		
                #Make the plot and wait for the clicks
                new_times=plt.ginput(3, show_clicks=True)
                
                plt.draw()
                plt.pause(0.001)
                plt.close()
                
                #Get the new start and end times from the clicks
                new_start=mpl.dates.num2date(new_times[0][0])
                new_end=mpl.dates.num2date(new_times[1][0])
                
                #Remove the timezone information
                new_start=new_start.replace(tzinfo=None)
                new_end=new_end.replace(tzinfo=None)
                
                #Check that new_start is smaller than new_end
                #If it is not, show the figure again.
                if new_start>new_end:
                    print("Select the starting time before the ending time! "
                          "The starting time cannot be later than the ending time!")
                    continue
                
                #Check that the selected fitting period is longer than fit_minimum in the config file 
                #If it is not, show the figure again.
                if (new_end-new_start).seconds<config.fit_minimum:
                    print("The selected period was shorter than the fit_minimum "
                          "in the config file! Select a longer period or adjust "
                          "the fit_minimum in the config file.")
                    continue
                
            except (IndexError, NameError):
                print("No starting or ending time was selected. Try again.")
                plt.close()
                continue
            
            break
    
        #Update the times_file with new interactively selected start and end times
        times_file['Start time'][ind]=new_start
        times_file['End time'][ind]=new_end
    
    #Save the updated times_file?
    if config.interactive_save==True:
	times_file_save=times_file.copy()
        datestrs=pd.Series()
        startstrs=pd.Series()
        endstrs=pd.Series()
        #Change Date and Time columns to string
        for date, st_time, et_time, ind in zip(times_file.Date, times_file['Start time'], 
                                               times_file['End time'], times_file.index):
            datestrs=pd.concat([datestrs,pd.Series(date[0:10], index=[ind])])
            startstrs=pd.concat([startstrs,pd.Series(st_time.strftime('%H:%M:%S'), index=[ind])])
            endstrs=pd.concat([endstrs,pd.Series(et_time.strftime('%H:%M:%S'), index=[ind])])
        
        #Replace Date, Start time and End time column datetimes with their respective strings
        times_file_save.Date=datestrs
        times_file_save['Start time']=startstrs
        times_file_save['End time']=endstrs
	
        #Get the name of the times file
        _,times_name=ntpath.split(config.times_loc)
        times_name=times_name[:-5] #Remove the extension of the filename, assuming xlsx
        #Save the times file as an excel file
        times_file.to_excel(config.result_loc+times_name+'_new.xlsx',index=False)
    
    return times_file

#The main function that handles the flux calculation procedure
def fluxcalc(times_file, data_file):
    #Initialize resulting dataframes
    results_co2=init_results('co2')
    results_ch4=init_results('ch4')
    results_n2o=init_results('n2o')
    results_co=init_results('co')
      
    #Check which gas columns exist in the data file
    cols_to_calc,data_file=check_columns(data_file)
    
    #Initialize resulting dataframe for the optional parameters
    #It is assumed that the optional parameters are the same in all data files
    results_extra=init_results_rest(times_file) #For the ancillary data
    results_rest=pd.DataFrame()
    
    #Start the progress bar
    with alive_bar(len(times_file)) as bar:    
        #Loop through the closures
        for ind in range(len(times_file)):    
            #Get current closure data, source chamber and starting time
            data_closure, cur_source, stime, skip_flag=start_end_times_source_name(data_file, times_file, ind)
            
            #Check if the data_closure table returned empty
            if skip_flag==1:
                timestr=times_file['Start time'][ind].strftime('%Y-%m-%d %H:%M:%S')
                sys.exit('No data for the measurement starting on: '+timestr+' was found.'+
                      ' Check the Start and End time in the times file for that measurement! '+
                      'If times file is ok, the data file does not contain the data for that measurement!')
            
            #Limit the current closure data according to minimum and maximum fit parameters
            data_closure, stime_new, skip_flag=limit_closure_time(data_closure, stime)
            
            #If the closure time is shorter than the minimum required length, skip it.
            if skip_flag==1:
                datestr=dt.datetime.strftime(stime,'%Y-%m-%d %H:%M:%S')
                sys.exit(cur_source + ' on '+datestr+' has a shorter closure time than "fit_minimum" in the config file!')
            
            #Get the current hs temperature, air pressure and system height from the times file
            cur_T=times_file['Headspace temperature (C)'][ind]
            
            #If headspace temperature is not given in the times file, check the data file and calculate the mean
            #If no headspace temperature is found, throw an error
            if np.isnan(cur_T)==True:
                try:
                    cur_T=data_closure['Headspace temperature (C)'].mean()
                
                    if np.isnan(cur_T)==True:
                        sys.exit("Headspace temperature for "+cur_source+" in "+str(stime)+
                                 " was not found neither from the times or data file!")

                except IndexError:
                        sys.exit("Headspace temperature for "+cur_source+" in "+str(stime)+
                                 " was not found neither from the times or data file!")
                    
            #If the air temperature is not given in the times file, check the data file and calculate the mean
            #If no air pressure is given, set it to nan
            cur_P=times_file['Air pressure (hPa)'][ind]
            if np.isnan(cur_P)==True:
                try:
                    cur_P=data_closure['Air pressure (hpa)'].mean()
                except KeyError: #If the air pressure column does not exist
                    pass
                    
            cur_height=times_file['System height (m)'][ind]
            anc_means=pd.DataFrame({'T_mean':cur_T,'P_mean':cur_P,'CMB_h':cur_height},index=[stime_new])
            
            #Calculate the means for the possible ancillary data given in the data file
            anc_means=datafile_means(anc_means,data_closure)
            
            #CO2
            if cols_to_calc.CO2_dry[0]==True and data_closure.CO2_dry.isnull().all()==False:
                #Get the initial and final CO2 mixing ratios during the closure
                concs_co2=pd.DataFrame({'init':data_closure.CO2_dry[0],'final':
                                        data_closure.CO2_dry[-1]},index=[0])
                #Make the fits
                coef_co2, fit_params_co2=gas_fitting(data_closure,'co2',cur_source) #Make the fits
                #Calculate the flux
                fluxes_co2, dcdt_co2=soil_flux(cols_to_calc, 'co2', coef_co2, cur_T, cur_P, cur_height)
                #Save the results to the results dataframe
                results_co2=result_array(results_co2, 'co2', fluxes_co2, concs_co2,
                                         fit_params_co2, data_closure)
            
            #Repeat for the rest of the gases
            #CH4
            if cols_to_calc.CH4_dry[0]==True and data_closure.CH4_dry.isnull().all()==False:
                concs_ch4=pd.DataFrame({'init':data_closure.CH4_dry[0],'final':
                                        data_closure.CH4_dry[-1]},index=[0])
                coef_ch4, fit_params_ch4=gas_fitting(data_closure,'ch4',cur_source)
                fluxes_ch4, dcdt_ch4=soil_flux(cols_to_calc, 'ch4', coef_ch4, cur_T, cur_P, cur_height)
                results_ch4=result_array(results_ch4, 'ch4', fluxes_ch4, concs_ch4, fit_params_ch4, data_closure)
            
            #N2O
            if cols_to_calc.N2O_dry[0]==True and data_closure.N2O_dry.isnull().all()==False:
                concs_n2o=pd.DataFrame({'init':data_closure.N2O_dry[0],'final':
                                        data_closure.N2O_dry[-1]},index=[0])
                coef_n2o, fit_params_n2o=gas_fitting(data_closure,'n2o',cur_source)
                fluxes_n2o, dcdt_n2o=soil_flux(cols_to_calc, 'n2o', coef_n2o, cur_T, cur_P, cur_height)
                results_n2o=result_array(results_n2o, 'n2o', fluxes_n2o, concs_n2o, fit_params_n2o, data_closure)
            
            #CO
            if cols_to_calc.CO_dry[0]==True and data_closure.CO_dry.isnull().all()==False:
                concs_co=pd.DataFrame({'init':data_closure.CO_dry[0],'final':
                                        data_closure.CO_dry[-1]},index=[0])
                coef_co, fit_params_co=gas_fitting(data_closure,'co',cur_source)    
                fluxes_co, dcdt_co=soil_flux(cols_to_calc, 'co', coef_co, cur_T, cur_P, cur_height)
                results_co=result_array(results_co, 'co', fluxes_co, concs_co, fit_params_co, data_closure)
                    
            #Save the current ancillary data and the data from the optional columns of the times file
            results_rest=result_array_rest(results_rest, results_extra, data_closure, anc_means, cur_source, ind)
            
            bar() #Update the progress bar
    
    #Create a dataframe of dataframes for easier moving of all the results within the program
    results=(results_rest, results_co2, results_ch4, results_n2o, results_co)
    
    return results

#Combine the resulting dataframes into one and order them somewhat sensibly
def results_combiner(results):
    results_rest=results[0]
    results_co2=results[1]
    results_ch4=results[2]
    results_n2o=results[3]
    results_co=results[4]
    
    #Add the first columns into the final result array. These columns should always exist
    results_all=pd.DataFrame(results_rest[['Source','CMB_h','Fit length [s]','T_mean','P_mean']], 
                index=results_rest.index)
    results_all.columns=['Source','Effective chamber height [m]','Fit length [s]',
                         'Headspace temperature [C]','Air pressure [hpa]']
    
    #Drop the columns, which were added to results_all
    results_rest=results_rest.drop(columns=['Source','CMB_h','Fit length [s]','T_mean','P_mean'])
    
    #Create dataframes for fluxes and statistics
    results_all2=pd.DataFrame(index=results_rest.index) #For fluxes
    results_all3=pd.DataFrame(index=results_rest.index) #For statistics
    
    if len(results_co2)!=0: #For CO2 fdata
        #Add flux and conc columns
        results_all2=results_all2.join(results_co2[['Lin_Flux_CO2 [mg CO2 m-2 h-1]', 
                                                    'Exp_Flux_CO2 [mg CO2 m-2 h-1]',
                                                    'Start_CO2 [ppm]','End_CO2 [ppm]']])
        #Drop flux and conc columns, leaving only statistics columns
        results_co2=results_co2.drop(columns=['Lin_Flux_CO2 [mg CO2 m-2 h-1]',
                                              'Exp_Flux_CO2 [mg CO2 m-2 h-1]',
                                              'Start_CO2 [ppm]','End_CO2 [ppm]']) 
        results_all3=results_all3.join(results_co2) #add statistics columns
            
    if len(results_ch4)!=0: #For CH4 data
        results_all2=results_all2.join(results_ch4[['Lin_Flux_CH4 [ug CH4 m-2 h-1]',
                                                    'Exp_Flux_CH4 [ug CH4 m-2 h-1]',
                                                    'Start_CH4 [ppm]','End_CH4 [ppm]']])
        results_ch4=results_ch4.drop(columns=['Lin_Flux_CH4 [ug CH4 m-2 h-1]',
                                              'Exp_Flux_CH4 [ug CH4 m-2 h-1]',
                                              'Start_CH4 [ppm]','End_CH4 [ppm]']) 
        results_all3=results_all3.join(results_ch4)
        
    if len(results_n2o)!=0: #For N2O data
        results_all2=results_all2.join(results_n2o[['Lin_Flux_N2O [ug N2O m-2 h-1]',
                                                    'Exp_Flux_N2O [ug N2O m-2 h-1]',
                                                    'Start_N2O [ppm]','End_N2O [ppm]']])
        results_n2o=results_n2o.drop(columns=['Lin_Flux_N2O [ug N2O m-2 h-1]',
                                              'Exp_Flux_N2O [ug N2O m-2 h-1]',
                                              'Start_N2O [ppm]','End_N2O [ppm]']) 
        results_all3=results_all3.join(results_n2o)
        
    if len(results_co)!=0: #For CO data
        results_all2=results_all2.join(results_co[['Lin_Flux_CO [ug CO m-2 h-1]',
                                                   'Exp_Flux_CO [ug CO m-2 h-1]',
                                                   'Start_CO [ppm]','End_CO [ppm]']])
        results_co=results_co.drop(columns=['Lin_Flux_CO [ug CO m-2 h-1]',
                                            'Exp_Flux_CO [ug CO m-2 h-1]',
                                            'Start_CO [ppm]','End_CO [ppm]']) 
        results_all3=results_all3.join(results_co)
    
    #Round the fluxes and statistics
    results_all2=results_all2.round(2)
    results_all3=results_all3.round(3)        
        
    #Combine everything
    results_all=results_all.join(results_all2) #Fluxes
    results_all=results_all.join(results_rest) #Extra columns added by the user
    results_all=results_all.join(results_all3) #Statistics
    
    #Set index column name
    results_all.index.name='Datetime (measurement start)'
    
    #Sort by index
    results_all=results_all.sort_index()
    
    #Save the result array
    results_all.to_csv(config.result_loc+config.site_name+'.csv',
                       na_rep='nan')
    
    print("Results data saved.")
    return

def main():
    print('Starting '+APPNAME+' '+'('+VERSION+').\n')
    #Start timer
    start = time.time()
    
    #Check the config file
    check_config()
    
    #Load the data file
    data_file=load_data_file()
    
    #Load the times file
    times_file=load_times_file()
    
    #Perform time averaging
    if config.time_avg>0:
        data_file=data_time_avg(data_file)
    
    #Select closure starting and ending points interactively
    if config.interactive_times==True:
        times_file = interactive_times(data_file, times_file)
    
    #Discard seconds from the start of the closures
    if config.skip_start>0:
        times_file['Start time']=times_file['Start time']+dt.timedelta(seconds=config.skip_start)
    
    #Calculate the fluxes
    print("Calculating fluxes.")
    results = fluxcalc(times_file, data_file)
    
    #Combine and save the results
    results_combiner(results)
     
    #End timer
    end = time.time()
    elapsed_time="{:.2f}".format(end-start)
    print('Time elapsed: '+elapsed_time+' seconds')
    
    print('Finished!')
    return

###################################################################################################
#IF NAME IS MAIN, RUN MAIN. YEAH!
###################################################################################################        
#If the file is not imported, start the program from function main()
if __name__ == "__main__":
   main()     
