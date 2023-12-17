
import matplotlib.pyplot as plt 
import numpy as np
import os
import xarray as xr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
from sklearn import linear_model

from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 

from scipy.stats import pearsonr
import statsmodels.api as sm

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error

import imp
import src.importData as imports
import src.settings as settings
import src.stats as stats
import src.calc as calc
import src.plots as plots
import src.concat as ct

def createModel(x, y, return_results = True):
    model = linear_model.LinearRegression()
    # X_scaled =  StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    model.fit(x_train, y_train)
    
    #test model
    y_pred = model.predict(x_test)

    if return_results == True:
        print('TESTING')
        print('R squared = {}'.format(r2_score(y_test, y_pred)))
        print('MSE = {}'.format(MSE(y_test, y_pred)))

    return model

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def predict_spice(x, y, return_results = True, return_coef = False):
    if return_results == True:
        model = createModel(x,y, return_results = True)
        # X_scaled =  StandardScaler().fit_transform(x)
        y_pred_all = model.predict(x)
        print('ALL')
        print('R squared = {}'.format(r2_score(y, y_pred_all)))
        print('MSE = {}'.format(MSE(y, y_pred_all)))
    else:
        model = createModel(x,y, return_results = False)
        y_pred_all = model.predict(x)

    if return_coef == True:
        print('regression coefficients: {}'.format(model.coef_))
        return y_pred_all, model.coef_
    else:
        return y_pred_all

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def standardized_coef(coef, x, y):
    '''
    Standardized coefficients signify the mean change of the dependent variable
    given a one standard deviation shift in an independent variable.
    
    std coef = regression coef * x std / y std (Andrew F. Siegel, in Practical Business Statistics (Seventh Edition), 2016)
    
    INPUT:
    coef = regression coefficient from linear model
    x = independent variable data e.g. x['depth_avg_eke']
    y = dependent variable data e.g. y['depth_avg_spice_std']
    '''
    
    std_coef = (coef * x.std())/y.std()
    return std_coef

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def print_stdcoef(X, coef, y):
    for i in range(0, len(X.columns.values)):
        name = X.columns.values[i]
        print(name)
        std_coef = standardized_coef(coef[i], X[name], y)
        print('standardized coef = {}'.format(std_coef))
    
# ------------------------------------------------------------------------------------------------------------------------------------------------------

def select_features(X_train, y_train, X_test, k = 5, corr = True, mutual_info = False):
    # configure to select a subset of features
    if mutual_info == True:
        fs = SelectKBest(score_func=mutual_info_regression, k=k)
    elif corr == True:
        fs = SelectKBest(score_func=f_regression, k=k)
        
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    
    return X_train_fs, X_test_fs, fs

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def PredVsActual(y_pred_all, y_actual, floatids, ds_noindex, ds2, dist = True, size = (14,3)):
    fig, ax = plt.subplots(figsize = size)
    if dist == True:
        rs_dist = get_rs_dist(floatids)
        x = ct.new_dist(rs_dist)
        ticks, values, flt_dist_loc = plots.concatTickValues(rs_dist)
        label = 'distance (km)'
    else: 
        x = ds_noindex.index
        label = 'profile number'
    
    plt.plot(x, y_pred_all, c = 'tab:red')
    plt.plot(x, y_actual, c = 'tab:cyan')
    plt.ylabel('$\overline{\sigma}_{spice}$')
    plt.xlabel(label)
    plt.legend(['predicted', 'actual'])
    
    if dist == True:
        ax.set_xticks(ticks)
        ax.set_xticklabels(values)
        for i in range(0,len(flt_dist_loc)):
            ax.axvline(x = flt_dist_loc[i], linestyle = '--', color = 'grey')
    else:
        ax.axvline(x = ds2.loc['8489'].profile[-1], linestyle = '--', color = 'grey')
        ax.axvline(x = ds2.loc['8489'].profile[-1] + ds2.loc['8492'].profile[-1] + 1, linestyle = '--', color = 'grey')
    
    ax.set_xlim(0,x[-1])
    ax.grid()

    if len(floatids) == 3:
        # ax.set_ylim(0, 0.065)
        ax.text(0.15, 1.02, str(floatids[0]), transform = ax.transAxes)
        ax.text(0.49, 1.02, str(floatids[1]), transform = ax.transAxes)
        ax.text(0.82, 1.02, str(floatids[2]), transform = ax.transAxes)
    elif len(floatids) == 4:
        # ax.set_ylim(0, 0.09)
        ax.text(0.09, 1.02, str(floatids[0]), transform = ax.transAxes)
        ax.text(0.36, 1.02, str(floatids[1]), transform = ax.transAxes)
        ax.text(0.66, 1.02, str(floatids[2]), transform = ax.transAxes)
        ax.text(0.89, 1.02, str(floatids[3]), transform = ax.transAxes)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

def get_rs_dist(floatids):
    ema = imports.importFloatData(floatids)
    rs = {}
    dist = {}
    rs_dist = {}
    for floatid in floatids:
        float_num = ema[floatid]
        rs["{}".format(floatid)] = calc.findRSperiod(float_num)
        dist["{}".format(floatid)] = calc.cum_dist(float_num.longitude, float_num.latitude)
        rs_dist["{}".format(floatid)] = dist[str(floatid)][rs[str(floatid)]]
        
    return rs_dist 

# ------------------------------------------------------------------------------------------------------------------------------------------------------
