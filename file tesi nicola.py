#!/usr/bin/env python
# coding: utf-8

# Importing modules
#graphic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time;
import pdb


# Import system modules
import arcpy
from arcpy import env
from arcpy.sa import *
from numpy import mean
print("all right")

# Set environment settings

# In[ ]:


env.workspace = "C:/Users/Tommaso/Documents/Tesi Nicola/shp file"
print ("set variables")
fc = "C:/Users/Tommaso/Documents/Tesi Nicola/shp file/chief_towns_points_new.shp"
sumMAE = 0
sumRMSE = 0
sumRMSEs = 0
sumRMSEu = 0
minMes = 999999999
maxMes = -1
i = 0

provList=[]
xList = []
yList = []
diffList = []
print("all right")


# In[ ]:


fc


# 

# In[ ]:


zField = "PC_TRASF_m"
cellSize = 5000.0 # instead of 2000 to speed up the process
power = 10


''' RUN JUST ONCE

for x in range(102):
    # Set local variables
    in_features = "chief_towns_points_new.shp"
    out_feature_class = "chief_towns_points_new_" + str(x) + ".shp"
    where_clause = '"FID" <> ' + str(x)
    # Execute Select
    arcpy.Select_analysis(in_features, out_feature_class, where_clause)
    print (out_feature_class + " created.")
'''

indices=[]
list_rows=[]

with arcpy.da.SearchCursor(fc, ['FID','Provincia', 'SHAPE@XY', 'PC_TRASF_m']) as cursor:
    for row in cursor:
        i = row[0]
        indices.append(i)
        list_rows.append(row)


# Search cursor

# In[ ]:
def IDW_params(power,numberOfPoints):
    province_list = []
    real_values_out_list=[]
    predicted_values_out_list=[]
    ape_values_out_list=[]
    arcpy.env.extent = arcpy.Extent(300000, 4000000, 1300000, 5300000) #hard code: the extent of raster in the current coords system
    
    for i in range(102): # hard code: it is the number of provinces
        fc="chief_towns_points_new_" + str(i) + ".shp"
        searchRadius = RadiusVariable(numberOfPoints,maxDistance=500000) # min (maxDistance) to work 68000
        # it takes the numberOfPoints, except for those farther than maxDistance
        outIDW = Idw(fc, zField, cellSize, power, searchRadius)
        n_raster = Raster(outIDW)
        print("IDW n."+ str(i)+" ottenuto usando "+fc+" (senza "+list_rows[i][1]+")")
        # fill the list of provinces
        province_list.append(list_rows[i][1])
        real_value_out=list_rows[i][3]
        long_value_out= list_rows[i][2][0]
        lat_value_out= list_rows[i][2][1]
        ''' 
        # SHOWS THE IN-SAMPLE REAL VALUES AND INTERPOLATIONS
        with arcpy.da.SearchCursor(fc, ['FID','Provincia', 'SHAPE@XY', 'PC_TRASF_m']) as cursor:
            for row in cursor:
                print(row)
                n = row[0]
                real_value=row[3]
                
                grid = "idwP2C10-" + str(n)
                x, y = row[2]  #x=longitude y=latitude
                
                #cvRes= arcpy.CrossValidation_ga (outIDW)
                pValue = arcpy.GetCellValue_management(n_raster,str(x)+" "+str(y),"1") 
                # 1st argument: the interpolation raster; 2nd arg: Longitude and Latitude converted into string; 3rd arg:optional ininfluent
                print(pValue)
                predicted_value=float(pValue.getOutput(0).replace(',','.'))
                diff = predicted_value - real_value
                #pStim = (0.607146991421771 * row[3]) + 539.125304424995
                #diffStim = pStim - row[3]
                #diffValueStim = float(pValue.getOutput(0).replace(',','.')) - pStim
                provList.append(row[1])
                xList = np.append(xList, [real_value])
                yList = np.append(yList, [predicted_value])
                diffList = np.append(diffList, [diff])
                print ("idwP2C10-" + str(n) + " value: " + str(real_value) + " predicted: " + str(predicted_value)+"  difference: "+str(diff))
                sumMAE = sumMAE + abs(diff)
                sumRMSE = sumRMSE + math.pow(diff,2)
                
                #sumRMSEs = sumRMSEs + math.pow(diffStim,2)
                #sumRMSEu = sumRMSEu + math.pow(diffValueStim,2) 
               
                i += 1
                if row[3] > maxMes:
                    maxMes = row[3]
                if row[3] < minMes:
                    minMes = row[3]
        '''
        # Prediction of the value related to the out-of-sample city (through longitude and latitude)
        pValue_out = arcpy.GetCellValue_management(n_raster,str(long_value_out)+" "+str(lat_value_out),"1") 
        # convert to a number format readable by Python
        predicted_value_out=float(pValue_out.getOutput(0).replace(',','.'))
        # compute the error
        diff_out = predicted_value_out - real_value_out
        # compute the absolute percentage error
        abs_perc_err=100*abs(diff_out)/real_value_out
        print(str(list_rows[i][1])+". Real value: "+str(real_value_out)+" ; Interpolated value: "+str(predicted_value_out)+" ; APE: "+str(round(abs_perc_err,1))+"%\n")
        # fill the list of real values
        real_values_out_list.append(real_value_out)
        # fill the list of predictions
        predicted_values_out_list.append(predicted_value_out)
        # fill the list of absolute percentage errors
        ape_values_out_list.append(abs_perc_err)
    # SEE THE DATAFRAME OF THE INTERPOLATED VALUES BY CITIES
    df_perf_interp_cv=pd.DataFrame() 
    df_perf_interp_cv['Province']=province_list
    df_perf_interp_cv['Real Value']=real_values_out_list
    df_perf_interp_cv['Interpolated Value']=predicted_values_out_list
    df_perf_interp_cv['Abs_perc_err']=ape_values_out_list
    
    # Computes and shows the MAPE statistics
    MAPE=mean(df_perf_interp_cv['Abs_perc_err'])
    print("MAPE = "+str(MAPE)+" %")
    return MAPE
    

# Resultsb

# In[ ]:
    
MAPE1=IDW_params(power=2,numberOfPoints=10)
print(MAPE1)
#EXAMPLES:
#35.852955484592286 % 150 km 10 nn; n
#35.85290118694821 % 500 km 10 nn
'''
print ("MAE:" + str(sumMAE / i) + " RMSE:" + str(math.sqrt(sumRMSE / i)) + " RMSEs:" + str(math.sqrt(sumRMSEs / i)) + " RMSEu:" + str(math.sqrt(sumRMSEu / i)))  
xvals = np.arange(minMes, maxMes, 50) # Grid
yvals = (0.607146991421771 * xvals) + 539.125304424995 # Evaluate regression function on xvals
plt.plot(xvals, yvals) # plot regression fuction from complete kriging surface 
plt.scatter(xList,yList) # plot scatter on measured/predicted values
ideal = [minMes,maxMes]
plt.plot(ideal, ideal) # plot ideal regression fuction


# Regression

# In[ ]:


#calculate best fitting regression on calculated errors
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = best_fit_slope_and_intercept(xList,yList)
print(m,b)
regression_line = [(m*x)+b for x in xList]
plt.plot(xList,regression_line)
plt.show() # Show the figure

'''