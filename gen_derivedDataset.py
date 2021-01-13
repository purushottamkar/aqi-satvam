import pandas as pd
import numpy as np

def avg_dataset( g, dataset, window = 1, offset = 0 ):
    '''
    Creates a dataset with data averaged over rolling windows
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains global variables like path, consts, etc.
        
    dataset : string
        Name of the dataset file, for example DD1(Jun).csv
        
    window : int, optional
        The number of intervals over which averaging will be done. 
        For window = 5, this function will perform averging over rolling windows 5 min intervals
        The default value is 1.
    
    offset : int, optional
        Number of time-stamps to be skipped at the starting of the dataset.
        The default value is 0.
    '''
    
    data = pd.read_csv( g.datasetPath + dataset )
    
    # Removing offset number of rows from the starting of the dataset
    for i in range( 0, offset ):
       data.drop( [data.index[0]], inplace = True )   
    
    newdata = pd.DataFrame()
    newdata['Time'] = data['Time']
    
    # Performing rolling average
    for feature in g.featureList[1:-1]:
        newdata[feature] = data[feature].rolling( window = window, min_periods = 0 ).mean()
    
    desired_indices = [ i for i in range( 0, len(newdata) ) if i % window == 0 ]
    newdata = newdata.iloc[desired_indices]
    
    # Data clean up
    newdata['Valid'] = 1 - newdata.isnull().any( axis = 1).astype( int )
    newdata['Valid'] = np.where( newdata['Temp(C)'] > 50 , 0, np.where( newdata['Temp(C)'] < 1, 0, newdata['Valid'] ) )
    newdata['Valid'] = np.where( newdata['RH(%)'] > 100, 0, np.where( newdata['RH(%)'] < 1, 0, newdata['Valid'] ) )
    newdata['Valid'] = np.where( newdata['no2op1(mV)'] > 400 , 0, np.where( newdata['no2op1(mV)'] < 1, 0, newdata['Valid'] ) )
    newdata['Valid'] = np.where( newdata['no2op2(mV)'] > 400 , 0, np.where( newdata['no2op2(mV)'] < 1, 0, newdata['Valid'] ) )
    newdata['Valid'] = np.where( newdata['o3op1(mV)'] > 400 , 0, np.where( newdata['o3op1(mV)'] < 1, 0, newdata['Valid'] ) )
    newdata['Valid'] = np.where( newdata['o3op2(mV)'] > 400 , 0, np.where( newdata['o3op2(mV)'] < 1, 0, newdata['Valid'] ) )
    newdata['Valid'] = np.where( newdata['Ref. O3(ppb)'] > 200 , 0, np.where( newdata['Ref. O3(ppb)'] < 1, 0, newdata['Valid'] ) )
    newdata['Valid'] = np.where( newdata['Ref. NO2(ppb)'] > 200 , 0, np.where( newdata['Ref. NO2(ppb)'] < 0, 0, newdata['Valid'] ) )
    
    # Saving the derived dataset
    newdata.to_csv( g.derivedDatasetPath + dataset[:-4] + '-AVG' + str(window) + '.csv', index = False )

def small_dataset( g, dataset, numSamples = 2500 ):
    '''
    Creates a smaller version of a dataset by subsampling time-stamps
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains global variables like paths, consts, etc.
    
    dataset : string
        Name of the dataset file, for example DD1(Jun).csv
        
    numSamples : int, optional
        The number of samples to be sampled from the dataset. 
        The default is 2500.
    '''
    data = pd.read_csv( g.datasetPath + dataset )
    
    data = data[ data['Valid'] == 1 ]
    
    data = data.iloc[:numSamples, :]
    data.to_csv( g.derivedDatasetPath + dataset[:-4] + '-SMALL.csv', index = False )

def agg_dataset( g, dataset ):
    '''
    For a given sensor, aggregates data from both deployments to create a new dataset
	
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains global variables like path, consts, etc.
        
    dataset : string
        Name of the sensor (without deployment information), for example DD1, MM5, etc.
    '''
    data1 = pd.read_csv( g.datasetPath + dataset + '(Jun).csv' )
    data2 = pd.read_csv( g.datasetPath + dataset + '(Oct).csv' )
    
    data1 = data1[ data1['Valid'] == 1 ]
    data2 = data2[ data2['Valid'] == 1 ]
    
    newdata = data1.append( data2, ignore_index = True )
    newdata.to_csv( g.derivedDatasetPath + dataset + '(Jun-Oct).csv', index = False )


