# This script contains the code for the Alphasense calibration models AS1, AS2, AS3, AS4

def formula1( we_raw, ae_raw, we_0e, ae_0e, nt, sens ):
  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= nt * ae_raw
  return we_raw/sens

def formula2( we_raw, ae_raw, we_0e, ae_0e, we_0t, ae_0t, kt, sens ):
  we_0 = we_0t - we_0e
  ae_0 = ae_0t - ae_0e
  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= kt * ae_raw * (we_0 / ae_0)
  return we_raw/sens

def formula3( we_raw, ae_raw, we_0e, ae_0e, we_0t, ae_0t, kt, sens ):
  we_0 = we_0t - we_0e
  ae_0 = ae_0t - ae_0e
  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= kt * ae_raw - (we_0 - ae_0)
  return we_raw/sens

def formula4( we_raw, ae_raw, we_0e, ae_0e, we_0t, ae_0t, kt, sens ):
  we_0 = we_0t - we_0e
  we_raw -= we_0e
  ae_raw -= ae_0e
  we_raw -= kt * ae_raw - we_0
  return we_raw/sens

def AS1_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, o3_const ):
    '''
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    dataset : string
        Name of the dataset file, for example DD1(Jun).csv
        
    no2x : array
        Electrode potentials no2op1(mV) and no2op2(mV) from the NO2 sensor
    
    no2y : array
        Reference values for NO2(ppb)
    
    o3x : array
        Electrode potentials o3op1(mV) and o3op2(mV) from the O3 sensor
        
    o3y : array
       Reference values for O3(ppb)
        
    no2_const : array
        Constants for NO2 calibration. Get these from globals.py
        
    o3_const : array
        Constants for O3 calibration. Get these from globals.py
    '''
    const = g.AS_const[dataset]
    
    pred_no2 = formula1(no2x[:,0], no2x[:,1], const['NO2_WE_0E'], const['NO2_AE_0E'], no2_const[:,0], const['SENSITIVITY_NO2'])
    
    pred_o3 = formula1(o3x[:,0], o3x[:,1], const['O3_WE_0E'], const['O3_AE_0E'], o3_const[:,0], const['SENSITIVITY_O3'])
        
    return pred_no2, pred_o3 - pred_no2

def AS2_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, o3_const ):
    '''
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    dataset : string
        Name of the dataset file, for example DD1(Jun).csv
        
    no2x : array
        Electrode potentials no2op1(mV) and no2op2(mV) from the NO2 sensor
    
    no2y : array
        Reference values for NO2(ppb)
    
    o3x : array
        Electrode potentials o3op1(mV) and o3op2(mV) from the O3 sensor
        
    o3y : array
       Reference values for O3(ppb)
        
    no2_const : array
        Constants for NO2 calibration. Get these from globals.py
        
    o3_const : array
        Constants for O3 calibration. Get these from globals.py
    '''
    const = g.AS_const[dataset]
    
    pred_no2 = formula2(no2x[:,0], no2x[:,1], const['NO2_WE_0E'], const['NO2_AE_0E'], const['NO2_WE_0T'], const['NO2_AE_0T'], no2_const[:,1], const['SENSITIVITY_NO2'])

    pred_o3 = formula2(o3x[:,0], o3x[:,1], const['O3_WE_0E'], const['O3_AE_0E'], const['O3_WE_0T'], const['O3_AE_0T'], o3_const[:,1], const['SENSITIVITY_O3'])
    
    return pred_no2, pred_o3 - pred_no2

def AS3_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, o3_const ):
    '''
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    dataset : string
        Name of the dataset file, for example DD1(Jun).csv
        
    no2x : array
        Electrode potentials no2op1(mV) and no2op2(mV) from the NO2 sensor
    
    no2y : array
        Reference values for NO2(ppb)
    
    o3x : array
        Electrode potentials o3op1(mV) and o3op2(mV) from the O3 sensor
        
    o3y : array
       Reference values for O3(ppb)
        
    no2_const : array
        Constants for NO2 calibration. Get these from globals.py
        
    o3_const : array
        Constants for O3 calibration. Get these from globals.py
    '''
    const = g.AS_const[dataset]
    
    pred_no2 = formula3(no2x[:,0], no2x[:,1], const['NO2_WE_0E'], const['NO2_AE_0E'], const['NO2_WE_0T'], const['NO2_AE_0T'], no2_const[:,2], const['SENSITIVITY_NO2'])
    
    pred_o3 = formula3(o3x[:,0], o3x[:,1], const['O3_WE_0E'], const['O3_AE_0E'], const['O3_WE_0T'], const['O3_AE_0T'], o3_const[:,2], const['SENSITIVITY_O3'])
    
    return pred_no2, pred_o3 - pred_no2
    
def AS4_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, o3_const ):
    '''
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    dataset : string
        Name of the dataset file, for example DD1(Jun).csv
        
    no2x : array
        Electrode potentials no2op1(mV) and no2op2(mV) from the NO2 sensor
    
    no2y : array
        Reference values for NO2(ppb)
    
    o3x : array
        Electrode potentials o3op1(mV) and o3op2(mV) from the O3 sensor
        
    o3y : array
       Reference values for O3(ppb)
        
    no2_const : array
        Constants for NO2 calibration. Get these from globals.py
        
    o3_const : array
        Constants for O3 calibration. Get these from globals.py
    '''
    const = g.AS_const[dataset]
    
    pred_no2 = formula4(no2x[:,0], no2x[:,1], const['NO2_WE_0E'], const['NO2_AE_0E'], const['NO2_WE_0T'], const['NO2_AE_0T'], no2_const[:,3], const['SENSITIVITY_NO2'])
    
    pred_o3 = formula4(o3x[:,0], o3x[:,1], const['O3_WE_0E'], const['O3_AE_0E'], const['O3_WE_0T'], const['O3_AE_0T'], o3_const[:,3], const['SENSITIVITY_O3'])
        
    return pred_no2, pred_o3 - pred_no2
