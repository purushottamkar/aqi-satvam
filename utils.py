import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
import alphasense as ac
import itertools as itt
from metric_learn import MLKR

algoTrainList={}
algoTestList={}
algoHyperparamList = {}

def KNND_MLTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	Training Code for the distance-weighed k-nearest neighbors method with a learnt metric KNN-D(ML)
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model_ox : model for O3 calibration
    
    model_no2 : model for NO2 calibration
	
    NWmodel : model for NW(ML) which is being used as a learnt metric
    '''
    X = data.iloc[:,2:]
    
    # Loading the Nadarya Watson(learnt metric) Model
    fileName = str(i) + '_' + 'NW(ML)' + '_' + dataset[:-4] + '_model'
    NWmodel = pickle.load( open(g.modelPath + fileName, 'rb') )['model']
    
    # For O3
    reg = KNeighborsRegressor( n_neighbors = hyperparams1['n_neighbors'], weights = 'distance' )
    y_ox = data.iloc[:,0]
    model_ox = reg.fit( X.dot( NWmodel[0].T ), y_ox )
    
    # For NO2
    reg = KNeighborsRegressor( n_neighbors = hyperparams2['n_neighbors'], weights = 'distance' )
    y_no2 = data.iloc[:,1]
    model_no2 = reg.fit( X.dot( NWmodel[1].T ), y_no2 )

    return ( model_ox, model_no2, NWmodel )

# Testing Code for KNN-D(ML)
def KNND_MLTest( model, data ):
    NWmodel = model[2]
    Xt = data.iloc[:,2:]
    
    yPred_ox = model[0].predict( Xt.dot( NWmodel[0].T ))
    yPred_no2 = model[1].predict( Xt.dot( NWmodel[1].T ))
    
    return np.column_stack(( yPred_ox, yPred_no2 ))

algoTrainList['KNN-D(ML)'] = KNND_MLTrain
algoTestList['KNN-D(ML)'] = KNND_MLTest
algoHyperparamList['KNN-D(ML)'] = { 'n_neighbors': np.array([2, 4, 6, 8, 10, 15, 20]) }

def KNN_MLTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	Training Code for the k-nearest neighbors method with a learnt metric KNN(ML)
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model_ox : model for O3 calibration
    
    model_no2 : model for NO2 calibration
        
    NWmodel : model for NW(ML) which is being used as a learnt metric
    '''
    X = data.iloc[:,2:]
    
    # Loading the Nadarya Watson(learnt metric) Model
    fileName = str(i) + '_' + 'NW(ML)' + '_' + dataset[:-4] + '_model'
    NWmodel = pickle.load( open( g.modelPath + fileName, 'rb') )['model']
    
    # For O3
    reg = KNeighborsRegressor( n_neighbors = hyperparams1['n_neighbors'] )
    y_ox = data.iloc[:,0]
    model_ox = reg.fit( X.dot( NWmodel[0].T ), y_ox )
    
    # For NO2
    reg = KNeighborsRegressor( n_neighbors = hyperparams2['n_neighbors'] )
    y_no2 = data.iloc[:,1]
    model_no2 = reg.fit( X.dot( NWmodel[1].T ), y_no2 )

    return ( model_ox, model_no2, NWmodel )

# Testing Code for KNN(ML)
def KNN_MLTest( model, data ):
    NWmodel = model[2]
    Xt = data.iloc[:,2:]

    yPred_ox = model[0].predict( Xt.dot( NWmodel[0].T ))
    yPred_no2 = model[1].predict( Xt.dot( NWmodel[1].T ))

    return np.column_stack(( yPred_ox, yPred_no2 ))

algoTrainList['KNN(ML)'] = KNN_MLTrain
algoTestList['KNN(ML)'] = KNN_MLTest
algoHyperparamList['KNN(ML)'] = { 'n_neighbors': np.array([2, 4, 6, 8, 10, 15, 20]) }

def KNNDTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	Training Code for the distance-weighed k-nearest neighbors method KNN-D 
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model_ox : model for O3 calibration
    
    model_no2 : model for NO2 calibration
    '''
    x = data.iloc[:,2:]
    
    # For O3
    reg = KNeighborsRegressor( n_neighbors = hyperparams1['n_neighbors'], weights = 'distance' )
    y_ox = data.iloc[:,0]
    model_ox = reg.fit( x, y_ox )
    
    # For NO2
    reg = KNeighborsRegressor( n_neighbors = hyperparams2['n_neighbors'], weights = 'distance' )
    y_no2 = data.iloc[:,1]
    model_no2 = reg.fit( x, y_no2 )
    
    return model_ox, model_no2

# Testing Code for KNN-D
def KNNDTest( model, data ):
    x = data.iloc[:,2:]
    
    yPred_ox = model[0].predict(x)
    yPred_no2 = model[1].predict(x)
    
    return np.column_stack(( yPred_ox, yPred_no2 ))

algoTrainList['KNN-D'] = KNNDTrain
algoTestList['KNN-D'] = KNNDTest
algoHyperparamList['KNN-D'] = { 'n_neighbors': np.array([2, 4, 6, 8, 10, 15, 20]) }

def KNNTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	Training Code for the k-nearest neighbors (KNN) method
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model_ox : model for O3 calibration
    
    model_no2 : model for NO2 calibration
    '''
    x = data.iloc[:,2:]
    
    # For O3
    reg = KNeighborsRegressor( n_neighbors = hyperparams1['n_neighbors'] )
    y_ox = data.iloc[:,0]
    model_ox = reg.fit( x, y_ox )
    
    # For NO2
    reg = KNeighborsRegressor( n_neighbors = hyperparams2['n_neighbors'] )
    y_no2 = data.iloc[:,1]
    model_no2 = reg.fit( x, y_no2 )
    
    return model_ox, model_no2

# Testing Code for KNN
def KNNTest( model, data ):
    x = data.iloc[:,2:]
    
    yPred_ox = model[0].predict(x)
    yPred_no2 = model[1].predict(x)    
    
    return np.column_stack((yPred_ox,yPred_no2))
    
algoTrainList['KNN'] = KNNTrain
algoTestList['KNN'] = KNNTest
algoHyperparamList['KNN'] = { 'n_neighbors': np.array([2, 4, 6, 8, 10, 15, 20]) }

def NW_MLTrain( g, i, dataset, data, cv = False ):
    '''
	Training Code for the Nadarya-Watson method with a learnt metric NW(ML)
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model_ox : model for O3 calibration
        
    model_no2 : model for NO2 calibration
        
    X : Training features
    
    y_ox : reference O3 levels
    
    y_no2 : reference NO2 levels
    '''
    maxTrain = 3000
    maxCV = 1000
    n = data.shape[0]
    if cv:
        nTrain = min( n, maxCV )
    else:
        nTrain = min( n, maxTrain )
        
    X = data.iloc[:nTrain, 2:]
    y_ox = data.iloc[:nTrain, 0]
    y_no2 = data.iloc[:nTrain, 1]
    
    # For O3
    mlkr_ox = MLKR()
    temp = mlkr_ox.fit( X, y_ox )
    model_ox = temp.components_
    
    # For NO2
    mlkr_no2 = MLKR()
    temp = mlkr_no2.fit( X, y_no2 )
    model_no2 = temp.components_
    
    return ( model_ox, model_no2, X, y_ox, y_no2 )

# Testing Code for NW(ML)
def NW_MLTest( model, data ):
    Xt = data.iloc[:,2:]
    X = model[2]
    y_ox = model[3]
    y_no2 = model[4]
    
    XnewOx = X.dot(model[0].T)
    XtnewOx = Xt.dot(model[0].T)
    
    Gt = getAllPairsDistances( XtnewOx, XnewOx )
    Gt = np.exp( -Gt )
    norm = np.sum( Gt, axis = 1 )
    yPred_ox = Gt.dot(y_ox)/norm

    XnewNox = X.dot(model[1].T)
    XtnewNox = Xt.dot(model[1].T)

    Gt = getAllPairsDistances( XtnewNox, XnewNox )
    Gt = np.exp( -Gt )
    norm = np.sum( Gt, axis = 1 )
    yPred_no2 = Gt.dot(y_no2)/norm

    return np.column_stack(( yPred_ox, yPred_no2 ))

algoTrainList['NW(ML)'] = NW_MLTrain
algoTestList['NW(ML)'] = NW_MLTest
algoHyperparamList['NW(ML)'] = {}

def NYSTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	Training Code for the Nystroem (NYS) method
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    dict : A dictionary which contains various parameters.
    '''
    nLandmarks = 400
    maxBandwidthVal = 4000
    tol = 1e-12 # Tolerance for small eigenvalues
    n = data.shape[0]
    
    if n < nLandmarks:
        nLandmarks = n
    
    X = data[data.columns[2:]].to_numpy()
    y_ox = data.iloc[:,0].to_numpy()
    y_no2 = data.iloc[:,1].to_numpy()
    
    # Find random landmarks
    randPerm = np.random.permutation(n)
    landmarkIdx = randPerm[:nLandmarks]
    L = X[landmarkIdx,:]
    
    # Find a good bandwidth parameter
    nHyp = min( n, maxBandwidthVal )
    hypIdx1 = randPerm[:nHyp]
    hypIdx2 = randPerm[-nHyp:]
    tempX1 = X[hypIdx1,:]
    tempX2 = X[hypIdx2,:]
    temp = getAllPairsDistances( tempX1, tempX2 )
    gamma = 1/np.quantile( temp, hyperparams1['gammaQuantile'] )
    
    # The landmark-landmark and data-landmark Gram matrices
    Gl = getAllPairsDistances( L, L )
    Gl = np.exp( -gamma * Gl )
    C = getAllPairsDistances( X, L )
    C = np.exp( -gamma * C )
    
    U, S, VT = np.linalg.svd( Gl, hermitian = True )
    S = np.maximum( S, tol )
    normalization = np.dot( U / np.sqrt(S), VT )
    
    Phi = np.dot( C, normalization.T )
    reg_ox = linear_model.Ridge( alpha = hyperparams1['regularizer'], fit_intercept = True )
    reg_ox.fit( Phi, y_ox )
    
    reg_no2 = linear_model.Ridge( alpha = hyperparams2['regularizer'], fit_intercept = True )
    reg_no2.fit( Phi, y_no2 )
    
    return { 'paramsox': reg_ox, 'paramsno2': reg_no2, 'landmarks': L, 'gamma': gamma, 'normalization': normalization, 'randperm': randPerm }

# Testing Code for NYS(Nystroem)
def NYSTest( model, data ):
    gamma = model['gamma']
    X = model['landmarks']
    reg_ox = model['paramsox']
    reg_no2 = model['paramsno2']
    normalization = model['normalization']
    
    Xt = data[data.columns[2:]].to_numpy()
    Gt = getAllPairsDistances( Xt, X )
    Gt = np.exp( - gamma * Gt )
    Phit = np.dot( Gt, normalization.T )
    
    yPred_ox = reg_ox.predict( Phit )
    yPred_no2 = reg_no2.predict( Phit )
    
    return np.column_stack(( yPred_ox, yPred_no2 ))

algoTrainList['NYS'] = NYSTrain
algoTestList['NYS'] = NYSTest
algoHyperparamList['NYS'] = { 'gammaQuantile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'regularizer': np.kron( [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], [1, 2, 5] ) }

def getAllPairsDistances( A, B ):
    squaredNormsA = np.square( np.linalg.norm( A, axis = 1 ) )
    squaredNormsB = np.square( np.linalg.norm( B, axis = 1 ) )
    return squaredNormsA[:, np.newaxis] + squaredNormsB - 2 * A.dot( B.T )

def KRRTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	 Training Code for the Kernel ridge regression (KRR) method
	 
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    dict : A dictionary which contains various parameters
    '''
    n = data.shape[0]
    maxTrain = 4000
    maxCV = 1000
    if cv:
        nTrain = min(n, maxCV)
    else:
        nTrain = min(n, maxTrain)
    
    dataEff = data.iloc[:nTrain, :]
    y_ox = dataEff.iloc[:,0].to_numpy()
    y_no2 = dataEff.iloc[:,1].to_numpy()

    X = dataEff[data.columns[2:]].to_numpy()
    
    # Get the Gram matrix
    G = getAllPairsDistances( X, X )
    # Get the bandwidth parameter
    gamma = 1/np.quantile( G, hyperparams1['gammaQuantile'] )
    
    G = np.exp( -gamma * G )
    
    kreg_ox = KernelRidge( alpha = hyperparams1['regularizer'], kernel = 'precomputed' )
    kreg_ox.fit( G, y_ox )
    kreg_no2 = KernelRidge( alpha = hyperparams2['regularizer'], kernel = 'precomputed' )
    kreg_no2.fit( G, y_no2 )
    
    return { 'paramsox': kreg_ox, 'paramsno2': kreg_no2, 'SV': X, 'gamma': gamma }

# Testing Code for KRR(Kernel ridge regression)
def KRRTest( model, data ):
    gamma = model['gamma']
    X = model['SV']
    kreg_ox = model['paramsox']
    kreg_no2 = model['paramsno2']
    
    Xt = data[data.columns[2:]].to_numpy()
    Gt = getAllPairsDistances( Xt, X )
    Gt = np.exp( - gamma * Gt )
    
    yPred_ox = kreg_ox.predict( Gt )
    yPred_no2 = kreg_no2.predict( Gt )
    
    return np.column_stack(( yPred_ox, yPred_no2 ))

algoTrainList['KRR'] = KRRTrain
algoTestList['KRR'] = KRRTest
algoHyperparamList['KRR'] = { 'gammaQuantile': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'regularizer': np.kron( [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], [1, 2, 5] ) }

def RTTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	Training Code for the Regression Tree (RT) method
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model_ox : model for O3 calibration
        
    model_no2 : model for NO2 calibration
    '''
    reg_ox = DecisionTreeRegressor( min_samples_split = hyperparams1['min_samples_split'] )
    x_ox = data.iloc[:,[2,3,6,7,9]]
    y_ox = data.iloc[:,0]
    model_ox = reg_ox.fit( x_ox, y_ox )
    
    reg_no2 = DecisionTreeRegressor( min_samples_split = hyperparams2['min_samples_split'] )
    x_no2 = data.iloc[:,[2,3,4,5,8]]
    y_no2 = data.iloc[:,1]
    model_no2 = reg_no2.fit( x_no2, y_no2 )
    
    return model_ox, model_no2

# Testing Code for RT(Regression Tree)
def RTTest( model, data ):
    x_ox = data.iloc[:,[2,3,6,7,9]]
    x_no2 = data.iloc[:,[2,3,4,5,8]]
    
    yPred_ox = model[0].predict(x_ox)
    yPred_no2 = model[1].predict(x_no2)    
    
    return np.column_stack(( yPred_ox, yPred_no2 ))
    
algoTrainList['RT'] = RTTrain
algoTestList['RT'] = RTTest
algoHyperparamList['RT'] = { 'min_samples_split': np.array([2, 4, 6, 8, 10, 15, 20]) }

def LS_MINTrain( g, i, dataset, data, cv = False ):
    '''
	Training Code for least squares method with reduced features LS(MIN)
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
                
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model : a model for joint O3 and NO2 calibration
    '''
    reg = linear_model.LinearRegression()
    x = data[data.columns[4:8]]
    y = data[data.columns[:2]]
    model = reg.fit( x, y )
    return model

# Testing Code for LS(MIN)
def LS_MINTest( model, data ):
    x = data[data.columns[4:8]]
    yPred = model.predict(x)
    return yPred

algoTrainList['LS(MIN)'] = LS_MINTrain
algoTestList['LS(MIN)'] = LS_MINTest
algoHyperparamList['LS(MIN)'] = {}

def LSTrain( g, i, dataset, data, cv = False ):
    '''
	Training Code for the least squares (LS) method
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
                
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model : a model for joint O3 and NO2 calibration
    '''
    reg = linear_model.LinearRegression()
    x = data[data.columns[2:]]
    y = data[data.columns[:2]]
    model = reg.fit( x, y )
    return model

# Testing Code for LS
def LSTest( model, data ):
    x = data[data.columns[2:]]
    yPred = model.predict(x)
    return yPred

algoTrainList['LS'] = LSTrain
algoTestList['LS'] = LSTest
algoHyperparamList['LS'] = {}

def lassoTrain( g, i, dataset, data, hyperparams1, hyperparams2, cv = False ):
    '''
	Training Code for the LASSO method
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
        
    hyperparams1 : dictionary
        Dictionary of hyper-paramters for O3.
        
    hyperparams2 : dictionary
        Dictionary of hyper-parameters for NO2.
        
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    model_ox : model for O3 calibration
        
    model_no2 : model for NO2 calibration
    '''
    if cv:
        max_iter = 1e3
    else:
        max_iter = 1e5
    
    reg = linear_model.Lasso( alpha = hyperparams1['alpha'], max_iter = max_iter, selection = 'random' )
    x_ox = data.iloc[:,[2,3,6,7,9]]
    y_ox = data.iloc[:,0]
    model_ox = reg.fit( x_ox, y_ox )
    
    reg = linear_model.Lasso( alpha = hyperparams2['alpha'], max_iter = max_iter, selection = 'random' )    
    x_no2 = data.iloc[:,[2,3,4,5,8]]
    y_no2 = data.iloc[:,1]
    model_no2 = reg.fit( x_no2, y_no2 )
    
    return model_ox, model_no2

# Testing Code for LASSO
def lassoTest( model, data ):
    x_ox = data.iloc[:,[2,3,6,7,9]]
    x_no2 = data.iloc[:,[2,3,4,5,8]]
    
    yPred_ox = model[0].predict(x_ox)
    yPred_no2 = model[1].predict(x_no2)    
    
    return np.column_stack(( yPred_ox, yPred_no2 ))
    
algoTrainList['LASSO'] = lassoTrain
algoTestList['LASSO'] = lassoTest
algoHyperparamList['LASSO'] = { 'alpha': np.kron( [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], [1, 2, 5] ) }

# Preprocess data before feeding it into the Alphasense calibration models
def AS_preproc(data):
    num_rows, num_cols = data.shape
    no2 = pd.read_csv( 'NO2B43F.csv', index_col = 0 ).to_numpy()
    ox = pd.read_csv( 'OXB431.csv', index_col = 0 ).to_numpy()
    
    ai = [3,4,5,6,7,8,0,1,2]
    temp = data[data.columns[2]].to_numpy()
    tempi = temp//10
    tempi = tempi.astype(int)
    
    no2_const = np.zeros((num_rows,4))
    ox_const = np.zeros((num_rows,4))
    
    for i in range(num_rows):
        no2_const[i] = no2[:,ai[tempi[i]+1]] + (no2[:,ai[tempi[i]]] - no2[:,ai[tempi[i]+1]]) * (temp[i] - tempi[i]*10) / 10
        ox_const[i] = ox[:,ai[tempi[i]+1]] + (ox[:,ai[tempi[i]]] - ox[:,ai[tempi[i]+1]]) * (temp[i] - tempi[i]*10) / 10
    
    return no2_const, ox_const

def AS1Train( g, i, dataset, data, cv = False ):
    '''
	An implementation of the Alphasense calibration method AS1
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
                
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    pred : dataframe
        Predictions on test data (there is no training required for this calibration technique)
    '''
    no2x = data[data.columns[4:6]].to_numpy()
    no2y = data[data.columns[1]].to_numpy()
    o3x = data[data.columns[6:8]].to_numpy()
    o3y = data[data.columns[0]].to_numpy()
    
    no2_const, ox_const = AS_preproc(data)
    
    pred_no2_f1, pred_ox_f1 = ac.AS1_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, ox_const )
    
    pred_f1 = np.column_stack(( pred_ox_f1, pred_no2_f1 ))
    pred = pd.DataFrame( data = pred_f1, index = data.index, columns = g.columns_pred )
    
    return pred

algoTrainList['AS1'] = AS1Train
algoTestList['AS1'] = AS1Train
algoHyperparamList['AS1'] = {}

def AS2Train( g, i, dataset, data, cv = False ):
    '''
	An implementation of the Alphasense calibration method AS2
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
                
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    pred : dataframe
        Predictions on test data (there is no training required for this calibration technique)
    '''
    no2x = data[data.columns[4:6]].to_numpy()
    no2y = data[data.columns[1]].to_numpy()
    o3x = data[data.columns[6:8]].to_numpy()
    o3y = data[data.columns[0]].to_numpy()
    
    no2_const, ox_const = AS_preproc(data)
    
    pred_no2_f2, pred_ox_f2 = ac.AS2_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, ox_const )
    
    pred_f2 = np.column_stack(( pred_ox_f2, pred_no2_f2 ))
    pred = pd.DataFrame( data = pred_f2, index = data.index, columns = g.columns_pred )
    
    return pred

algoTrainList['AS2'] = AS2Train
algoTestList['AS2'] = AS2Train
algoHyperparamList['AS2'] = {}


def AS3Train( g, i, dataset, data, cv = False ):
    '''
	An implementation of the Alphasense calibration method AS3
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
                
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    pred : dataframe
        Predictions on test data (there is no training required for this calibration technique)
    '''
    no2x = data[data.columns[4:6]].to_numpy()
    no2y = data[data.columns[1]].to_numpy()
    o3x = data[data.columns[6:8]].to_numpy()
    o3y = data[data.columns[0]].to_numpy()
    
    no2_const, ox_const = AS_preproc(data)
        
    pred_no2_f3, pred_ox_f3 = ac.AS3_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, ox_const )
    
    pred_f3 = np.column_stack(( pred_ox_f3, pred_no2_f3 ))
    pred = pd.DataFrame( data = pred_f3, index = data.index, columns = g.columns_pred )
    
    return pred

algoTrainList['AS3'] = AS3Train
algoTestList['AS3'] = AS3Train
algoHyperparamList['AS3'] = {}


def AS4Train( g, i, dataset, data, cv = False ):
    '''
	An implementation of the Alphasense calibration method AS4
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables
        
    i : int
        Index of the dataset in datasetList.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    data : dataframe
        Training data.
                
    cv : bool, optional
        Used for cross-validation. The default is False.

    Returns
    -------
    pred : dataframe
        Predictions on test data (there is no training required for this calibration technique)
    '''
    no2x = data[data.columns[4:6]].to_numpy()
    no2y = data[data.columns[1]].to_numpy()
    o3x = data[data.columns[6:8]].to_numpy()
    o3y = data[data.columns[0]].to_numpy()
    
    no2_const, ox_const = AS_preproc(data)
    
    pred_no2_f4, pred_ox_f4 = ac.AS4_compute( g, dataset, no2x, no2y, o3x, o3y, no2_const, ox_const )
    
    pred_f4 = np.column_stack(( pred_ox_f4, pred_no2_f4 ))
    pred = pd.DataFrame( data = pred_f4, index = data.index, columns = g.columns_pred )
    
    return pred

algoTrainList['AS4'] = AS4Train
algoTestList['AS4'] = AS4Train
algoHyperparamList['AS4'] = {}

def createSplits( g, split, datasetName ):
    '''
	Generate random permutations that enable train-test splits to be created
	Warning: invoking this method will overwrite existing permutations
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables.
        
    split : int
        The split number.
        
    datasetName : string
        Name of the dataset, for example DD1(Jun).csv
    '''
    # Reading the dataset
    raw = pd.read_csv( g.datasetPath + datasetName )
    
    # Selecting only those rows that are valid
    cleanSubset = raw[raw['Valid'] == 1]
    n = len( cleanSubset )
    
    # Generating a random permutation for the given dataset
    perm = np.random.permutation( n )
    
    # Storing the permutation
    np.savetxt(g.permPath + 'perm_' + str(split) + '/' + datasetName[:-4] + '_perm', perm, fmt = '%d' )

def loadSplits( g, split, datasetName, time = False, valid = False, trainFraction = 0.7 ):
    '''
	Load a certain train-test split of a given dataset
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables.
        
    split : int
        The split number.
        
    datasetName : string
        Name of the dataset, for example DD1(Jun).csv
        
    time : bool, optional
        If True, the column 'Time' will not be removed from the dataset. The default is False.
        
    valid : bool, optional
        If True, the column 'Valid' will not be removed from the dataset. The default is False.
        
    trainFraction : float, optional
        It denotes the fraction of the data to be used for training. The default is 0.7.

    Returns
    -------
    data : dictionary
        Train and test data
    '''
    raw = pd.read_csv( g.datasetPath + datasetName )
    raw = raw[raw['Valid'] == 1]
    raw = raw[g.featureList]
    # Create two new features, one containing no2op1 - no2op2, and the other containing oxop1 - oxop2
    raw['no2op1 - no2op2(mV)'] = raw['no2op1(mV)'] - raw['no2op2(mV)']
    raw['o3op1 - o3op2(mV)'] = raw['o3op1(mV)'] - raw['o3op2(mV)']
    
    # Replacing NULL values with zero
    raw.fillna( 0, inplace = True )
    
    # Loading permutation for dataset
    perm = np.loadtxt( g.permPath + 'perm_' + str(split) + '/' + datasetName[:-4] + '_perm' )
    n = len( perm )
    nTrain = int( trainFraction * n )
    idxTrain = perm[:nTrain]
    idxTest = perm[nTrain:]
    
    # Remove column 'Time'
    if time == False:
        raw.drop( ['Time'], axis = 1, inplace = True )
    
    # Remove column 'Valid' 
    if valid == False:
        raw.drop( ['Valid'], axis = 1, inplace = True )
    
    data = {}
    data['train'] = raw.iloc[idxTrain] # Training Data
    data['test'] = raw.iloc[idxTest] # Testing Data
    return data

def trainAlgo( g, split, dataset, algoName, data, trainFrac = 0.7, maxTrainForVal = 5000 ):
    '''
	Invoke the training routine for a certain algorithm and perform hyper-parameter tuning
	
    Parameters
    ----------
    g : object
        Object of class My_Globals in globals.py that contains all the global variables.
        
    split : int
        The split number.
        
    dataset : string
        Name of the dataset, for example DD1(Jun).csv
        
    algoName : string
        Name of the algorithm for which the model will be trained
        
    data : dataframe
        Training data.
        
    trainFrac : float, optional
        It denotes the fraction of the data to be used for training. The default is 0.7.
        
    maxTrainForVal : int, optional
        Maximum number of samples to be used for cross-validation. The default is 5000.

    Returns
    -------
    dict : A dictionary of model and hyper-parameters
    '''
    train = algoTrainList[algoName]
    test = algoTestList[algoName]
    hyperparamList = algoHyperparamList[algoName]
    
    n = data.shape[0]
    nTrain = min( int(trainFrac * n), maxTrainForVal )
    
    hyperparamNames = hyperparamList.keys()
    bestHyperparams = dict(zip( hyperparamNames, np.zeros(len(hyperparamNames)) ))
    bestPerf = np.inf
    if len(hyperparamNames) > 0:
        for hyperparamVals1 in itt.product( *hyperparamList.values() ):
            for hyperparamVals2 in itt.product( *hyperparamList.values() ):
                hyperparams1 = dict(zip( hyperparamNames, hyperparamVals1 ))
                hyperparams2 = dict(zip( hyperparamNames, hyperparamVals2 ))
                params = train( g, split, dataset, data.iloc[:nTrain, :], hyperparams1, hyperparams2, cv = True )
                perf = np.sum(np.abs( test( params, data.iloc[nTrain:, :] ) - data.iloc[nTrain:,:2].to_numpy() ))
                if perf < bestPerf:
                    bestPerf = perf
                    bestHyperparams = [ hyperparams1, hyperparams2 ]
            break
        
        hyperparams2 = bestHyperparams[1]
        for hyperparamVals1 in itt.product( *hyperparamList.values() ):
            hyperparams1 = dict(zip( hyperparamNames, hyperparamVals1 ))
            params = train( g, split, dataset, data.iloc[:nTrain, :], hyperparams1, hyperparams2, cv = True )
            perf = np.sum(np.abs( test( params, data.iloc[nTrain:, :] ) - data.iloc[nTrain:,:2].to_numpy() ))
            if perf < bestPerf:
                bestPerf = perf
                bestHyperparams = [ hyperparams1, hyperparams2 ]
            
        return { 'model': train( g, split, dataset, data, bestHyperparams[0], bestHyperparams[1] ), 'hyperparams': bestHyperparams }
    else:
        return { 'model': train( g, split, dataset, data ), 'hyperparams': None }

def testAlgo( algoName, model, data ):
    '''
	Return the predictions of a certain algorithm on the given data
	
    Parameters
    ----------
    algoName : string
        Name of the algorithm for testing the model.
        
    model : dictionary
        Model which is to be tested.
        
    data : dataframe
        Test data.
    '''
    algo = algoTestList[algoName]
    
    return algo( model['model'], data )

