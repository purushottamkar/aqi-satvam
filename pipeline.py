import utils
import pickle
import os
import numpy as np
import pandas as pd
import copy
import loss_func
from scipy import stats as st

def load_data( g, new_permutations = False ):
    '''
    Return training and test splits
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains all the global variables.
    
    new_permutations : bool
        Should new permutations be generated or should existing ones be used?
        Existing permutations will be picked up from the directory specified in the My_Globals object g
        Default value is False
        
    Returns
    -------
    datasets : dictionary
        Dictionary of training and testing data for all splits
        
    datasets_norm : dictionary
        Dictionary of normalised training and testing data for all splits.
    
    If you wish to reproduce the results mentioned in the paper, please use the permutations supplied
    with the repository and set new_permutations = False. 
    
    However, if you wish to start afresh, using new_permutations = True will cause fresh permutations to
    be generated for the given datasets. These new permutations will then overwrite old ones. Minor differences
    are to be expected in the results if fresh permutations are used to conduct the experiments.    
    '''
    
    if new_permutations == True:
        for split in range( g.numSplits ):
            
            if os.path.isdir( g.permPath + 'perm_' + str(split) ) == 0:
                    os.mkdir( g.permPath + 'perm_' + str(split) )
            
            for dataset in g.datasetList:
                utils.createSplits( g, split, dataset )
    
    
    # Load a dataset and perform preprocessing
    
    datasets = [dict() for i in range(g.numSplits)]

    # datasets_norm contains normalised datasets.
    datasets_norm = [dict() for i in range(g.numSplits)]
    
    for split in range(g.numSplits):
        for dataset in g.datasetList:
            data = utils.loadSplits( g, split, dataset, trainFraction = 0.7 )
            datasets[split][dataset] = copy.deepcopy(data)
            
            # Normalising the data
            mu = np.mean( data['train'], axis = 0 )    
            std = np.std( data['train'], axis = 0 )
            mu[:2] = 0
            std[:2] = 1
            
            data['train'] = ( data['train'] - mu ) / std
            data['test'] = ( data['test'] - mu ) / std
            data['mean'] = mu
            data['std'] = std
            datasets_norm[split][dataset] = copy.deepcopy(data)
    
    return datasets, datasets_norm

def doExperiment( g, algoList, datasets, datasets_norm, ASNorm = True ):
    '''
    Run experiments with multiple algorithms on multiple datasets
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains all the global variables.
        
    algoList : list of strings
        List of names of algorithms for which the models are to be trained.

    datasets : dictionary
        Dictionary of training and testing data for all splits
        
    datasets_norm : dictionary
        Dictionary of normalised training and testing data for all splits.
        
    ASNorm : bool, optional
        If true, normalised data will be used for the Alphasense calibration models AS1, AS2, AS3 and AS4,
        otherwise un-normalised data will be used
    '''
    
    # Start Training Models
    for split in range(g.numSplits):
        print()
        print('--------------------------------------------------------------')
        print('Training for split:', split)
    
        for dataset in g.datasetList:
            print()
            print('Dataset:', dataset)
    
            for algo in algoList:
                print(algo, end=' ')
                
                if algo[:-1] == 'AS':
                    if ASNorm == True:
                        test = utils.trainAlgo( g, split, dataset, algo, datasets_norm[split][dataset]['test'] )
                    else:
                        test = utils.trainAlgo( g, split, dataset, algo, datasets[split][dataset]['test'] )
                    testPrediction = test['model']
                    
                    fileName = str(split) + '_' + algo + '_' + dataset
                    testErrors = testPrediction.to_numpy() - datasets[split][dataset]['test'].iloc[:,:2].to_numpy()
                    testErrors = pd.DataFrame( data = testErrors, index = datasets[split][dataset]['test'].index, columns = g.columns_pred )
                    testErrors.to_csv( g.testErrorsPath + fileName[:-4] + '_Errors.csv' )
                    testPrediction.to_csv( g.testPredictionsPath + fileName[:-4] + '_Predictions.csv' )
                
                else:
                    model = utils.trainAlgo( g, split, dataset, algo, datasets_norm[split][dataset]['train'] )
                    model['mean'] = datasets_norm[split][dataset]['mean']
                    model['std'] = datasets_norm[split][dataset]['std']
                    testPrediction = utils.testAlgo( algo, model, datasets_norm[split][dataset]['test']  )
                    
                    fileName = str(split) + '_' + algo + '_' + dataset
                    pickle.dump( model, open( g.modelPath + fileName[:-4] + '_model', 'wb' ) )
                    
                    testPrediction = pd.DataFrame( data = testPrediction, index = datasets_norm[split][dataset]['test'].index, columns = g.columns_pred )
                    testPrediction.to_csv( g.testPredictionsPath + fileName[:-4] + '_Predictions.csv')
                    
                    testErrors = testPrediction.to_numpy() - datasets_norm[split][dataset]['test'].iloc[:,:2].to_numpy()
                    testErrors = pd.DataFrame( data = testErrors, index = datasets_norm[split][dataset]['test'].index, columns = g.columns_pred )
                    testErrors.to_csv( g.testErrorsPath + fileName[:-4] + '_Errors.csv' )
            print()

def transferTesting( g, algoList, datasetTrainList, datasetTestList, datasets, datasets_norm ):
    '''
    Perform transfer learning experiments
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains all the global variables.
        
    algoList : list of strings
        List of names of algorithms for which the models have to be trained.
        
    datasetTrainList : list of strings
        List of datasets whose models will be used for testing on datasetTestList
        
    datasetTestList : list of strings
        List of datasets upon which the models of datasetTrainList will be tested.
        
    datasets : dictionary
        Dictionary of training and testing data for all splits
        
    datasets_norm : dictionary
        Dictionary of normalised training and testing data for all splits.
    '''
    
    for split in range(g.numSplits):
        print()
        print('--------------------------------------------------------------')
        print('Testing for split:', split)
        
        for datasetTrain, datasetTest in zip( datasetTrainList, datasetTestList ):
            print()
            print('trainDataset:', datasetTrain, 'testDataset:', datasetTest)
            
            for algo in algoList:
                print(algo, end=' ')
                
                fileName = str(split) + '_' + algo + '_' + datasetTrain[:-4]
                model = pickle.load( open( g.modelPath + fileName + '_model', 'rb' ) )
                mu = model['mean']
                std = model['std']
                
                data = ( datasets[split][datasetTest]['test'] - mu ) / std
                
                testPrediction = utils.testAlgo( algo, model, data  )
                
                fileName = str(split) + '_' + algo + '_' + datasetTrain[:-4] + '_' + datasetTest[:-4]
                testPrediction = pd.DataFrame( data = testPrediction, index = data.index, columns = g.columns_pred )
                testPrediction.to_csv( g.testPredictionsPath + fileName + '_Predictions.csv' )
                
                testErrors = testPrediction.to_numpy() - data.iloc[:,:2].to_numpy()
                testErrors = pd.DataFrame( data = testErrors, index = data.index, columns = g.columns_pred )
                testErrors.to_csv( g.testErrorsPath + fileName + '_Errors.csv' )
            print()

def gen_errorMetrics( g, algoList, datasets, datasets_norm ): 
    '''
    Calculate error metrics for regular experiments
    
    Parameters
    ----------
    g : object
        Object of class My_Globals taht contains all the global variables.
        
    algoList : list of strings
        List of names of algorithms for which the models have to be trained.

    datasets : dictionary
        Dictionary of training and testing data for all splits
        
    datasets_norm : dictionary
        Dictionary of normalised training and testing data for all splits.
    '''              
    
    # Calculating MAE, RMSE, MAPE(%), R2
    for split in range(g.numSplits):
        for algo in algoList:
            err_no2 = np.zeros( ( len(g.datasetList), len(g.columns_err) ) )
            err_o3 = np.zeros( ( len(g.datasetList), len(g.columns_err) ) )
            i = 0
            for dataset in g.datasetList:
                fileName = str(split) + '_' + algo + '_' + dataset[:-4]
                
                yPred = pd.read_csv( g.testPredictionsPath + fileName + '_Predictions.csv',index_col=0 ).to_numpy()
                y = (datasets_norm[split][dataset]['test']).to_numpy()
                
                err_o3[i, 0] = loss_func.mae(y[:,0], yPred[:,0])
                err_o3[i, 1] = loss_func.rmse(y[:,0], yPred[:,0])
                err_o3[i, 2] = loss_func.mape(y[:,0], yPred[:,0])
                err_o3[i, 3] = loss_func.coeff_r2(y[:,0], yPred[:,0])
    
                err_no2[i, 0] = loss_func.mae(y[:,1], yPred[:,1])
                err_no2[i, 1] = loss_func.rmse(y[:,1], yPred[:,1])
                err_no2[i, 2] = loss_func.mape(y[:,1], yPred[:,1])
                err_no2[i, 3] = loss_func.coeff_r2(y[:,1], yPred[:,1])
                i += 1
                
            err_o3 = pd.DataFrame( data = err_o3, index = g.datasetList, columns = g.columns_err )
            err_o3.to_csv( g.errorMetricsPath + str(split) + '_' + algo + '_err_o3.csv')
    
            err_no2 = pd.DataFrame( data = err_no2, index = g.datasetList, columns = g.columns_err )
            err_no2.to_csv( g.errorMetricsPath + str(split) + '_' + algo + '_err_no2.csv')

def gen_transferErrorMetrics( g, algoList, datasetTrainList, datasetTestList, datasetList, datasets, datasets_norm ):
    '''
    Calculate error metrics for transfer learning experiments
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains all the global variables.
        
    algoList : list of strings
        List of names of algorithms for which the models have to be trained.
        
    datasetTrainList : list of strings
        List of datasets whose models will be used for testing on datasetTestList
        
    datasetTestList : list of strings
        List of datasets upon which the models of datasetTrainList will be tested.
    
    datasetList: list of strings
        Labels for combined list of datasetTrainList and datasetTestList, for example 'DD1(Jun) -> DD1(Oct)'
        
    datasets : dictionary
        Dictionary of training and testing data for all splits
        
    datasets_norm : dictionary
        Dictionary of normalised training and testing data for all splits.

    '''
    
    for split in range(g.numSplits):
        for algo in algoList:
            err_no2 = np.zeros( ( len(datasetList), len(g.columns_err) ) )
            err_o3 = np.zeros( ( len(datasetList), len(g.columns_err) ) )
            i = 0
            for datasetTrain, datasetTest in zip( datasetTrainList, datasetTestList ):
                fileName = str(split) + '_' + algo + '_' + datasetTrain[:-4] + '_' + datasetTest[:-4]
                
                yPred = pd.read_csv( g.testPredictionsPath + fileName + '_Predictions.csv', index_col = 0 ).to_numpy()
                y = (datasets[split][datasetTest]['test']).to_numpy()
                
                err_o3[i, 0] = loss_func.mae(y[:,0], yPred[:,0])
                err_o3[i, 1] = loss_func.rmse(y[:,0], yPred[:,0])
                err_o3[i, 2] = loss_func.mape(y[:,0], yPred[:,0])
                err_o3[i, 3] = loss_func.coeff_r2(y[:,0], yPred[:,0])
    
                err_no2[i, 0] = loss_func.mae(y[:,1], yPred[:,1])
                err_no2[i, 1] = loss_func.rmse(y[:,1], yPred[:,1])
                err_no2[i, 2] = loss_func.mape(y[:,1], yPred[:,1])
                err_no2[i, 3] = loss_func.coeff_r2(y[:,1], yPred[:,1])
                i += 1
                
            err_o3 = pd.DataFrame( data = err_o3, index = datasetList, columns = g.columns_err )
            err_o3.to_csv( g.transferErrorMetricsPath + str(split) + '_' + algo + '_err_o3.csv')
    
            err_no2 = pd.DataFrame( data = err_no2, index = datasetList, columns = g.columns_err )
            err_no2.to_csv( g.transferErrorMetricsPath + str(split) + '_' + algo + '_err_no2.csv')
     
def gen_avgErrorMetrics( g, algoList, datasets, datasets_norm ):      
    '''
    Calculate average error metrics over all the splits  
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains all the global variables.
        
    algoList : list of strings
        List of names of algorithms for which the models have to be trained.

    datasets : dictionary
        Dictionary of training and testing data for all splits
        
    datasets_norm : dictionary
        Dictionary of normalised training and testing data for all splits.
    '''
    
    assert g.numSplits > 1, "Number of splits must be greater than 1!"

    for algo in algoList:
        o3, no2 = [], []
        
        for split in range( g.numSplits ):
            fileName = str(split) + '_' + algo
            o3_error = pd.read_csv( g.errorMetricsPath + fileName + '_err_o3.csv', index_col = 0 )
            no2_error = pd.read_csv( g.errorMetricsPath + fileName + '_err_no2.csv', index_col = 0 )
            
            o3.append(o3_error)
            no2.append(no2_error)
    
        o3 = pd.concat(o3)
        no2 = pd.concat(no2)
        
        o3_mean, no2_mean = np.zeros( ( len(g.datasetList), len(g.columns_err) ) ), np.zeros( ( len(g.datasetList), len(g.columns_err) ) )
        o3_std, no2_std = np.zeros( ( len(g.datasetList), len(g.columns_err) ) ), np.zeros( ( len(g.datasetList), len(g.columns_err) ) )
        j = 0
        
        for dataset in g.datasetList:
            
            # Calculating mean and std over all the error metrics and datasets
            err_o3 = o3.loc[dataset]
            mean_o3 = err_o3.mean().to_numpy()
            std_o3 = err_o3.std().to_numpy()
            mean_o3.resize( 1, len(g.columns_err) )
            std_o3.resize( 1, len(g.columns_err) )
            
            err_no2 = no2.loc[dataset]
            mean_no2 = err_no2.mean().to_numpy()
            std_no2 = err_no2.std().to_numpy()
            mean_no2.resize( 1, len(g.columns_err) )
            std_no2.resize( 1, len(g.columns_err) )
            
            o3_mean[j] = mean_o3
            no2_mean[j] = mean_no2
            
            o3_std[j] = std_o3
            no2_std[j] = std_no2
            
            j += 1
            
        mean_o3 = pd.DataFrame( data = o3_mean, index = g.datasetList, columns = g.columns_err )
        mean_no2 = pd.DataFrame( data = no2_mean, index = g.datasetList, columns = g.columns_err )
        
        std_o3 = pd.DataFrame( data = o3_std, index = g.datasetList, columns = g.columns_err )
        std_no2 = pd.DataFrame( data = no2_std, index = g.datasetList, columns = g.columns_err )
            
        mean_o3.to_csv( g.errorMetricsPath + algo + '_mean_o3.csv' )
        mean_no2.to_csv( g.errorMetricsPath + algo + '_mean_no2.csv' )
        
        std_o3.to_csv( g.errorMetricsPath + algo + '_std_o3.csv' )
        std_no2.to_csv( g.errorMetricsPath + algo + '_std_no2.csv' )
        
    # For convenience, generating files that contain both mean and std
    gasList = [ 'o3', 'no2' ]
    
    digits = 2
    
    for gas in gasList:
        
        for algo in algoList:
            final = pd.DataFrame( columns = g.columns_err, index = g.datasetList )
            mean = pd.read_csv( g.errorMetricsPath + algo + '_mean_' + gas + '.csv', index_col = 0 )
            std = pd.read_csv( g.errorMetricsPath + algo + '_std_' + gas + '.csv', index_col = 0 )
    
            for dataset in g.datasetList:
                for metric in g.columns_err :
                    if metric == "R2":
                        final.loc[ dataset ][ metric ] = str( np.round( mean.loc[ dataset ][ metric ], digits + 1 ) ) + '\u00B1' + str( np.round( std.loc[ dataset ][ metric ], digits + 1 ) )
                    else:
                        final.loc[ dataset ][ metric ] = str( np.round( mean.loc[ dataset ][ metric ], digits ) ) + '\u00B1' + str( np.round( std.loc[ dataset ][ metric ], digits ) )
    
            final.to_csv( g.errorMetricsPath + algo + '_mean_std_' + gas + '.csv' )

def gen_winMatrix( g, algoList, datasets, datasets_norm ):
    '''
    Find win mtrices that rank all algorithms for each dataset
    
    Parameters
    ----------
    g : object
        Object of class My_Globals that contains all the global variables.
        
    algoList : list of strings
        List of names of algorithms for which the models have to be trained.

    datasets : dictionary
        Dictionary of training and testing data for all splits
        
    datasets_norm : dictionary
        Dictionary of normalised training and testing data for all splits.
    '''
    
    alpha = 0.05 # At what significance level are we operating?
    gasList = [ 'o3', 'no2' ]
    numAlgos = len(algoList)
    
    for gas in gasList:
        
        gasLabel = 'NO2(ppb)'
        if gas == 'o3':
            gasLabel = 'O3(ppb)'
        
        for dataset in g.datasetList:
            winMatrixThisDataset = np.zeros( (numAlgos, numAlgos) )
            
            for split in range( g.numSplits ):
                winMatrixThisSplit = np.ones( (numAlgos, numAlgos) ) * 2 # Dummy initial values
                
                for algo1 in algoList:
                    for algo2 in algoList:
                        if algo2 == algo1:
                            winMatrixThisSplit[algoList.index(algo1), algoList.index(algo2)] = 0
                            winMatrixThisSplit[algoList.index(algo2), algoList.index(algo1)] = 0
                            continue
                    
                        if winMatrixThisSplit[algoList.index(algo1), algoList.index(algo2)] != 2:
                            continue
                        
                        if winMatrixThisSplit[algoList.index(algo2), algoList.index(algo1)] != 2:
                            continue
    
                        fileName = g.testErrorsPath + str(split) + '_'
                        
                        # List of MAE values for algo1 on this dataset and this split
                        MAE1 = pd.read_csv( fileName + algo1 + '_' + dataset[:-4] + '_Errors.csv', index_col = 0 )
                        MAE1 = np.abs( MAE1[gasLabel].to_numpy() )
                        
                        # List of MAE values for algo2 on this dataset and this split
                        MAE2 = pd.read_csv( fileName + algo2 + '_' + dataset[:-4] + '_Errors.csv', index_col = 0 )
                        MAE2 = np.abs( MAE2[gasLabel].to_numpy() )
                        
                        (t, p) = st.wilcoxon( MAE1, MAE2, alternative = "greater" )
    
                        if p <= alpha: # This means that MAE1 >> MAE2 i.e. win for algo2
                            winMatrixThisSplit[algoList.index(algo1), algoList.index(algo2)] = -1
                            winMatrixThisSplit[algoList.index(algo2), algoList.index(algo1)] = 1
                        
                        elif p >= 1 - alpha: # This means that MAE1 << MAE2 i.e. win for algo1
                            winMatrixThisSplit[algoList.index(algo1), algoList.index(algo2)] = 1
                            winMatrixThisSplit[algoList.index(algo2), algoList.index(algo1)] = -1
                        
                        else: # Draw
                            winMatrixThisSplit[algoList.index(algo1), algoList.index(algo2)] = 0
                            winMatrixThisSplit[algoList.index(algo2), algoList.index(algo1)] = 0
    
                winMatrixThisSplit[ winMatrixThisSplit < 0 ] = 0
                winMatrixThisDataset += winMatrixThisSplit
                
            winMatrixThisDataset /= g.numSplits
            winMatrixThisDataset = pd.DataFrame( data = winMatrixThisDataset, index = algoList, columns = algoList )
            winMatrixThisDataset.to_csv( g.winMatrixPath + dataset[:-4] + '_' + gas + '.csv' )
    
    # Generating rankings over all the datasets
    for gas in gasList:
        winMatrix = np.zeros( (numAlgos, numAlgos) )
        for dataset in g.datasetList:
            winMatrixThisDataset = pd.read_csv( g.winMatrixPath + dataset[:-4] + '_' + gas + '.csv', index_col=0 ).to_numpy()
            winMatrix += winMatrixThisDataset
    
        winMatrix /= len(g.datasetList)
        winMatrix = pd.DataFrame( data = winMatrix, index = algoList, columns = algoList )
        winMatrix.to_csv( g.winMatrixPath + gas + '.csv' )

