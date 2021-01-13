import os

class My_Globals:
    
    def __init__(self):
        
        # Dataset path
        self.datasetPath = 'data/'
        
        # Derived dataset path ( DD1(Jun)-AVG5, etc.. )
        self.derivedDatasetPath = 'derivedData/'
        if os.path.isdir( self.derivedDatasetPath ) == 0:
            os.mkdir( self.derivedDatasetPath )
        
        # Model path
        self.modelPath = 'models/'
        if os.path.isdir( self.modelPath ) == 0:
            os.mkdir( self.modelPath )
        
        # Permutations path
        self.permPath = 'perm/'
        if os.path.isdir( self.permPath ) == 0:
            os.mkdir( self.permPath )
            
        # Test predictions path
        self.testPredictionsPath = 'testPredictions/'
        if os.path.isdir( self.testPredictionsPath ) == 0:
            os.mkdir( self.testPredictionsPath )
            
        # Test errors path
        self.testErrorsPath = 'testErrors/'
        if os.path.isdir( self.testErrorsPath ) == 0:
            os.mkdir( self.testErrorsPath )
            
        # Error metrics ( MAE, RMSE, MAPE(%), R2, PEARSON ) path
        self.errorMetricsPath = 'errorMetrics/'
        if os.path.isdir( self.errorMetricsPath ) == 0:
            os.mkdir( self.errorMetricsPath )
            
        # Win-matrix path (rank test)
        self.winMatrixPath = 'winMatrix/'
        if os.path.isdir( self.winMatrixPath ) == 0:
            os.mkdir( self.winMatrixPath )
        
        # Derived dataset error metrics ( MAE, RMSE, MAPE(%), R2, PEARSON ) path
        self.derivedErrorMetricsPath = 'derivedErrorMetrics/'
        if os.path.isdir( self.derivedErrorMetricsPath ) == 0:
            os.mkdir( self.derivedErrorMetricsPath )
            
        # Derived win-matrix path (rank test)
        self.derivedWinMatrixPath = 'derivedWinMatrix/'
        if os.path.isdir( self.derivedWinMatrixPath ) == 0:
            os.mkdir( self.derivedWinMatrixPath )
        
        # Transfer experiment error metrics path
        self.transferErrorMetricsPath = 'transferErrorMetrics/'
        if os.path.isdir( self.transferErrorMetricsPath ) == 0:
            os.mkdir( self.transferErrorMetricsPath )
                    
        # Number of splits
        self.numSplits = 10
        
        # List of features in each dataset
        self.featureList = ['Time','Ref. O3(ppb)','Ref. NO2(ppb)','Temp(C)','RH(%)','no2op1(mV)','no2op2(mV)','o3op1(mV)',
                'o3op2(mV)', 'Valid']
        
        # List of datasets
        self.datasetList = os.listdir( self.datasetPath )
        self.datasetList.sort()
        
        # List of features to be predicted
        self.columns_pred = ['O3(ppb)', 'NO2(ppb)']
        
        # List of error metrics
        self.columns_err = ['MAE','RMSE','MAPE(%)','R2']
        
        # Constants for the Alphasense calibration models AS1, AS2, AS3, AS4
        self.AS_const = {}
        self.AS_const['DD1(Jun).csv'] = {'NO2_WE_0T' : 225, 'NO2_AE_0T' : 212, 'NO2_WE_0E' : 236, 
                                     'NO2_AE_0E' : 216, 'SENSITIVITY_NO2' : 0.282,
                                     'O3_WE_0T' : 232, 'O3_AE_0T' : 245, 'O3_WE_0E' : 241,
                                     'O3_AE_0E' : 252, 'SENSITIVITY_O3' : 0.331}
        
        self.AS_const['DD1(Oct).csv'] = {'NO2_WE_0T' : 225, 'NO2_AE_0T' : 212, 'NO2_WE_0E' : 236, 
                                     'NO2_AE_0E' : 216, 'SENSITIVITY_NO2' : 0.282,
                                     'O3_WE_0T' : 232, 'O3_AE_0T' : 245, 'O3_WE_0E' : 241,
                                     'O3_AE_0E' : 252, 'SENSITIVITY_O3' : 0.331}
        
