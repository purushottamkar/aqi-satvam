import numpy as np

# Root mean squared error
def rmse( y_true, y_pred ):
  rmse = np.sqrt(np.sum(np.square( y_pred - y_true )) / np.size(y_true))
  return rmse

# Mean absolute percentage error
def mape( y_true, y_pred ):
  mape = np.sum(np.abs(( y_pred - y_true ) / y_true))
  mape = mape * 100 / np.size(y_true)
  return mape

# Mean absolute error
def mae( y_true, y_pred ):
  mae = np.sum(np.abs( y_true - y_pred ))
  mae = mae / np.size(y_true)
  return mae

# Pearson's coefficient
def pearson( y_true, y_pred ):
  N = np.size(y_true)
  num = np.sum(y_true * y_pred) * N - np.sum(y_true) * np.sum(y_pred)
  den = (N * np.sum(np.square(y_true))) - np.square(np.sum(y_true))
  den = den * ((N * np.sum(np.square(y_pred))) - np.square(np.sum(y_pred)))
  den = np.sqrt(den)
  if den == 0:
      return 0
  else :
      return (num/den)

# R2 coefficient
def coeff_r2( y_true, y_pred ):
  mu = np.mean(y_true)
  ss_res = np.sum(np.square(y_true - y_pred))
  ss_tot = np.sum(np.square(y_true - mu))
  r2 = 1 - (ss_res / ss_tot)
  return r2
