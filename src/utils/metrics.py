import numpy as np
def mae(y,yhat):  return float(np.mean(np.abs(yhat-y)))
def mse(y,yhat):  return float(np.mean((yhat-y)**2))
def rmse(y,yhat): return float(np.sqrt(mse(y,yhat)))
def smape(y,yhat,eps=1e-6):
    denom=(np.abs(y)+np.abs(yhat)+eps)/2.0
    return float(np.mean(np.abs(yhat-y)/denom)*100.0)
def mase(y,yhat):
    q=float(np.mean(np.abs(y[1:]-y[:-1]))+1e-6)
    return float(np.mean(np.abs(yhat-y))/q)
def les(err_eval, err_train): return float(err_eval)/(float(err_train)+1e-12)
