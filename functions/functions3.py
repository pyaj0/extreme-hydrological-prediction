import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# define activation functions
class ClassReprMeta(type):
    def __repr__(cls):
        return cls.__name__

class IdentityFn(nn.Module, metaclass=ClassReprMeta):
    def __init__(self, slope):
        super().__init__()
        self.slope = slope
    def forward(self, x):
        return x
    def __repr__(self):
        return f'{self.__class__.__name__}({self.slope})'

class SigmoidSlope(nn.Module, metaclass=ClassReprMeta):
    def __init__(self, slope):
        super().__init__()
        self.slope = slope
    def forward(self, x):
        return 1. / (1 + torch.exp(-x * self.slope))
    def __repr__(self):
        return f'{self.__class__.__name__}({self.slope})'

class TanhSlope(nn.Module, metaclass=ClassReprMeta):
    def __init__(self, slope):
        super().__init__()
        self.slope = slope
    def forward(self, x):
        return  (1 - torch.exp(-2*x*self.slope)) / (1 + torch.exp(-2*x*self.slope))
    def __repr__(self):
        return f'{self.__class__.__name__}({self.slope})'

class ReLUSlope(nn.Module, metaclass=ClassReprMeta):
    def __init__(self, slope):
        super().__init__()
        self.slope = slope
    def forward(self, x):
        return  x * (x > 0)
    def __repr__(self):
        return f'{self.__class__.__name__}({self.slope})'

# adds temporal depth to data
def input_temporal_depth(events, event_number, column_name, d_w, d_events):
    xs = []
    skip_shift = False
    mask = events['Evenement'] == event_number
    x = events[column_name].copy()
    if len(d_w[column_name]) == 2:
        w_step = 1
    if len(d_w[column_name]) == 3:
         w_step = d_w[column_name][2]
    w_min, w_max = d_w[column_name][0], d_w[column_name][1]
    x_start, x_end = d_events[event_number][0], d_events[event_number][1]
    
    if w_min < 0: # for prediction shift
        xp = x.shift(w_min).ffill(limit=-w_min)
        xp.name = xp.name + ' P' + str(-w_min)
        xs.append(xp) 
        w_min = 0

    if np.isnan(w_max): # skip shifts for column
        skip_shift = True
        w_max = 0
    
    for w in range(w_min, w_max+1, w_step): # for previous step shifts
        xw = x.shift(+w)
        xw.name = xw.name + str(w)
        xs.append(xw)
    xs = pd.concat(xs, axis=1)
    
    # remove remove other events after applying temporal depth
    xs = xs[mask]
    xs = xs.loc[x_start:x_end]

    if skip_shift: # skip shifts for column
        xs = xs.drop(columns=column_name+'0')
    
    return xs

# define functions to normalize (and reverse norm) variables to any interval
def norm_var(var, interval):
    var_min, var_max = np.nanmin(var), np.nanmax(var)
    var_norm = (var - var_min) * (interval[1] - interval[0]) / (var_max - var_min) + interval[0]
    if pd.isna(var_norm).all() == True:
        var_norm = np.nan_to_num(var_norm, 0)
    if pd.isna(var_norm).any() == True:
        print('Presence of nan in data please check.')
    return var_norm, var_min, var_max
    
def rev_norm_var(var_norm, var_min, var_max, interval): 
    var = var_min + ((var_norm - interval[0]) * (var_max - var_min)) / (interval[1] - interval[0])
    return var

# define functions to normalize (and reverse norm) dataframes
def normalize_dataframe(dataframe, interval):
    norm_dataframe = []
    d_norm_min_max = {}
    
    for i in range(dataframe.shape[1]):
        var_name = dataframe.columns[i]
        var = dataframe[var_name].values
        var_norm, var_min, var_max = norm_var(var, interval)
        norm_dataframe.append(pd.Series(var_norm))
        d_norm_min_max[var_name] = [var_min, var_max]
        
    norm_dataframe = pd.concat(norm_dataframe, axis=1)
    norm_dataframe.columns = dataframe.columns
    norm_dataframe.index = dataframe.index
    return norm_dataframe, d_norm_min_max

def rev_norm_dataframe(norm_dataframe, interval, d_norm_min_max, one_var=None):
    # if type(norm_dataframe) == np.array specify var_name as one_var
    if one_var: 
        var_min, var_max = d_norm_min_max[one_var]
        var = rev_norm_var(norm_dataframe, var_min, var_max, interval)
        return var
    
    else: # if type(norm_dataframe) == pd.DataFrame
        dataframe = []
        
        for i in range(norm_dataframe.shape[1]):
            var_name = norm_dataframe.columns[i]
            var_norm = norm_dataframe[var_name].values
            var_min, var_max = d_norm_min_max[var_name]
            var = rev_norm_var(var_norm, var_min, var_max, interval)
            dataframe.append(pd.Series(var))
            
        dataframe = pd.concat(dataframe, axis=1)
        dataframe.columns = norm_dataframe.columns
        dataframe.index = norm_dataframe.index
        return dataframe

# function for k-fold cross-validation
def k_fold_cross_validation(num_rows_data, k, shuffle_seed=False):
    """
    Gets indices of training and validation for each k-fold in cross-validation.

    Parameters:
    - num_rows_data: Number of rows of data or lenght of data 
    - k: Shuffle or not order of events (if not precise random seed number)

    Returns:
    - List of training and validation indices tuples for each fold
    """
    
    indices = np.arange(num_rows_data)
    fold_sizes = np.full(k, num_rows_data // k, dtype=int)
    fold_sizes[: num_rows_data % k] += 1
    
    if shuffle_seed:
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)

    fold_indices = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        current = stop
        fold_indices.append((train_indices, val_indices))

    return fold_indices

# gets indexes of events from event number
def get_event_indices(list_of_keys, d_iloc_events):
    list_idx = []
    for i in list_of_keys:
        list_idx.append(list(range(d_iloc_events[i][0], d_iloc_events[i][1]+1)))
     
    # flatten list
    list_idx = [item for sublist in list_idx for item in sublist]
    return list_idx

# function to calculate the mean absolute percentage error
def mape(obs, pred):
    ape = (obs - pred) / obs
    sum_ape = torch.sum(torch.abs(ape))
    return (100 * sum_ape) / len(obs)

# function to calculate NASH error
def nse(obs, pred):
    num = torch.sum(torch.pow((obs - pred), 2))
    denum = torch.sum(torch.pow((obs - torch.mean(obs)), 2))
    return 1 - (num/denum)

# function to calculate persistency error
def pers(obs, pred, horizon):
    num = torch.sum(torch.pow(obs[horizon:] - pred[horizon:], 2))
    denum = torch.sum(torch.pow(obs[horizon:] - obs[:-horizon], 2))
    return 1 - num/denum

# function to calculate synchronous percentage of peak error
def peak(obs, pred):
    argmax = np.nanargmax(obs)
    peak = pred[argmax] / obs[argmax]
    return 1 - abs(1-peak)

# function to calculate synchronous percentage of peak error without the minimum
def peak_ref_min(obs, pred):
    argmax = np.nanargmax(obs)
    peak = (pred[argmax] -  torch.min(obs)) / (obs[argmax] -  torch.min(obs))
    return 1 - abs(1-peak)

# function to calculate KGE error
def kge(obs, pred):
    m1, m2 = torch.mean(obs, axis=0), torch.mean(pred, axis=0)
    num_r = torch.sum((obs - m1) * (pred - m2), axis=0)
    den_r = torch.sqrt(torch.sum((obs - m1) ** 2, axis=0)) * torch.sqrt(torch.sum((pred - m2) ** 2, axis=0))
    r, beta = num_r / den_r, m2 / m1
    gamma = (torch.std(pred, axis=0) / m2) / (torch.std(obs, axis=0) / m1)
    return 1. - torch.sqrt((r - 1.) ** 2 + (beta - 1.) ** 2 + (gamma - 1.) ** 2)

# functions to calculate persistency error at high water levels
def consecutive(array, stepsize=1):
    return np.split(array, np.where(np.diff(array) != stepsize)[0]+1)

def add_values(array, horizon):
    values_add = [array[0] - i for i in range(horizon, 0, -1)]
    return np.insert(array, 0, values_add)

def persh(obs, pred, horizon, h=0.5, method='threshold'):
    if type(obs) == torch.Tensor:
        obs, pred = obs.detach().numpy(), pred.detach().numpy()
    if method == 'threshold':
        indices = np.where(obs > h)[0]
        indices = consecutive(indices)
        indices = [add_values(ind, horizon) for ind in indices]
        indices = np.unique(np.concatenate(indices))
        return pers(torch.tensor(obs[indices]), torch.tensor(pred[indices]), horizon)
    if method == 'quantile':
        q = np.percentile(obs, h)
        indices = np.where(obs > q)[0]
        indices = consecutive(indices)
        indices = [add_values(ind, horizon) for ind in indices]
        indices = np.unique(np.concatenate(indices))
        return pers(torch.tensor(obs[indices]), torch.tensor(pred[indices]), horizon)

def peaksh(obs, pred, h=0.5, method='threshold'):
    if type(obs) == torch.Tensor:
        obs, pred = obs.detach().numpy(), pred.detach().numpy()
    if method == 'threshold':
        indices = np.where(obs > h)[0]
        indices = consecutive(indices)
        peak_indices = [np.argmax(obs[ind]) for ind in indices]
        peak_indices = [ind[p] for ind, p  in zip(indices, peak_indices)]
        return [1 - abs(1-(pred[peak_ind]-min(obs))/(obs[peak_ind]-min(obs))) for peak_ind in peak_indices]
    if method == 'quantile':
        q = np.percentile(obs, h)
        indices = np.where(obs > q)[0]
        indices = consecutive(indices)
        peak_indices = [np.argmax(obs[ind]) for ind in indices]
        peak_indices = [ind[p] for ind, p  in zip(indices, peak_indices)]
        return [1 - abs(1-(pred[peak_ind]-min(obs))/(obs[peak_ind]-min(obs))) for peak_ind in peak_indices]
