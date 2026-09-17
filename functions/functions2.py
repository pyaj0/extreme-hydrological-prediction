import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as pe

# EVENT SELECTION

def get_pluvio_cumul(dataframe, nhours=24, threshold=50):
    """
    Gets cumulative precipitation and beginning and ending of cumulative events
    
    Args:
        dataframe: pd.DataFrame containing 
            - index pd.DatetimeIndex dates with time and 
            - columns pd.Series representing stations and their precipitation measurements
        nhours: int number of hours of cumulative sum (default 24 h)
        threshold: int value of cumulative sum to reach before selecting event (default 50 mm/ timestep of dataframe)
        
    Returns:
        dataframe_cumsum: pd.DataFrame containing
            - index pd.DatetimeIndex dates with time and 
            - columns pd.Series representing statuibs and their cumulative precipitation
        begin_end_events_cum: list of pd.Timestamp containing beginning and ending of cum event
    """
    # get n hours dynamic cumulative sum
    rolling_step = nhours * 12
    dataframe_cumsum = dataframe.fillna(0).rolling(rolling_step).sum()  # .shift(shift_step)

    # select events that have a nhours cumsum greater than threshold
    events_cum = dataframe_cumsum[dataframe_cumsum.values >= threshold]

    # get unique date events
    begin_end_events_cum = events_cum.index.unique()
    
    # get timestep of dataframe
    timestep = dataframe.index[1] - dataframe.index[0]

    # get beginning and ending of each cumulated event
    condition = begin_end_events_cum.to_series().diff() != timestep
    begin_end_events_cum = begin_end_events_cum[condition.where(condition).bfill(limit=1).fillna(0).astype(bool)]
    # don't forget to add end of last cumulated event
    begin_end_events_cum = begin_end_events_cum.union([events_cum.index[-1]])
    return dataframe_cumsum, begin_end_events_cum

# get beginning and ending for precipitation from begin and end of cumulative event
def get_begin_event(dataframe, time, begin_threshold=0, begin_delta_time=2):
    """
    Gets beginning of precipitation event from its cumulated precipitation event
    
    Args:
        dataframe: pd.DataFrame containing 
            - index pd.DatetimeIndex dates with time and 
            - columns pd.Series representing stations and precipitation measurements
        time: pd.Timestamp begining of cumulated precipitation event
        begin_delta_time: int number of hours without precipitation (default 2)
        begin_threshold: int minimum amount of precipitation permitted (default 0)
        timestep: int timestep of dataframe in minute (default 5)
        
    Returns:
        time: pd.Timestamp beginning of precipitation event
    """
    timestep = dataframe.index[1] - dataframe.index[0]
    begin_delta_time = pd.Timedelta(hours=begin_delta_time)
    
    while sum(dataframe.loc[time - begin_delta_time:time].sum() > begin_threshold) >= 1:
        time = time - timestep
    return time

def get_end_event(dataframe, time, end_threshold=0, end_delta_time=24):
    """
    Gets ending of precipitation event from its cumulated precipitation event
    
    Args:
        dataframe: pd.DataFrame containing 
            - index pd.DatetimeIndex dates with time and 
            - columns pd.Series representing stations and precipitation measurements
        time: pd.Timestamp beginning of cumulated precipitation event
        end_delta_time: int number of hours without precipitation (default 24)
        end_threshold: int minimum amount of precipitation permitted (default 0)
        timestep: int timestep of dataframe in minute (default 5)
        
    Returns:
        time: pd.Timestamp ending of precipitation event
    """
    timestep = dataframe.index[1] - dataframe.index[0]
    end_delta_time=pd.Timedelta(hours=end_delta_time)

    while sum(dataframe.loc[time:time + end_delta_time].sum() > end_threshold) >= 1:
        time = time + timestep
    return time


def get_pluvio_events(dataframe, begin_end_events_cum, threshold=(0,0), delta_time=(2,24)):
    """
    Gets beginning and ending for precipitation from begin and end of cumulative event
    
    Args:
        dataframe: pd.DataFrame containing 
            - index pd.DatetimeIndex dates with time and 
            - columns pd.Series representing stations and precipitation measurements
        begin_end_events_cum: list of pd.Timestamp representing beginning and ending of cum event
        threshold: tuple of int representing beginning and ending threshold in mm
        delta_time: tuple of int representing beginning and ending delta_time in hour
        
    Returns:
            begin_end_events: list of pd.Timestamp representing the  beginning and ending of precipitation event
    """
    begin_threshold, end_threshold = threshold[0], threshold[1]
    begin_delta_time, end_delta_time =  delta_time[0], delta_time[1]
    begin_end_events = []
    
    for i in range(0, len(begin_end_events_cum), 2):
        begin_end_events.append(get_begin_event(dataframe, begin_end_events_cum[i], begin_threshold, begin_delta_time))
        begin_end_events.append(get_end_event(dataframe, begin_end_events_cum[i], end_threshold, end_delta_time))
    return begin_end_events

# function to get nex key of a dictionary using previous key
def next_key(dict, key):
    keys = iter(dict)
    key in keys
    return next(keys, False)

# function to add margin to start and/or end of event in such a way that dates don't overlap
def add_margin(dictionary, time_step, margin_before=False, margin_after=False):

    dict_copy = dictionary.copy()
    
    if margin_before:
        time_before = pd.to_timedelta(margin_before)
    else:
        time_before = pd.to_timedelta('0d')
        
    if margin_after:
        time_after = pd.to_timedelta(margin_after)
    else:
        time_after = pd.to_timedelta('0d')
        
    time_step = pd.to_timedelta(time_step)
    
    for e in dict_copy.keys():
        
        init_start, init_end = dict_copy[e][0], dict_copy[e][1]
        
        if (type(margin_before) == str) & (type(margin_after) == str):
            final_start, final_end = dict_copy[e][0] - time_before, dict_copy[e][1] + time_after
        if (type(margin_before) == str) & (margin_after==False):
            final_start, final_end = dict_copy[e][0] - time_before, dict_copy[e][1]
        if (margin_before == False) & (type(margin_after) == str):
            final_start, final_end = dict_copy[e][0], dict_copy[e][1] + time_after
        if (margin_before == False) & (margin_after==False):
            final_start, final_end = dict_copy[e][0], dict_copy[e][1]
            
        # get next event from dictionary
        next_e = next_key(dict_copy, e)

        # to stop iteration
        if next_e == False:
            print('\n End of events')
            previous_e = list(dict_copy.keys())[-2]
            ## print('End', e, next_e, previous_e)
            dict_copy[e] = [final_start, final_end]
            print('\t', e, final_start, final_end)
            break
        
        # to start iteration
        if e == list(dict_copy.keys())[0]:
            previous_e = False
            print('Start of events')
            ## print('Start', e, next_e, previous_e)
            dict_copy[e] = [final_start, final_end]
            previous_e = list(dict_copy.keys())[0]
            print('\t',e, final_start, final_end)
        
        else:
            ## print('\n', e, next_e, previous_e)
            
            if final_end > dict_copy[next_e][0]:
                print(f"\nEvent {e}'s ending overlaps {next_e}")
                print(f'Event {e} should begin at {final_start} and end at {final_end}')
                print(f'But event {next_e} begins at {dict_copy[next_e][0]}')
                final_end = dict_copy[next_e][0] - time_step
                dict_copy[e] = [final_start, final_end]
                print(f"Change event {e}'s ending time to {final_end}")
                print('\n\t', e, final_start, final_end)
               
            elif final_start < dict_copy[previous_e][1]:
                print(f"\nEvent {e}'s beginning overlaps {previous_e}")
                print(f'Event {e} should begin at {final_start} and end at {final_end}')
                print(f'But event {previous_e} ends at {dict_copy[previous_e][1]}')
                final_start = dict_copy[previous_e][1] + time_step
                dict_copy[e] = [final_start, final_end]
                print(f"Change event {e}'s starting time to {final_start}")
                print('\n\t', e, final_start, final_end)

            else:
                dict_copy[e] = [final_start, final_end]
                print('\n\t', e, final_start, final_end)
                
            previous_e = e
            
    return dict_copy

# CORRELATION

def correlation(x, y, lag=True, kmax=None):
    """
    Calculates the correlation coefficient of two vectors according to Alain Mangin 1984
    dispite avec vectors having nan values
    https://www.sciencedirect.com/science/article/pii/0022169484902300?fr=RR-2&ref=pdf_download&rr=7f71bf946e2700ca

    Args:
        x: pandas.Series containing first vector
        y: pandas.Series containing second vector
        lag: specify if their is lag included in calculation
        kmax: specify the lag value
    
    Returns:
        rk: Numpy.array correlation coefficient of shape (k_max,)
    """
    # get number of elements of variables and deduct kmax
    n = len(x)  # len(x) should be equal to len(y)
    
    # if there is a specific value of kmax
    if kmax:
        k_max = kmax
    # if there is lag or not
    elif lag == True:
        k_max = int(n / 3)
    else:
        k_max = 1
    
    # if x or y are constants returns 0 correlation instead of nan
    if (np.diff(x) == 0).all() or (np.diff(y) == 0).all() == True:
        rk = np.full(k_max, np.nan)
        
    # if x or y are nan returns 0 correlation instead of nan
    elif np.isnan(x).all() or np.isnan(y).all() == True:
        rk = np.full(k_max, np.nan)
        
    else:
        # get mean and std of variables
        x_mean, y_mean = np.nanmean(x), np.nanmean(y)
        x_std, y_std = np.nanstd(x), np.nanstd(y)
        # get correlation
        Ck = []
        for k in range(k_max):
            product_list = []
            for i in range(n - k):
                if np.isnan(x.iloc[i]) or np.isnan(y.iloc[i + k]): # pass if nan in series
                    pass
                else:
                    product = (x.iloc[i] - x_mean) * (y.iloc[i + k] - y_mean)
                    product_list.append(product)
                
            product_list = np.array(product_list)
            len_product_list = np.sum(~np.isnan(product_list)) # lenght without nan values
            sum_product = np.nansum(product_list)
            Ck.append(sum_product / len_product_list) # ratio : n - k - number_nan
            
        rk = np.array(Ck / (x_std * y_std))
    return rk


# get correlations within one dataframe example pluvio
def one_dataframe_cor(pluvio, lag=True):
    """
    Loops through dataframe columns and computes <serie wise> correlation
    
    Agrs:
        pluvio: pandas.DataFrame containing n columns
        lag: specify if their is lag included in calculation 
    
    Returns:
        cor: NumPy.array of shape (n, n, kmax)
    """
    if lag == True:
        kmax = int(pluvio.shape[0] / 3)
    else:
        kmax = 1
    cor = np.zeros((pluvio.shape[1], pluvio.shape[1], kmax))
    for i in range(pluvio.shape[1]):
        for j in range(pluvio.shape[1]):
            if lag == True:
                cor[i][j] = correlation(pluvio.iloc[:, i], pluvio.iloc[:, j])
            else:
                cor[i][j] = correlation(pluvio.iloc[:, i], pluvio.iloc[:, j], lag)
    return cor


# get correlations from 2 dataframe
def two_dataframe_cor(pluvio, limni, lag=True):
    """
    Loops through two dataframe columns and computes <serie wise> correlation
    
    Agrs:
        pluvio: pandas.DataFrame containing n columns
        limni: pandas.DataFrame containing m columns
        lag: specify if their is lag included in calculation 
    
    Returns:
        cor: NumPy.array of shape (n, m, kmax)
    """
    if lag == True:
        kmax = int(pluvio.shape[0] / 3)
    else:
        kmax = 1
    cor = np.zeros((pluvio.shape[1], limni.shape[1], kmax))
    for i in range(pluvio.shape[1]):
        for j in range(limni.shape[1]):
            if lag == True:
                cor[i][j] = correlation(pluvio.iloc[:,i], limni.iloc[:,j])
            else:
                cor[i][j] = correlation(pluvio.iloc[:,i], limni.iloc[:,j], lag)
    return cor

# get correlation full correlation between series
def correlation_pos_neg(x, y, lag=True, kmax=None):
    """
    Calculates the full correlation coefficient (causal and anti-causal) of two vectors
    
    Args:
        x: pandas.Series containing first vector
        y: pandas.Series containing second vector
        lag: specify if their is lag included in calculation
        kmax: specify the lag value
    
    Returns:
        full_cor: Numpy.array of full correlation coefficient of shape (k_max,)
        range_cor: Numpy.array of full correlation range (indices)
    """
    # get number of elements of variables and deduct kmax
    n = len(x)  # len(x) should be equal to len(y)
    
    # if there is a specific value of kmax
    if kmax:
        k_max = kmax
    # if there is lag or not
    elif lag == True:
        k_max = int(n / 3)
    else:
        k_max = 1
        
    cor = correlation(x, y, lag, k_max)
    cor_neg = correlation(y, x, lag, k_max)
    cor_neg_flip = np.flip(cor_neg[1:])
    
    full_cor = np.concatenate((cor_neg_flip, cor))
    range_cor = np.arange(-k_max+1, k_max)
    
    return full_cor, range_cor

# reponse time and memory effect
def consecutive_accending(cor, index):
    """
    Finds index where consecutive accending numbers stops
    
    Args:
        cor: NumPy.array of one dimension
        index: int start index
    
    Returns:
        index: int where consecutive accending numbers stops 
    """
    for i in range(index, len(cor)):
        if (cor[i - 1] < cor[i]) and (i - (i - 1) == 1):
            index = i - 1
        else:
            index = i - 1
            break
    return index


def consecutive_descending(cor, index=1):
    """
    Finds index where consecutive descending numbers stops
    
    Args:
        cor: NumPy.array of one dimension
        index: int start index (default 1)
    
    Returns:
        index: int where consecutive descending numbers stops 
    """
    for i in range(index, len(cor)):
        if (cor[i - 1] > cor[i]) and (i - (i - 1) == 1):
            index = i - 1
        else:
            index = i - 1
            break
    return index


# first peak of correlation vector
def response_index(cor):
    """
    Finds the index of array's first peak 
    
    Args:
        cor: NumPy.array of one dimension
    
    Returns:
        index: int index of array's first peak
        - index == np.nan if cor is nan
        - index == -1 if cor < 0.2
        - index == int if cor > 0.2
        
    """
    if np.isnan(cor).all() == True:
        index = np.nan
    else:
        #index = consecutive_descending(cor)
        #index = consecutive_accending(cor, index + 1)
        index = consecutive_accending(cor, 1)
        if cor[index] < 0.2: # -1 is assigned if correlation weak
            index = -1
    return index


# max of correlation vector
def custom_nanargmax(cor):
    """
    Finds the index of array's overall peak (max)
    
    Args:
        cor: NumPy.array of one dimension
    
    Returns:
        index: int index of array's overall peak
        - index == np.nan if cor is nan
        - index == -1 if cor < 0.2
        - index == int if cor > 0.2
        
    """
    if np.isnan(cor).all() == True:
        index = np.nan
    else:
        try:
            index = np.nanargmax(cor)
            if cor[index] < 0.2:
                index = -1
            else:
                index = np.nanargmax(cor)
        except ValueError:
            index = np.nan
        
    return index

    
# first consistent peak of correlation vector
def first_peak(cor):
    """
    Finds the index of array's first peak
    
    Args:
        cor: NumPy.array of one dimension
    
    Returns:
        index: int index of array's first peak
        - index == np.nan if cor is nan
        - index == -1 if cor < 0.2
        - index == int if cor > 0.2
        
    """
    if np.isnan(cor).all() == True:
        index = np.nan
    else:
        index = consecutive_accending(cor, 1)
        
    return index

def next_peak(cor, index):
    """
    Finds the index of array's next peak from previous peak
    
    Args:
        cor: NumPy.array of one dimension
        index: int index of array's previous peak
        
    Returns:
        index: int index of array's next peak
        - index == np.nan if cor is nan
        - index == int elsewise
    """
    if np.isnan(cor).all() == True:
        index = np.nan
    else:
        index = consecutive_descending(cor, index+1)
        index = consecutive_accending(cor, index+1)
    
    return index

def consistent_first_peak(cor, patience_interval):
    """
    Finds the index of array's consistent first peak by ignoring false 
    first peaks due to early rainfall or pluvio data instability
    
    Args:
        cor: NumPy.array of one dimension
        patience_interval: int number of steps to consider between false peak and real peak
    
    Returns:
        index_current_peak : int
        - index == np.nan if cor is nan
        - index == -1 if cor < 0.2
        - index == int if cor > 0.2
    """
    if np.isnan(cor).all() == True:
        index_current_peak = np.nan
    else:
        index_current_peak = first_peak(cor) 
        index_next_peak = next_peak(cor, index_current_peak+1)
        
        try:
            while ((cor[index_current_peak] < cor[index_next_peak]) & 
                   (index_next_peak - index_current_peak <= patience_interval)):

                index_current_peak = index_next_peak
                index_next_peak = next_peak(cor, index_current_peak+1)
            
            if cor[index_current_peak] < 0.2: # -1 is assigned if correlation weak
                index_current_peak = -1
                    
        except IndexError:
            index_current_peak = np.nan
            
    return index_current_peak

# if correlation at first peak > 0.2, get index before correlation equals 0.2
def memory_index(cor, index):
    """
    Finds the index before array value equals 0.2 if index if first peak > 0.2
    
    Args:
        cor: NumPy.array of one dimension
        index: int index of array's first peak
    
    Returns:
        index: int index before array values equals 0.2
        - index == np.nan if (cor is nan) or (index is nan)
        - index == -1 if cor < 0.2
        - index == int if cor > 0.2
    """
    if np.isnan(cor).all() == True:
        index = np.nan
        
    elif np.isnan(index) == True:
        index = np.nan
        
    else:
        if cor[index] < 0.2:
            index = -1
        else:
            for i in range(index, len(cor)):
                if cor[i] < 0.2:
                    index = i - 1
                    break
            else:
                cor = cor[~np.isnan(cor)]
                index = len(cor) - 1
                
    return index


def percentage_temporal_interval(cor, response_index, percentage=0.5):
    """
    Finds the indices at a certain percentage of correlation at index of response time    
    
    Args:
        cor: NumPy.array of one dimension
        response_index: int index of response time
        percentage: float percentage of correlation at index of response time (default 0.5)
        
    Returns:
        lower_max_p: int lower index at percentage of correlation response index
        higher_max_p: int higher index at percentage of correlatio response index
        - lower_max_p, lower_max_p == np.nan, np.nan if cor is nan
        - lower_max_p, lower_max_p == -1, -1 if response_index == -1
        - lower_max_p, lower_max_p == int, int if response_index > 0.2
    """
    if np.isnan(cor).all() == True:
        lower_max_p, higher_max_p = np.nan, np.nan
        
    elif response_index == -1:
        lower_max_p, higher_max_p = -1, -1
    
    else:
        r_max = cor[response_index]
        r_max_p = r_max * percentage

        condition_array = np.argwhere(cor <= r_max_p).reshape(-1)

        if len(condition_array[condition_array < response_index]) == 0:
            lower_max_p = 0
        else:
            lower_max_p = np.max(condition_array[condition_array < response_index])

        if len(condition_array[condition_array > response_index]) == 0:
            higher_max_p = len(cor[~np.isnan(cor)]) - 1
        else:
            higher_max_p = np.min(condition_array[condition_array > response_index])

    return lower_max_p, higher_max_p

# function to get 4d-array using the maximum of all kmax
def set_array(array):
    """
    Transforms array-array-array-list to 4d-array
    
    Args:
        array : Numpy.array aray-array-array-list
        
    Returns:
        cor : Numpy.array of 4 dimensions
    """

    # get max lenght of kmax of all events
    max_k_max = 0
    for i in array:
        for j in i:
            for k in j:
                if np.array(k).shape[0] > max_k_max:
                    max_k_max = np.array(k).shape[0]

    # set rk from list to array with max kax lenght
    cor = np.full([array.shape[0], array.shape[1], array.shape[2], max_k_max], np.nan)
    for e in range(array.shape[0]):
        for i in range(array.shape[1]):
            for j in range(array.shape[2]):
                k_array = np.array(array[e, i, j]).copy()
                cor[e, i, j] = np.append(k_array, cor[e, i, j, len(k_array):])
    return cor