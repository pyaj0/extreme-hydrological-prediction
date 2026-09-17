import os
import gc # to kill globals
import concurrent.futures # to program in parallel
import numpy as np
import pandas as pd
import json # to save dates of events
import csv # to save results
import time, datetime

import torch
import torch.nn as nn
import torch_levenberg_marquardt as tlm

from functions.functions2 import *
from functions.functions3 import *

# get overtall beginning time of experience
beginning_all = [datetime.datetime.now(), time.perf_counter()]

# set experience number
exp_prefix = '14_1_MLP_AD_inputs_BM_at_k_sl_8_nl_1_nn_3_test_lr_05'

print('\n', exp_prefix)

# create working directories with exp_predix but pass if folder existes
try:
    os.mkdir(f'./modeles/{exp_prefix}')
except FileExistsError:
    pass

# load observation dataset
events = pd.read_csv('./donnees/traitees/hydro-meteo/dardennes_pluvio_limni_2012_2018_15min_interpolate_mean_cumsum_etp_events_fill_stantoine_rapid_events_models_predict_at_k_ok.csv', index_col=0)
events.index = pd.to_datetime(events.index)

# load event dates
with open('./donnees/traitees/results/2_2_d_events_modif.json', 'r') as f:
    d_events = json.load(f)

# convert dates to timestamp and keys to int in dictionary
# starting and ending dates of event 
d_events = {int(k): [pd.to_datetime(v[0]), pd.to_datetime(v[1])] for k, v in d_events.items()}

# select testing events and early stopping event
testing_list, early_stop = [27, 33], [45, 21] # [], np.nan # no test, no early stop #[27, 33], [45, 21]
# [27, 47], 45 # rapid
# [33, 54], 21 # karst

# specify output column name
output_col = 'H LasLagoubran'

# event numbers
event_numbers = [1, 2, 27, 29, 42, 45, 46, 47, 50, 51, 52, 57,
4, 5, 8, 11, 16, 17, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 48]

#[1, 2, 27, 29, 42, 45, 46, 47, 50, 51, 52, 57,
#4, 5, 8, 11, 16, 17, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 48]

# [1, 2, 27, 29, 42, 45, 46, 47, 50, 51, 52, 57] # rapid
# [4, 5, 6, 7, 8, 9, 11, 16, 17, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 48, 53, 54, 55] # karst # nan H StAntoine 53, 54, 55

# select event numbers
events = events[events['Evenement'].isin(event_numbers)]
d_events = {k: d_events[k] for k in d_events.keys() if k in event_numbers}

# set normalization between 0 and 0.9
my_interval = (0, 0.9)

# set random options
n_i_s = 10 # number of model's parameters initiation of each experience
i_m = 'uniform' # 'uniform', 'normal', 'xavier_uniform'
e_s = 0 # event seed: seed to shuffle events (no shuffle if e_s = 0)

# set optimizer and settings
list_optimizer = ['AD'] # 'LM', 'AD' # opt
ad_max_epoch, lm_max_epoch = 100, 15
ad_learning_rate, lm_learning_rate = 0.5, 1.0 #0.1, 1.0

# set range of hidden layers
ad_num_hidden = np.array([3]) # np.array([1])
lm_num_hidden = np.array([3]) # np.arange(1, 15+1, 1) #nn = np.arange(0, 10+1, 2); nn[0] = 1

# specify prevision horizon list
list_prevision_horizon = [2] # 1, 2, 3, 4 # pr_hs

# select constant input data
ad_limni_lag, ad_limni_step = 2, 1 # np.nan if no limni to input use 
lm_limni_lag, lm_limni_step = 2, 1
ad_d_w_static = {'H Ragas': [0, 0, 1], 'H StAntoine': [0, 0, 1], 'Toulon': [0, 0, 1], 'Cas. SAPC': [0, 0, 1], 'Castellet': [0, 0, 1]}
 # {} without none iterating pluvio
lm_d_w_static = {'H Ragas': [0, 0, 1], 'H StAntoine': [0, 0, 1], 'Toulon': [0, 0, 1], 'Cas. SAPC': [0, 0, 1], 'Castellet': [0, 0, 1]}
  # {'Station1': [0, 2], 'Station2':[0, 3, 1]}
# 'Toulon', 'Cap Cepet', 'Cas. SAPC', 'Castellet', 'Ste Baume', 'Meounes', 'Cuers'
# 'Rapid Model', 'Karst Model', 

# select input variables for temporal depth in iteration
ad_variables = ['Meounes'] # [output_col] to iterate over limni
lm_variables = ['Meounes'] # 'Station2'] to iterate over pluvio

# list of max temporal depths in iteration
ad_list_w = [0], 1 # [0], 1 # w, w_step
lm_list_w = [0], 1 # np.arange(0, 10+1, 1), 3
# to iterate over limni temporal depth with no limni as input, uncomment below
#list_w = np.concatenate([np.array([np.nan]), list_w])
#list_w = [int(x) if not np.isnan(x) else np.nan for x in list_w]

# set activation functions
ad_act_func1, ad_act_func2 = [TanhSlope], [TanhSlope] # IdentityFn, SigmoidSlope, TanhSlope act_func1
lm_act_func1, lm_act_func2 = [TanhSlope], [TanhSlope]

# list of activation function's slopes
ad_slope1, ad_slope2 = np.array([1]), np.array([1]) # np.array([0]) # slope # np.array([1, 3, 5]), np.array([0.1, 0.5, 1])
lm_slope1, lm_slope2 = np.array([1]), np.array([1]) # np.round(np.concatenate([np.arange(0.1, 1, 0.1), np.arange(1, 10+1, 1)]), 1)

# save model's parameters
save_params = False

class CustomModel(nn.Module):
    def __init__(
        self, input_size, num_hidden, act_func1, slope1, act_func2, slope2, initialization_seed,
        initialization_method, max_input_features=1000):
        
        super(CustomModel, self).__init__()
        
        # Set the initialization seed so that weight initialization is reproducible
        torch.manual_seed(initialization_seed)
        
        # Define layers
        self.fc1 = nn.Linear(input_size, num_hidden, bias=True)
        self.act1 = act_func1(slope1)
        self.fc2 = nn.Linear(num_hidden, 1, bias=True)
        self.act2 = act_func2(slope2)
        
        # Initialize weights using the provided method and a big weight tensor.
        self.initialize_weights(initialization_method, input_size, num_hidden, max_input_features)
    
    def initialize_weights(self, initialization_method, input_size, num_hidden, max_input_features):
        # Create a large weight tensor that will be used to initialize the layers
        if initialization_method == 'uniform':
            big_weight_tensor = torch.empty((num_hidden, max_input_features)).uniform_(-0.1, 0.1)
        elif initialization_method == 'normal':
            big_weight_tensor = torch.empty((num_hidden, max_input_features)).normal_(0.0, 0.1)
        elif initialization_method == 'xavier_uniform':
            big_weight_tensor = torch.empty((num_hidden, max_input_features))
            torch.nn.init.xavier_uniform_(big_weight_tensor)
        else:
            raise ValueError(
                f"Unknown initialization method: {initialization_method}. "
                "Use 'uniform', 'normal', or 'xavier_uniform'."
            )
        
        with torch.no_grad():
            # Initialize the first (hidden) layer:
            # Replace fc1.weight with the first 'input_size' columns of big_weight_tensor.
            self.fc1.weight.copy_(big_weight_tensor[:, :input_size])
            nn.init.constant_(self.fc1.bias, 0.1)
            
            # Initialize the second (output) layer:
            # The original code used the first column of big_weight_tensor for each hidden neuron.
            # fc2.weight has shape (1, num_hidden). We mimic that behavior by extracting the first
            # column of each hidden unit and reshaping to match.
            fc2_weight_vector = big_weight_tensor[:num_hidden, 0].unsqueeze(0)  # shape becomes (1, num_hidden)
            self.fc2.weight.copy_(fc2_weight_vector)
            nn.init.constant_(self.fc2.bias, 0.1)
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x
    
    def get_parameter_vector(self):
        return torch.cat([param.reshape(-1) for param in self.parameters()])
    
# create a list of set of argument to use in function
hyperparameters = []
# get cross validation events
cross_val_list = [x for x in event_numbers if x not in testing_list + early_stop]
for opt in list_optimizer:
    # set range of hidden layers and number of max epochs
    if opt == 'AD': # Adam
        list_num_hidden, max_epoch, learning_rate = ad_num_hidden, ad_max_epoch, ad_learning_rate
        list_w, list_act_func1, list_slope1, list_act_func2, list_slope2 = ad_list_w, ad_act_func1, ad_slope1, ad_act_func2, ad_slope2
        d_w_static, variables, limni_lag, limni_step = ad_d_w_static, ad_variables, ad_limni_lag, ad_limni_step
    if opt == 'LM': # Levenberg-Marquard
        list_num_hidden, max_epoch, learning_rate = lm_num_hidden, lm_max_epoch, lm_learning_rate
        list_w, list_act_func1, list_slope1, list_act_func2, list_slope2 = lm_list_w, lm_act_func1, lm_slope1, lm_act_func2, lm_slope2
        d_w_static, variables, limni_lag, limni_step = lm_d_w_static, lm_variables, lm_limni_lag, lm_limni_step
    for pr_h in list_prevision_horizon:
        for i_s in range(0, n_i_s):
            for num_hidden in list_num_hidden:
                for act_func1 in list_act_func1:
                    for slope1 in list_slope1:
                        for act_func2 in list_act_func2:
                            for slope2 in list_slope2:
                                for col in variables:
                                    for w in list_w[0]:
                                        w_step = list_w[1]
                                        for val_key in cross_val_list:
                                            if col == output_col: # add condition for varying limni lag
                                                d_w = {output_col: [int(-pr_h), w, w_step]}
                                                d_w = {k: v for d in (d_w, d_w_static) for k, v in d.items()}
                                                hyperparameters.append((opt, pr_h, i_s, num_hidden, act_func1, slope1, act_func2, slope2, col, w, val_key, d_w, max_epoch, learning_rate))
                                            else:  # specify constant inputs of model and adding it to dynamic inputs
                                                d_w = {output_col: [int(-pr_h), limni_lag, limni_step]}
                                                d_w_dynamic = {col: [0, w, w_step]}
                                                d_w = {k: v for d in (d_w, d_w_static, d_w_dynamic) for k, v in d.items()} #d_w, d_w_static, d_w_dynamic
                                                hyperparameters.append((opt, pr_h, i_s, num_hidden, act_func1, slope1, act_func2, slope2, col, w, val_key, d_w, max_epoch, learning_rate))
                                            
def process_model(opt, pr_h, i_s, num_hidden, act_func1, slope1, act_func2, slope2, col, w, val_key, d_w=d_w, max_epoch=max_epoch, learning_rate=learning_rate,
                  events=events, output_col=output_col, my_interval=my_interval, testing_list=testing_list, 
                  early_stop=early_stop, i_m=i_m, e_s=e_s, save_params=save_params):
    
    # remove early stopping from events 
    d_events_loop = {k: v for k, v in d_events.items() if k not in early_stop} 

    # transform into time depth dataframe and get start and end of event
    events_stations_temporal = []
    for station in d_w.keys():
        events_station_temporal = []
        for e in d_events:
            event_station_temporal = input_temporal_depth(events, e, station, d_w, d_events)
            events_station_temporal.append(event_station_temporal)
        events_station_temporal = pd.concat(events_station_temporal)
        events_stations_temporal.append(events_station_temporal)
    events_stations_temporal = pd.concat(events_stations_temporal, axis=1)
    
    # move output data at last column
    events_time = events_stations_temporal.copy()
    if d_w[output_col][0] >= 0: # in simulation
        output_col_name =  output_col + '0'
    else: # in prediction
        output_col_name = output_col + ' P' + str(-d_w[output_col][0])
    events_time = events_time[[col for col in events_time.columns if col != output_col_name] + [output_col_name]]
    
    if np.isnan(early_stop).any() == True:
        pass
    else:    
    # same for early stopping event
        early_events_stations_temporal = []
        for station in d_w.keys():
            early_events_station_temporal = []
            for e in early_stop:
                early_event_station_temporal = input_temporal_depth(events, e, station, d_w, d_events)
                early_events_station_temporal.append(early_event_station_temporal)
            early_events_station_temporal = pd.concat(early_events_station_temporal)
            early_events_stations_temporal.append(early_events_station_temporal)
        early_events_stations_temporal = pd.concat(early_events_stations_temporal, axis=1)
        early_event_time = early_events_stations_temporal.copy()
        early_event_time = early_event_time[[col for col in early_event_time.columns if col != output_col_name] + [output_col_name]]
        
        # get time of beginning and ending of each event in date and iloc dictionary
    d_iloc_events = {}
    for e in d_events_loop.keys():
        start_event, end_event = d_events_loop[e][0], d_events_loop[e][1]
        event_time = events_time.loc[start_event:end_event]
        i_start_event, i_end_event = events_time.index.get_loc(start_event), events_time.index.get_loc(end_event)
        d_iloc_events[e] = [i_start_event, i_end_event]
    
    # normalize data
    events_norm, d_min_max_events = normalize_dataframe(events_time, my_interval)
    
    
    # seperate input from output data
    X = events_norm.iloc[:,:-1]
    y = events_norm.iloc[:,[-1]]
    
    if np.isnan(early_stop).any() == True:
        pass
    else:
        # normalize early stopping event
        early_event_norm, d_min_max_early = normalize_dataframe(early_event_time, my_interval)
        # same for early stopping event
        x_early = early_event_norm.iloc[:,:-1]
        y_early = early_event_norm.iloc[:,[-1]]
        # convert early stopping set to tensor
        x_early, y_early = torch.tensor(x_early.values).float(), torch.tensor(y_early.values).float()
    
    # all event number in list 
    list_keys = list(d_iloc_events.keys())
    
    # Get validation data from d_iloc_events
    train_keys = [k for k in d_iloc_events.keys() if k not in [val_key] + early_stop + testing_list]
    val_keys = [val_key]

    # get indexes of different events
    train_idx = get_event_indices(train_keys, d_iloc_events)
    val_idx = get_event_indices(val_keys, d_iloc_events)

    # seperate input from output variable
    x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # convert sets to torch tensor float
    x_train, y_train = torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float()
    x_val, y_val = torch.tensor(x_val.values).float(), torch.tensor(y_val.values).float()
    
    # set model seed
    torch.manual_seed(i_s)

    # to get errors and to plot training, validation and early stopping curve
    list_train_loss = []
    list_val_loss = []
    list_early_loss = []
    list_val_kge = []
    list_val_nse = []
    list_val_pers = []
    list_val_mape = []
    list_val_peak = []

    # to plot observation vs prediction at best model's performance
    list_predict = []

    # to get model's parameters evolution during training
    list_params = []
    
    # create model   
    model = CustomModel(input_size=x_train.shape[1], num_hidden=num_hidden, act_func1=act_func1, slope1=slope1,
        act_func2=act_func2, slope2=slope2, initialization_seed=i_s, initialization_method=i_m,
    )

    # set optimizer module
    if opt == 'AD':        
        module = tlm.training.OptimizerModule(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
            loss_fn=torch.nn.MSELoss(),
        )
    if opt == 'LM':
        module = tlm.training.LevenbergMarquardtModule(
            model=model,
            loss_fn=tlm.loss.MSELoss(),
            learning_rate=learning_rate, attempts_per_step=10, solve_method='qr',
        )
        
        
    # get beginning time of experience
    beginning = time.time()
    
    # iterate over epochs and get loss and prediction information 
    for epoch in range(max_epoch):
    
        # get model predictions
        yhat_train = model(x_train)
        yhat_val = model(x_val)
        
        train_loss = torch.nn.functional.mse_loss(yhat_train, y_train)
        val_loss = torch.nn.functional.mse_loss(yhat_val, y_val)
        
        # get other validation loss
        val_mape = mape(y_val, yhat_val) # mean absolute error
        val_kge = kge(y_val, yhat_val) # kge error
        val_nse = nse(y_val, yhat_val) # nash error
        val_pers = pers(y_val, yhat_val, pr_h) # persistency error
        val_peak = peak_ref_min(y_val, yhat_val) # peak error
        
        # to save model's stats in training and validation at each k-fold
        list_train_loss.append(train_loss.item())
        list_val_loss.append(val_loss.item())
        list_val_kge.append(val_kge.item())
        list_val_nse.append(val_nse.item())
        list_val_pers.append(val_pers.item())
        list_val_mape.append(val_mape.item())
        list_val_peak.append(val_peak.item())
        
        if np.isnan(early_stop).any() == True:
            pass
        else:
            yhat_early = model(x_early)
            early_loss = torch.nn.functional.mse_loss(yhat_early, y_early)
            list_early_loss.append(early_loss.item())
            best_epoch_early = np.nanargmin(list_early_loss)
        
        # reverse normalize observations and predictions
        yhat_val_rev = yhat_val.detach().numpy().reshape(-1)
        yhat_val_rev = rev_norm_dataframe(yhat_val_rev, my_interval, d_min_max_events, output_col_name)
        
        # to save model's predictions
        list_predict.append(yhat_val_rev.tolist())  
        
        # to save model's parameters
        list_params.append(model.get_parameter_vector().clone().detach().tolist())      
        
        # update parameters
        module.training_step(x_train, y_train)
    
    # get duration of optimization
    duration = time.time() - beginning  
        
    # get best validation and early stopping epochs
    best_epoch_val = np.nanargmin(list_val_loss)
    
    # choose between best validation and early stopping epochs
    if np.isnan(early_stop).any() == True:
        best_epoch = -1
        best_epoch_early = -1
    else:
        if opt == 'LM':
            best_epoch = best_epoch_early
        if opt == 'AD':
            best_epoch = -1

    #print(f'\n OPT {opt} PRV {pr_h} IS {i_s} HID {num_hidden} STAT {col} W {w} VAL {val_key} MAXEPOCH {max_epoch} LR {learning_rate}')
    
    if save_params == False:
        list_params = [[]]
    
    return [opt, pr_h, i_s, num_hidden, act_func1, slope1, act_func2, slope2, col, w, val_key, list_train_loss, list_val_loss, list_early_loss, 
            list_val_kge, list_val_nse, list_val_pers, list_val_mape, list_val_peak, best_epoch_val, best_epoch_early, 
            list_predict[best_epoch_val], list_predict[best_epoch_early], list_predict[-1], duration, list_params, d_w]
                
d_information = {
    'Experience prefix': exp_prefix,
    'Events': event_numbers,
    'Early stopping event': early_stop,
    'Test events': testing_list,
    'Normilization interval': my_interval,
    'Model Optimizers': list_optimizer,
    'Number of Initialization': n_i_s,
    'Initialization Method': i_m,
    'Prevision Horizons': list_prevision_horizon,
    'Loss function': 'MSE',
    'Event seed': e_s,
    'Save Parameters': save_params,
    'AD Static Input': ad_d_w_static,
    'LM Static Input': lm_d_w_static,
    'AD Limni Lag': ad_limni_lag,
    'LM Limni Lag': lm_limni_lag,
    'AD Variables': ad_variables,
    'LM Variables': lm_variables,
    'AD Temporal window': ad_list_w,
    'LM Temporal window': lm_list_w,
    'AD Num Hid': ad_num_hidden,
    'LM Hum Hid': lm_num_hidden,
    'AD Activation Function 1': str(ad_act_func1),
    'LM Activation Function 1': str(lm_act_func1),
    'AD Slope 1': ad_slope1,
    'LM Slope 1': lm_slope1,
    'AD Activation Function 2': str(ad_act_func2),
    'LM Activation Function 2': str(lm_act_func2),
    'AD Slope 2': ad_slope2,
    'LM Slope 2': lm_slope2,
    'AD Learning Rate': learning_rate,
    'AD Max Epoch': ad_max_epoch,
    'AD Early Stop': -1,
    'LM Early Stop': early_stop,
    'LM Max Epoch': lm_max_epoch,  
}

# filter keys based on optimizer
if list_optimizer in [['AD'], ['LM']]:
    opt_prefix = list_optimizer[0]
    d_opt = {k: v for k, v in d_information.items() if opt_prefix in k}
    d_first = {k: v for k, v in list(d_information.items())[:12]}
    d_information = {**d_first, **d_opt}
    
d_information = {str(k): str(v) for k, v in d_information.items()}

# save experience information
with open(f'./modeles/{exp_prefix}/d_information.json', 'w') as f:
    json.dump(d_information, f)

# to save useful information
path_exp_table = f'./modeles/{exp_prefix}/exp_table.csv'
with open(path_exp_table, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Optimizer', 'Prevision Horizon', 'Initialization Seed','N Neurons', 'Act Function 1', 'Slope 1',
        'Act Function 2', 'Slope 2', 'Station', 'Temporal Window', 'Validation Event', 'Train Loss', 'Val Loss', 
        'Early Loss', 'KGE', 'NSE', 'PERS', 'MAPE', 'PEAK', 'Best Epoch Val', 'Best Epoch Early', 
        'Predictions Val', 'Predictions Early', 'Predictions Final', 'Duration', 'Params', 'Inputs'
    ])

if __name__ == '__main__':
    
    with concurrent.futures.ProcessPoolExecutor() as executor: #concurrent.futures.ProcessPoolExecutor(max_workers=46) #78
        futures = [executor.submit(process_model, opt, pr_h, i_s, num_hidden, act_func1, slope1, act_func2, slope2, col, w, val_key, d_w, max_epoch, learning_rate)
                   for opt, pr_h, i_s, num_hidden, act_func1, slope1, act_func2, slope2, col, w, val_key, d_w, max_epoch, learning_rate in hyperparameters]
        
        with open(path_exp_table, mode='a', newline='') as file:
            writer = csv.writer(file)

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                writer.writerow(result)

    # get overall ending time of experience
    ending_all = [datetime.datetime.now(), time.perf_counter()]

    # show overall beginning, ending and duration of experience
    print('\n')
    print(exp_prefix)
    print(f'Start \t {beginning_all[0]}')
    print(f'End \t {ending_all[0]}')
    print(f'Duration \t {str(ending_all[0] - beginning_all[0])} \t {np.round(ending_all[1] - beginning_all[1], 3)} seconds \n \n')
    
    def clear_globals():
        keep = {'__name__', '__doc__', '__package__', '__loader__',
                '__spec__', '__annotations__', '__builtins__', 'gc'}
        keys_to_delete = [key for key in globals() if key not in keep]
        for key in keys_to_delete:
            del globals()[key]
    
    clear_globals()
    gc.collect()