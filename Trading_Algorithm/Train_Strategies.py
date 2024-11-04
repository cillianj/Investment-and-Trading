

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:09:41 2024

@author: cilli
"""


import yfinance as yf
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from itertools import product



logging.basicConfig(
    filename='training_expanded.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

""" Above sets up logging so all log messages are stored in 
 a file called training_expanded.log"""

initial_capital = 1000000 #sets the initial capital of pur portfolio

# Define the strategy functions with parameter flexibility
def buy_and_hold(data):
    """Implements a simple buy and hold strategy, the benchmark strategy
       parameters: data - this is the 'Close' price of the asset
       
    """
    data['position'] = 1 #take position 1 i.e buy asset
    data['strategy_returns'] = data['Close'].pct_change().fillna(0) #tracks returns
    return data



def mean_reversion_strategy(data, window=20, threshold=0.02, mode='deviation'):
    """ This implements a mean-reversion strategy 
        Parameters:
            data- Close price 
            window- this is the number of periods(days) to calculate the moving average
            threshold- this is the daeviation from the mean required to trigger a signal
            mode-  there are two modes deviation and crossover they have different requierements for a signal to occur
            
       Deviation- if (close - moving_avg)/moving_avg<-threshold
                  then we get a buy signal 
       crossover- if close>moving_avg then we get a buy signal
              
       so the two strategies are quite different
    
    """
    data['moving_avg'] = data['Close'].rolling(window).mean()
    data['signal'] = 0

    if mode == 'deviation':
        data['signal'] = np.where(
            (data['Close'] - data['moving_avg']) / data['moving_avg'] < -threshold, 1, 0 #buy
        )
        data['signal'] = np.where(
            (data['Close'] - data['moving_avg']) / data['moving_avg'] > threshold, -1, data['signal'] #sell
        )
    elif mode == 'crossover':
        data['signal'] = np.where(data['Close'] > data['moving_avg'], 1, 0) #sell
        data['signal'] = np.where(data['Close'] < data['moving_avg'], -1, data['signal']) #buy
    else:
        logging.warning(f"Unsupported mode '{mode}' for mean_reversion_strategy.")

    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
    return data

def trend_following_strategy(data, short_window=20, long_window=50):
    """Parameters-
    data- input data we import from Yahoo finance
    short_window ~ the window size for the short term moving average
    long_window ~ the window size for the long term moving average
    
    Overview:
        when the short term average is greater than the long term average 
        this indicates an upwards trends so we get a buy signal.
    """
    
    data['short_mavg'] = data['Close'].rolling(window=short_window).mean()
    data['long_mavg'] = data['Close'].rolling(window=long_window).mean()
    data['signal'] = np.where(data['short_mavg'] > data['long_mavg'], 1, 0)
    data['position'] = data['signal'].shift().fillna(0)
    data['strategy_returns'] = data['position'] * data['Close'].pct_change().fillna(0)
    return data

def breakout_strategy(data, window=20, quantile=0.75):
    """ 
    Parameters:
        data- contains close price and we use that to compute volatility
        window- window for compution rolling average of volatility
        quantile- the number the volatility has to exceeed to trigger 
                  a buy signal
    Overview:
        This strategy is concerned with volatility
        The hypothesis is that high volatility results in returns
        so a buy signal is issued if the volatility rolling average
        exceeds the quantile level.
    
    
    """
    
    data['volatility'] = data['Close'].rolling(window).std()
    threshold = data['volatility'].quantile(quantile)
    #data['signal'] = np.where(data['volatility'] > threshold, 1, 0)
    data['signal'] = np.where(data['volatility'] > threshold, np.where(data['Close'].pct_change() < 0, -1, 1), 0)

    data['position'] = data['signal'].shift().fillna(0)
    data['strategy_returns'] = data['position'] * data['Close'].pct_change().fillna(0)
    return data

def momentum_strategy(data, window=20, threshold=0.02):
    """ 
    parameters:
        data: Close price imported from Yahoo finance
        window: period for calculating momentum i.e pct change
        threshold: level if the momentum oges above, a signal is generated
        
    Overview:
        momentum is calculated as the percentage change in the assets 
        value over the window.
        If the momentum is greater than the threshold then a buy signal 
        is generated.
     """
    
    data['momentum'] = data['Close'].pct_change(periods=window)
    data['signal'] = np.where(data['momentum'] > threshold, 1, 0)
    data['position'] = data['signal'].shift().fillna(0)
    data['strategy_returns'] = data['position'] * data['Close'].pct_change().fillna(0)
    return data

def rsi_strategy(data, window=14, overbought=70, oversold=30):
    """
    Parameters:
        data: Close price on previous day(or hour, can be adapted)
        window: the period in whih we are calculating the gain/loss 
        overbought: The rsi value above which the asset is considered overbought
        oversold: The rsi value below which the asset is considered oversold
        
    Calculations:
        delta: price change over one period
        gain: rolling mean of positive price changes over window
        loss: absolute value of the rolling mean of the negative price changes over the period
        rs: calculates relative strength = gain/loss
        rsi= Relative Strength Index is stored in data as 100 - (100 / (1 + rs))
        
     Overview:
         The signal is set to 0 automatically
         if RSI < oversold then a signal of 1 (buy) is generated
         if RSI> overbought then a signal of -1 (sell) is generated
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['signal'] = 0
    data['signal'] = np.where(data['RSI'] < oversold, 1, data['signal'])
    data['signal'] = np.where(data['RSI'] > overbought, -1, data['signal'])
    data['position'] = data['signal'].shift().fillna(0)
    data['strategy_returns'] = data['position'] * data['Close'].pct_change().fillna(0)
    return data

def moving_average_crossover_strategy(data, short_window=50, long_window=200):
    """
    Parameters:
        data: Close price 
        short_window: window over which short term moving average is calculated
        long_window:  window over which long term moving average is calculated
        
     Calculations:
         We calculate the short term an long term moving averages
     
    Overview:
        The signal is initially set to 0
        If the short term MA crosses above the long term MA this suggests
        an uptrend so a 1 (buy) signal is generated
        If the short term MA crosses below the long term MA this suggests
        a downward trend so a -1 (sell) signal is generated
        
         
    
    """
    
    data['short_ma'] = data['Close'].rolling(window=short_window).mean()
    data['long_ma'] = data['Close'].rolling(window=long_window).mean()
    data['signal'] = 0
    data['signal'] = np.where(data['short_ma'] > data['long_ma'], 1, 0)
    data['signal'] = np.where(data['short_ma'] < data['long_ma'], -1, data['signal'])
    data['position'] = data['signal'].shift().fillna(0)
    data['strategy_returns'] = data['position'] * data['Close'].pct_change().fillna(0)
    return data

def bollinger_bands_strategy(data, window=20, num_std=2):
    """
    Parameters:
        data: Close price
        Window: number of periods used to calculate moving average
        num_std: number of std deviatons away from mean used for calculating
                 the upper and lower bands
                 
     Calculations:
         We calculate the rolling moving average over the window,
         We calculate the rolling std dev over the window
         The upper band is the MA+num_std*std
         The lower band is the MA+num_std*std
         
     Overview:
         A Signal of 0 is automatically set.
         
         A buy signal 1 is generated if the price drops below the
         lower bound indicating it is oversold and could revert.
         
         A sell signal -1 is generated if the price goes above the 
         upper bound indicating it is overbought and the price could 
         drop.
         
         
         
    
    """
    
    
    data['ma'] = data['Close'].rolling(window).mean()
    data['std'] = data['Close'].rolling(window).std()
    data['upper_band'] = data['ma'] + (data['std'] * num_std)
    data['lower_band'] = data['ma'] - (data['std'] * num_std)
    data['signal'] = 0
    data['signal'] = np.where(data['Close'] < data['lower_band'], 1, data['signal'])
    data['signal'] = np.where(data['Close'] > data['upper_band'], -1, data['signal'])
    data['position'] = data['signal'].shift().fillna(0)
    data['strategy_returns'] = data['position'] * data['Close'].pct_change().fillna(0)
    return data

# Function to fetch data for assets
def fetch_data_for_assets(assets, start_date, end_date):
    """
    Parameters:
        assets ~ list of assets that we want to trade 
                 stored in Assets.json to be loaded in later
        start_date ~ start date of data to be loaded in
        end_date ~ end date of data to be loaded in
    
    Overview:
        loads in data for assets in list assets
        from start date until end date, 
        
        try is used toensure if any errors occur the program returns 
        none or nan instead of crashing.
        
        Threads allows us to download data simultaneously for 
        faster downloads
        
        The interval 1d means that the algorithm makes trades based 
        off daily close price data this can be changed to have a more 
        active or less active algorithm.
        
    
    """
    
    try:
        data = yf.download(
            tickers=list(assets),
            start=start_date,
            end=end_date,
            interval='1m', 
            group_by='ticker',
            threads=True
        )
        return data
    except Exception as e:
        logging.error(f"Error fetching data for assets: {e}")
        return None

# Function to evaluate strategies
#Cumulative Returns
def evaluate_strategy(data, strategy_func, **kwargs):
   """
   Parameters:
       data: Our data frame containing historical price data
       strategy_func: the strategy being applied to the data
       e.g buy and hold, mean reversion etc
       **kwargs: additional arguments that are the parameters of 
                 the strategy functions.
    Overview: 
        the code calcuates the cumulative returns for each strategy
        
   """
   try: #using try in case of errors in data import or elsewise
        strategy_data = strategy_func(data.copy(), **kwargs)
        # Calculate the cumulative returns of the strategy
        cumulative_returns = strategy_data['strategy_returns'].cumsum().iloc[-1]
        return cumulative_returns
   except Exception as e:
        logging.error(f"Error evaluating strategy: {e}")
        return float('-inf')  # Return a very low value if there's an error



#MSE
"""
def evaluate_strategy(data, strategy_func, **kwargs):
    try:
        strategy_data = strategy_func(data.copy(), **kwargs)
        mse = mean_squared_error(strategy_data['Close'], strategy_data['strategy_returns'].cumsum())
        return mse
    except Exception as e:
        logging.error(f"Error evaluating strategy: {e}")
        return float('inf')
"""

#======================
#mse criterion,no parameter tuning
#======================
"""
def train_best_strategies(start_date, end_date):
    # Load assets from JSON configuration
    try:
        with open('Assets.json', 'r') as f:
            assets = json.load(f)  # Assuming this is a list of asset symbols
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON file: {e}")
        print(f"Error loading JSON file: {e}")
        return

    # Fetch historical data
    data = fetch_data_for_assets(assets, start_date, end_date)
    if data is None:
        print("Error fetching historical data.")
        return

    best_strategies = {}
    
    # Iterate over each asset and find the best strategy
    for symbol in assets:
        try:
            asset_data = data[symbol].dropna()
        except KeyError:
            logging.error(f"Data for {symbol} not found. Skipping.")
            continue

        if asset_data.empty:
            continue

        # Define strategy candidates and their parameters
        strategy_candidates = [
            {'name': 'buy_and_hold', 'func': buy_and_hold, 'params': {}},
            {'name': 'mean_reversion_deviation', 'func': mean_reversion_strategy, 'params': {'window': 20, 'threshold': 0.02, 'mode': 'deviation'}},
            {'name': 'mean_reversion_crossover', 'func': mean_reversion_strategy, 'params': {'window': 20, 'mode': 'crossover'}},
            {'name': 'trend_following', 'func': trend_following_strategy, 'params': {'short_window': 20, 'long_window': 50}},
            {'name': 'breakout', 'func': breakout_strategy, 'params': {'window': 20, 'quantile': 0.75}},
            {'name': 'momentum', 'func': momentum_strategy, 'params': {'window': 20, 'threshold': 0.02}},
            {'name': 'rsi', 'func': rsi_strategy, 'params': {'window': 14, 'overbought': 70, 'oversold': 30}},
            {'name': 'moving_average_crossover', 'func': moving_average_crossover_strategy, 'params': {'short_window': 50, 'long_window': 200}},
            {'name': 'bollinger_bands', 'func': bollinger_bands_strategy, 'params': {'window': 20, 'num_std': 2}}
        ]

        # Evaluate all strategies
        best_strategy = None
        best_mse = float('inf')

        for candidate in strategy_candidates:
            mse = evaluate_strategy(asset_data, candidate['func'], **candidate['params'])
            if mse < best_mse:
                best_mse = mse
                best_strategy = candidate

        # Store the best strategy for the asset, excluding unnecessary fields
        if best_strategy:
            best_strategies[symbol] = {
                'strategy': best_strategy['name'],
                'parameters': best_strategy['params']
            }

    # Save the best strategies to a JSON file, excluding unnecessary fields
    cleaned_strategies = {
        symbol: {
            'strategy': info['strategy'],
            'parameters': info.get('parameters', {})
        }
        for symbol, info in best_strategies.items()
    }

    with open('best_expanded_strategies.json', 'w') as f:
        json.dump(cleaned_strategies, f, indent=4)
    logging.info("Best expanded strategies saved to best_expanded_strategies.json.")
"""

#======================
# Cumulative Returns No Parameter Tuning
#======================

def train_best_strategies(start_date, end_date):

    #Parameters:
      #  start_date: the start date of our testing
       # end_dat: the end date of our testing
 #   Overview:
  #      This function ties uses all the previous functions
   #     steps:
    #        1) Loads in the assets as alist from the json file (if error report)
     #       2)feches historical price data for all the assets in the list and
      #        gives the dataframe label "data"
       #     3)We create an empty dictionry best_strategies{} to store the 
        #      asset labels and the strategies
         #   4)we define our candidate strtegies in a list
          #  5)Evaluate strategies for each asset
#              for each asset the function iterates and calculates 
 #             the cumulative returns using the evaluate strategies function
  #          6)the best strategy found is stored in the dictionary
   #           along with the asset label, the parameters and the 
    #          cumulative returns.
     #       7)the dictionary is converted to a json file so we can read
      #        it and it can be used in the trading code.
    
    try:
        with open('Assets.json', 'r') as f:
            assets = json.load(f)  #Loading in list of assets for testing
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON file: {e}")
        print(f"Error loading JSON file: {e}")
        return

    # Fetch  data
    data = fetch_data_for_assets(assets, start_date, end_date)
    if data is None:
        print("Error fetching historical data.")
        return

    best_strategies = {}
    
    # Iterate over each asset and find the best strategy
    for symbol in assets:
        try:
            asset_data = data[symbol].dropna()
        except KeyError:
            logging.error(f"Data for {symbol} not found. Skipping.")
            continue

        if asset_data.empty:
            continue

        # Define strategy candidates and their parameters
        strategy_candidates = [
            {'name': 'buy_and_hold', 'func': buy_and_hold, 'params': {}},
            {'name': 'mean_reversion_deviation', 'func': mean_reversion_strategy, 'params': {'window': 20, 'threshold': 0.02, 'mode': 'deviation'}},
            {'name': 'mean_reversion_crossover', 'func': mean_reversion_strategy, 'params': {'window': 20, 'mode': 'crossover'}},
            {'name': 'trend_following', 'func': trend_following_strategy, 'params': {'short_window': 20, 'long_window': 50}},
            {'name': 'breakout', 'func': breakout_strategy, 'params': {'window': 20, 'quantile': 0.75}},
            {'name': 'momentum', 'func': momentum_strategy, 'params': {'window': 20, 'threshold': 0.02}},
            {'name': 'rsi', 'func': rsi_strategy, 'params': {'window': 14, 'overbought': 70, 'oversold': 30}},
            {'name': 'moving_average_crossover', 'func': moving_average_crossover_strategy, 'params': {'short_window': 50, 'long_window': 200}},
            {'name': 'bollinger_bands', 'func': bollinger_bands_strategy, 'params': {'window': 20, 'num_std': 2}}
        ]

        # Evaluate all strategies
        best_strategy = None
        highest_returns = float('-inf')

        for candidate in strategy_candidates:
            cumulative_returns = evaluate_strategy(asset_data, candidate['func'], **candidate['params'])
            if cumulative_returns > highest_returns:
                highest_returns = cumulative_returns
                best_strategy = candidate

        # Store the best strategy for the asset
        if best_strategy:
            best_strategies[symbol] = {
                'strategy': best_strategy['name'],
                'parameters': best_strategy['params'],
                'cumulative_returns': highest_returns
            }

    # Save the best strategies to a JSON file
    cleaned_strategies = {
        symbol: {
            'strategy': info['strategy'],
            'parameters': info.get('parameters', {}),
            'cumulative_returns': info['cumulative_returns']
        }
        for symbol, info in best_strategies.items()
    }

    with open('best_expanded_strategies.json', 'w') as f:
        json.dump(cleaned_strategies, f, indent=4)
    logging.info("Best expanded strategies saved to best_expanded_strategies.json.")


#=======================
# Cumulative Returns, Parameter tuning 
#=======================

"""
def train_best_strategies(start_date, end_date):
    
    try:
        with open('Assets.json', 'r') as f:
            assets = json.load(f)  # Loading in list of assets for testing
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON file: {e}")
        print(f"Error loading JSON file: {e}")
        return

    # Fetch historical data
    data = fetch_data_for_assets(assets, start_date, end_date)
    if data is None:
        print("Error fetching historical data.")
        return

    best_strategies = {}

    # Parameter grids for strategies
    parameter_grids = {
        'mean_reversion_deviation': {
            'window': [10, 20, 30],
            'threshold': [0.01, 0.02, 0.03],
            'mode': ['deviation']
        },
        'mean_reversion_crossover': {
            'window': [10, 20, 30],
            'mode': ['crossover']
        },
        'trend_following': {
            'short_window': [10, 20, 30],
            'long_window': [50, 100, 150]
        },
        'breakout': {
            'window': [10, 20, 30],
            'quantile': [0.7, 0.75, 0.8]
        },
        'momentum': {
            'window': [10, 20, 30],
            'threshold': [0.01, 0.02, 0.03]
        },
        'rsi': {
            'window': [10, 14, 20],
            'overbought': [65, 70, 75],
            'oversold': [25, 30, 35]
        },
        'moving_average_crossover': {
            'short_window': [20, 50, 100],
            'long_window': [100, 150, 200]
        },
        'bollinger_bands': {
            'window': [10, 20, 30],
            'num_std': [1.5, 2, 2.5]
        }
    }

    # Iterate over each asset and find the best strategy
    for symbol in assets:
        try:
            asset_data = data[symbol].dropna()
        except KeyError:
            logging.error(f"Data for {symbol} not found. Skipping.")
            continue

        if asset_data.empty:
            continue

        best_strategy = None
        highest_returns = float('-inf')

        # Define strategy candidates and their parameters
        strategy_candidates = [
            {'name': 'buy_and_hold', 'func': buy_and_hold, 'params': {}},
            {'name': 'mean_reversion_deviation', 'func': mean_reversion_strategy},
            {'name': 'mean_reversion_crossover', 'func': mean_reversion_strategy},
            {'name': 'trend_following', 'func': trend_following_strategy},
            {'name': 'breakout', 'func': breakout_strategy},
            {'name': 'momentum', 'func': momentum_strategy},
            {'name': 'rsi', 'func': rsi_strategy},
            {'name': 'moving_average_crossover', 'func': moving_average_crossover_strategy},
            {'name': 'bollinger_bands', 'func': bollinger_bands_strategy}
        ]

        for candidate in strategy_candidates:
            strategy_name = candidate['name']
            strategy_func = candidate['func']

            if strategy_name in parameter_grids:
                param_combinations = list(product(*parameter_grids[strategy_name].values()))
                param_names = list(parameter_grids[strategy_name].keys())

                for param_set in param_combinations:
                    params = dict(zip(param_names, param_set))
                    cumulative_returns = evaluate_strategy(asset_data, strategy_func, **params)
                    if cumulative_returns > highest_returns:
                        highest_returns = cumulative_returns
                        best_strategy = {
                            'name': strategy_name,
                            'params': params
                        }
            else:
                # If no parameter grid, use default params
                cumulative_returns = evaluate_strategy(asset_data, strategy_func, **candidate.get('params', {}))
                if cumulative_returns > highest_returns:
                    highest_returns = cumulative_returns
                    best_strategy = candidate

        # Store the best strategy for the asset
        if best_strategy:
            best_strategies[symbol] = {
                'strategy': best_strategy['name'],
                'parameters': best_strategy['params'],
                'cumulative_returns': highest_returns
            }

    # Save the best strategies to a JSON file
    with open('best_expanded_strategies.json', 'w') as f:
        json.dump(best_strategies, f, indent=4)
    logging.info("Best expanded strategies saved to best_expanded_strategies.json.")
"""

#================
# MSE with tuned parameters
#================
"""
def train_best_strategies(start_date, end_date):
    
    try:
        with open('Assets.json', 'r') as f:
            assets = json.load(f)  # Loading in list of assets for testing
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON file: {e}")
        print(f"Error loading JSON file: {e}")
        return

    # Fetch historical data
    data = fetch_data_for_assets(assets, start_date, end_date)
    if data is None:
        print("Error fetching historical data.")
        return

    best_strategies = {}

    # Parameter grids for strategies
    parameter_grids = {
        'mean_reversion_deviation': {
            'window': [10, 20, 30],
            'threshold': [0.01, 0.02, 0.03],
            'mode': ['deviation']
        },
        'mean_reversion_crossover': {
            'window': [10, 20, 30],
            'mode': ['crossover']
        },
        'trend_following': {
            'short_window': [10, 20, 30],
            'long_window': [50, 100, 150]
        },
        'breakout': {
            'window': [10, 20, 30],
            'quantile': [0.7, 0.75, 0.8]
        },
        'momentum': {
            'window': [10, 20, 30],
            'threshold': [0.01, 0.02, 0.03]
        },
        'rsi': {
            'window': [10, 14, 20],
            'overbought': [65, 70, 75],
            'oversold': [25, 30, 35]
        },
        'moving_average_crossover': {
            'short_window': [20, 50, 100],
            'long_window': [100, 150, 200]
        },
        'bollinger_bands': {
            'window': [10, 20, 30],
            'num_std': [1.5, 2, 2.5]
        }
    }

    # Iterate over each asset and find the best strategy
    for symbol in assets:
        try:
            asset_data = data[symbol].dropna()
        except KeyError:
            logging.error(f"Data for {symbol} not found. Skipping.")
            continue

        if asset_data.empty:
            continue

        best_strategy = None
        lowest_mse = float('inf')

        # Define strategy candidates and their parameters
        strategy_candidates = [
            {'name': 'buy_and_hold', 'func': buy_and_hold, 'params': {}},
            {'name': 'mean_reversion_deviation', 'func': mean_reversion_strategy},
            {'name': 'mean_reversion_crossover', 'func': mean_reversion_strategy},
            {'name': 'trend_following', 'func': trend_following_strategy},
            {'name': 'breakout', 'func': breakout_strategy},
            {'name': 'momentum', 'func': momentum_strategy},
            {'name': 'rsi', 'func': rsi_strategy},
            {'name': 'moving_average_crossover', 'func': moving_average_crossover_strategy},
            {'name': 'bollinger_bands', 'func': bollinger_bands_strategy}
        ]

        for candidate in strategy_candidates:
            strategy_name = candidate['name']
            strategy_func = candidate['func']

            if strategy_name in parameter_grids:
                param_combinations = list(product(*parameter_grids[strategy_name].values()))
                param_names = list(parameter_grids[strategy_name].keys())

                for param_set in param_combinations:
                    params = dict(zip(param_names, param_set))
                    mse = evaluate_strategy(asset_data, strategy_func, **params)
                    if mse < lowest_mse:
                        lowest_mse = mse
                        best_strategy = {
                            'name': strategy_name,
                            'params': params
                        }
            else:
                # If no parameter grid, use default params
                mse = evaluate_strategy(asset_data, strategy_func, **candidate.get('params', {}))
                if mse < lowest_mse:
                    lowest_mse = mse
                    best_strategy = candidate

        # Store the best strategy for the asset
        if best_strategy:
            best_strategies[symbol] = {
                'strategy': best_strategy['name'],
                'parameters': best_strategy['params'],
                'mse': lowest_mse
            }

    # Save the best strategies to a JSON file
    with open('best_expanded_strategies.json', 'w') as f:
        json.dump(best_strategies, f, indent=4)
    logging.info("Best expanded strategies saved to best_expanded_strategies.json.")

"""

if __name__ == "__main__":
    """this code runs the training code for 6-months before 
    the end date."""
    
    end_date =  datetime(2024, 9, 1)
    start_date = datetime(2024,7,1)
    train_best_strategies(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
