
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import json
import jsonschema
from jsonschema import validate
from datetime import datetime
import matplotlib.pyplot as plt


# Load environment variables from .env file (if needed)
load_dotenv()

# Initialize Logging
logging.basicConfig(
    filename='trading.log',
    level=logging.DEBUG,  # Set to DEBUG for detailed logs; change to INFO in production
    format='%(asctime)s:%(levelname)s:%(message)s'
)



# Define JSON schema for validation
schema = {
    "type": "object",
    "patternProperties": {
        "^[A-Z0-9=-]+$": { 
            "type": "object",
            "properties": {
                "strategy": {"type": "string"},
                "parameters": {"type": "object"}
            },
            "required": ["strategy"],
            "additionalProperties": True
        }
    },
    "additionalProperties": True
}



def load_best_strategies(json_path='best_expanded_strategies.json'):
    """
    This loads the json file that stores the assets and their
    strategies and verifies it has loaded in correctly
    
    """
    global best_strategies  
    try:
        with open(json_path, 'r') as f:
            best_strategies = json.load(f)
        validate(instance=best_strategies, schema=schema)
        logging.info(f"JSON configuration '{json_path}' loaded and validated successfully.")
        print(f"JSON configuration '{json_path}' loaded and validated successfully.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON file '{json_path}': {e}")
        print(f"Error loading JSON file '{json_path}': {e}")
        sys.exit(1)  # Terminate the script
    except jsonschema.exceptions.ValidationError as ve:
        logging.error(f"JSON validation error in '{json_path}': {ve}")
        print(f"JSON validation error in '{json_path}': {ve}")
        sys.exit(1)  

# Load strategies initially
load_best_strategies()




strategy_mapping = {
    'buy_and_hold': 'buy_and_hold_strategy',
    'rsi': 'rsi_strategy',
    'mean_reversion_deviation': 'mean_reversion_strategy',
    'mean_reversion_crossover': 'mean_reversion_strategy',
    'trend_following': 'trend_following_strategy',
    'breakout': 'breakout_strategy',
    'bollinger_bands': 'bollinger_bands_strategy',
    'momentum': 'momentum_strategy',
    'moving_average_crossover': 'moving_average_crossover_strategy'
    # Add more mappings as necessary
}
"""The Strategy mapping ensures all strategies 
    are labelled correctly to be used in functions
"""





initial_capital = 1_000_000  #starting capital of portfolio

#We start with 0 invested in each asset
holdings = {
    symbol: 0 for symbol in best_strategies.keys()
}

# Initialize a DataFrame to store portfolio values over time
portfolio_df = pd.DataFrame(columns=['Date', 'Total Portfolio Value'])
portfolio_df.set_index('Date', inplace=True)

# Initialize a DataFrame to store trade logs
trade_logs = pd.DataFrame(columns=[
    'DateTime', 'Symbol', 'Action', 'Quantity',
    'Price', 'Total Value', 'Uninvested Cash'
])



def buy_and_hold_strategy(data, **params):
    """Implements a simple buy and hold strategy, the benchmark strategy
       parameters: data - this is the 'Close' price of the asset
       
    """
    data['position'] = 1
    data['strategy_returns'] = data['Close'].pct_change().fillna(0)
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
    data['signal'] = 1

    if mode == 'deviation':
        data['signal'] = np.where(
            (data['Close'] - data['moving_avg']) / data['moving_avg'] < -threshold, 1, 0
        )
    elif mode == 'crossover':
        data['signal'] = np.where(data['Close'] > data['moving_avg'], 1, 0)
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
     data['desired_position'] = data['signal'].shift().fillna(0)
     data['position'] = data['desired_position']
     data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
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
        The hypothesi is that high volatility results in returns
        so a buy signal is issued if the volatility rolling average
        exceeds the quantile level.
    
    
    """
    data['volatility'] = data['Close'].rolling(window).std()
    threshold = data['volatility'].quantile(quantile)
    data['signal'] = np.where(data['volatility'] > threshold, 1, 0)
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
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
    data['signal'] = 1
    data['signal'] = np.where(data['momentum'] > threshold, 1, data['signal'])
    data['signal'] = np.where(data['momentum'] < -threshold, -1, data['signal'])
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
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
    data['signal'] = 1
    data['signal'] = np.where(data['RSI'] < oversold, 1, data['signal'])
    data['signal'] = np.where(data['RSI'] > overbought, -1, data['signal'])
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
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
    data['signal'] = 1
    data['signal'] = np.where(data['short_ma'] > data['long_ma'], 1, data['signal'])
    data['signal'] = np.where(data['short_ma'] < data['long_ma'], -1, data['signal'])
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
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
   data['signal'] = 1
   data['signal'] = np.where(data['Close'] < data['lower_band'], 1, data['signal'])
   data['signal'] = np.where(data['Close'] > data['upper_band'], -1, data['signal'])
   data['desired_position'] = data['signal'].shift().fillna(0)
   data['position'] = data['desired_position']
   data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
   return data



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
       logging.info(f"Fetching data for assets from {start_date} to {end_date}.")
       print(f"Fetching data for assets from {start_date} to {end_date}.")

       data = yf.download(
           tickers=list(assets),
           start=start_date,
           end=end_date,
           interval='1d',  #We will trade hourly as we want a more active algorithm
           group_by='ticker',
           threads=True,
           auto_adjust=True  # Adjust for dividends and splits
       )

       """# Check if data is fetched for each asset
       for asset in assets:
           if asset not in data.columns.levels[0]:
               logging.warning(f"No data fetched for asset {asset}.")
               print(f"Warning: No data fetched for asset {asset}.")
           else:
               logging.info(f"Data fetched for asset {asset}: {data[asset].shape[0]} rows.")
               print(f"Data fetched for asset {asset}: {data[asset].shape[0]} rows.")
               print(data[asset].head())  # Debug: Print first few rows

       # Handle cases where only one asset is fetched (different DataFrame structure)
       if len(assets) == 1:
           data = {assets[0]: data}

       # Localize timezone to UTC for consistency
       if isinstance(data, pd.DataFrame):
           if data.index.tz is None:
               data = data.tz_localize('UTC')
           else:
               data = data.tz_convert('UTC')
       else:
           for asset in data:
               if data[asset].index.tz is None:
                   data[asset] = data[asset].tz_localize('UTC')
               else:
                   data[asset] = data[asset].tz_convert('UTC')

       # Forward fill and backward fill missing values
       if isinstance(data, pd.DataFrame):
           data = data.ffill().bfill()
       else:
           for asset in data:
               data[asset] = data[asset].ffill().bfill()
"""
       return data
   except Exception as e:
       logging.error(f"Error fetching data for assets: {e}")
       print(f"Error fetching data for assets: {e}")
       return None



def apply_strategy_up_to_date(symbol, asset_data, strategy, params, current_date):
    """
    Parameters: 
        Symbol: defines the symbol of the asset that the function is
        working on.
        asset_data: our data frame containing the historical price data
        for the asset that we downloaded with YF
        strategy: the strategy being applied to th asset
        params: the parameters of the asset strategy
        current_date : date up untilt the stategy should be applied
        e.g 2024-09-01
    
    Overview:
        this function calculate the returns for the assets 
        up until the starting date of the backtesting period
        so we have all the numbers we need to apply the strategies
        i.e moving averages, std_devs etc..
    """
   

    strategy_func = globals()[strategy]
    strategy_result = strategy_func(asset_data.copy(), **params)

    # Handle cases where the current_date might not be in the strategy_result
    if current_date in strategy_result.index:
        return strategy_result.loc[current_date]
    """else:
        # Find the last available date before current_date
        available_dates = strategy_result.index[strategy_result.index <= current_date]
        if not available_dates.empty:
            last_date = available_dates[-1]
            return strategy_result.loc[last_date]
        else:
            logging.warning(f"No strategy data available for {symbol} up to {current_date}.")
            print(f"Warning: No strategy data available for {symbol} up to {current_date}.")
            return None"""

def backtest(start_date, end_date):
    """
    Parameters:
        start_date: the starting date of our backtesting
        end_date: the ending date of our backtesting
        
    Overview:
        This is the core function of the backtesting script. So
        this will be quite involved.
        
        Steps:
            1) we initialize the portfolio setting portfolio_value
               and available_cash to initial_capital
            2) We fetch the historical data for the assets
            3) Set common datetimes for all assets for consistency (to avoid errors etc...)
            4) We re-initialize our portfolio_df to store the assetsand
               the amounts invested in each asset
            5) iterate over each date, note this will be hours if we download hourly data
            6) iterate over each asset and apply strategy up to current date
               using apply_strategy_up_to_date
            7) we evaluate strategy signals, 1 buy, 0 hold, -1 sell
            8) we execute trades if a buy signal is detected we buy
                up to 10% of available cash in the asset. and if
                there is a sell signal we sell and the available cash
                is updated accordingly.
            9) we update the assets holdings value based off imported data
               note this can be any periodly however it is refereed to 
               as daily
            10) update total holdings as the sum of asset holdings and
               available cash at each time point
            11) save results we save the datafames into csv files
                for further analysis
    """
    # Initialize portfolio value locally
    portfolio_value = initial_capital
    available_cash = initial_capital

    # Fetch historical data for selected assets
    assets = best_strategies.keys()
    data = fetch_data_for_assets(assets, start_date, end_date)
    

    # Determine common dates across all assets
    if isinstance(data, pd.DataFrame):
        common_dates = data.index
    else:
        
        common_dates = data[next(iter(data))].index
        for asset in data:
            common_dates = common_dates.intersection(data[asset].index)

    #print(f"Number of common dates after alignment: {len(common_dates)}")
    logging.info(f"Number of common dates after alignment: {len(common_dates)}")

    # Initialize portfolio_df with Total Portfolio Value and individual holdings
    portfolio_df = pd.DataFrame(index=common_dates)
    portfolio_df['Total Portfolio Value'] = initial_capital

    for symbol in assets:
        portfolio_df[f'Holdings_{symbol}'] = 0.0

    
    for current_date in common_dates:
        
        logging.debug(f"Processing date: {current_date}")
        #print(f"Processing date: {current_date}")

        # Iterate over each asset
        for symbol in assets:
            if isinstance(data, pd.DataFrame):
                asset_df = data
            else:
                asset_df = data[symbol]

            # Check if data exists for the current date
            if isinstance(data, pd.DataFrame):
                if symbol not in data.columns.levels[0]:
                    continue
                asset_data = data[symbol].loc[:current_date].dropna()
            else:
                asset_data = asset_df.loc[:current_date].dropna()

            if asset_data.empty:
                continue  # No data available up to current_date

            # Get strategy and parameters
            strategy = best_strategies[symbol]['strategy']
            params = best_strategies[symbol]['parameters']

            # Map strategy name to function name
            mapped_strategy = strategy_mapping.get(strategy)
            if not mapped_strategy:
                logging.error(f"Strategy '{strategy}' for symbol '{symbol}' is not mapped.")
                print(f"Error: Strategy '{strategy}' for symbol '{symbol}' is not mapped.")
                continue

            # Apply strategy up to current date
            strategy_signal = apply_strategy_up_to_date(symbol, asset_data, mapped_strategy, params, current_date)

            if strategy_signal is None:
                continue  # Skip if no strategy signal

            # Get position and returns
            position = strategy_signal['position']
            strategy_return = strategy_signal['strategy_returns']
            current_price = strategy_signal['Close']

            # Debug: Print strategy signal
            logging.debug(f"Symbol: {symbol}, Date: {current_date}, Position: {position}, Price: {current_price}")
            print(f"Symbol: {symbol}, Date: {current_date}, Position: {position}, Price: {current_price}")

            # Execute trades based on position
            if position == 1:
                # Buy signal
                allocation_fraction = 0.02 # Allocate 10% of available cash per buy signal
                investment_amount = available_cash * allocation_fraction
                quantity = investment_amount // current_price  # Number of shares to buy (integer)
                if quantity > 0:
                    holdings[symbol] += quantity
                    available_cash -= quantity * current_price
                    # Log the trade
                    trade_logs.loc[len(trade_logs)] = [
                        current_date, symbol, 'BUY', quantity, current_price,
                        quantity * current_price, available_cash
                    ]
                    logging.info(f"BUY: {quantity} shares of {symbol} at ${current_price:.2f} on {current_date}")
                    print(f"BUY: {quantity} shares of {symbol} at ${current_price:.2f} on {current_date}")
            elif position == -1 and holdings[symbol] > 0:
                # Sell signal
                sell_quantity = holdings[symbol]
                sell_amount = sell_quantity * current_price
                holdings[symbol] = 0
                available_cash += sell_amount
                # Log the trade
                trade_logs.loc[len(trade_logs)] = [
                    current_date, symbol, 'SELL', sell_quantity, current_price,
                    sell_amount, available_cash
                ]
                logging.info(f"SELL: {sell_quantity} shares of {symbol} at ${current_price:.2f} on {current_date}")
                print(f"SELL: {sell_quantity} shares of {symbol} at ${current_price:.2f} on {current_date}")

            # Calculate daily return from holdings
            if holdings[symbol] > 0:
                asset_return = strategy_return
                daily_return = holdings[symbol] * (current_price * asset_return)
                portfolio_value += daily_return

                # Update individual asset holdings in portfolio_df
                portfolio_df.loc[current_date, f'Holdings_{symbol}'] = holdings[symbol] * current_price
            else:
                # No holdings, set to 0
                portfolio_df.loc[current_date, f'Holdings_{symbol}'] = 0.0

        # Calculate total holdings value
        total_holdings = portfolio_df.loc[current_date, [f'Holdings_{s}' for s in assets]].sum()
        # Update Total Portfolio Value
        portfolio_df.loc[current_date, 'Total Portfolio Value'] = total_holdings + available_cash

        # Debug: Log portfolio value
        logging.debug(f"Date: {current_date}, Portfolio Value: ${portfolio_value:.2f}, Available Cash: ${available_cash:.2f}")
        print(f"Date: {current_date}, Portfolio Value: ${portfolio_value:.2f}, Available Cash: ${available_cash:.2f}")

    # Save portfolio history and trade logs to CSV
    portfolio_df.to_csv('portfolio_history.csv')
    trade_logs.to_csv('trade_logs.csv', index=False)
    logging.info("Portfolio history saved to 'portfolio_history.csv'.")
    logging.info("Trade logs saved to 'trade_logs.csv'.")
    print("Backtest completed. Portfolio history and trade logs have been saved.")

    return portfolio_df, trade_logs


"""This code runs the backtesting function"""
if __name__ == "__main__":
    # Define backtest period
    start_date = '2024-09-01'
    end_date = '2024-10-01' # Sets end_date to today

    
    portfolio_df, trade_logs = backtest(start_date=start_date, end_date=end_date)

    """This Code Below calculates Total return and plots it"""

    if portfolio_df is not None and trade_logs is not None:
       

        # Calculate overall return
        overall_return = (portfolio_df['Total Portfolio Value'].iloc[-1] - initial_capital) / initial_capital
        print(f"\nOverall Return: {overall_return:.2%}")

       

        try:
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_df.index, portfolio_df['Total Portfolio Value'], label='Total Portfolio Value')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.title('Portfolio Value Over Time')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('portfolio_value_over_time.png')
            plt.show()
            print("\nPortfolio performance plot saved as 'portfolio_value_over_time.png'.")
        except Exception as e:
            logging.warning(f"Error during visualization: {e}")
            print(f"Warning: Could not generate plot due to error: {e}")
    else:
        print("Backtest failed. Please check 'trading.log' for more details.")


