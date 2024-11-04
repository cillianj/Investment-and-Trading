import sys
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import json
import jsonschema
from jsonschema import validate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import os

# ============================
# 1. Environment Setup
# ============================

# Load environment variables from .env file (if needed)
load_dotenv()

# Initialize Logging
logging.basicConfig(
    filename='live_trading.log',
    level=logging.DEBUG,  # Set to DEBUG for detailed logs; change to INFO in production
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ============================
# 2. JSON Schema Definition
# ============================

# Define JSON schema for validation
schema = {
    "type": "object",
    "patternProperties": {
        "^[A-Z0-9=-]+$": {  # Updated regex to include hyphen
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

# ============================
# 3. Load and Validate Configuration
# ============================

def load_best_strategies(json_path='best_expanded_strategies.json'):
    """
    Load and validate strategies from a JSON file.
    """
    global best_strategies  # Declare as global
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
        sys.exit(1)  # Terminate the script

# Load strategies initially
load_best_strategies()

# ============================
# 4. Strategy Mapping Setup
# ============================

# Define a mapping from strategy names in JSON to function names in the script
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

# ============================
# 5. Strategy Parameters Setup
# ============================

# Define default parameters for each strategy
default_parameters = {
    'buy_and_hold_strategy': {},
    'rsi_strategy': {'window': 14, 'overbought': 70, 'oversold': 30},
    'mean_reversion_strategy': {'window': 20, 'threshold': 0.02, 'mode': 'deviation'},
    'trend_following_strategy': {'short_window': 20, 'long_window': 50},
    'breakout_strategy': {'window': 20, 'quantile': 0.75},
    'bollinger_bands_strategy': {'window': 20, 'num_std': 2},
    'momentum_strategy': {'window': 20, 'threshold': 0.02},
    'moving_average_crossover_strategy': {'short_window': 50, 'long_window': 200}
    # Add more strategies as needed
}

# Validate and set default parameters for each asset's strategy
for symbol, strategy_info in best_strategies.items():
    strategy = strategy_info.get('strategy')
    parameters = strategy_info.get('parameters', {})
    
    # Map the strategy name to the function name
    mapped_strategy = strategy_mapping.get(strategy)
    
    if mapped_strategy and mapped_strategy in default_parameters:
        # Apply default parameters
        for param, default in default_parameters[mapped_strategy].items():
            parameters.setdefault(param, default)
        best_strategies[symbol]['parameters'] = parameters
    else:
        logging.warning(f"No default parameters for strategy '{strategy}' of symbol '{symbol}'.")
        print(f"Warning: No default parameters for strategy '{strategy}' of symbol '{symbol}'.")

# ============================
# 6. Portfolio Initialization
# ============================

initial_capital = 1_000_000  # Starting with $1,000,000

# Initialize holdings: Start with 0 invested in each asset
holdings = {
    symbol: 0 for symbol in best_strategies.keys()
}

# Initialize available cash
available_cash = initial_capital

# Initialize a DataFrame to store portfolio values over time
portfolio_history = pd.DataFrame(columns=['Date', 'Total Portfolio Value'])
portfolio_history.set_index('Date', inplace=True)

# Initialize a DataFrame to store trade logs
trade_logs = pd.DataFrame(columns=[
    'DateTime', 'Symbol', 'Action', 'Quantity',
    'Price', 'Total Value', 'Uninvested Cash'
])

# ============================
# 7. Strategy Function Definitions
# ============================

def buy_and_hold_strategy(data, **params):
    """
    Buy and Hold Strategy: Always hold the asset.
    """
    data['position'] = 1
    data['strategy_returns'] = data['Close'].pct_change().fillna(0)
    return data

def mean_reversion_strategy(data, window=20, threshold=0.02, mode='deviation'):
    """
    Mean Reversion Strategy:
    - Deviation Mode: Buy when price deviates below the moving average by a threshold.
    - Crossover Mode: Buy when price crosses above the moving average.
    """
    data['moving_avg'] = data['Close'].rolling(window).mean()
    data['signal'] = 0

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
    """
    Trend Following Strategy: Buy when the short-term moving average crosses above the long-term moving average.
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
    Breakout Strategy: Buy when volatility exceeds a certain quantile threshold.
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
    Momentum Strategy: Buy when momentum is above a threshold, sell when below.
    """
    data['momentum'] = data['Close'].pct_change(periods=window)
    data['signal'] = 0
    data['signal'] = np.where(data['momentum'] > threshold, 1, data['signal'])
    data['signal'] = np.where(data['momentum'] < -threshold, -1, data['signal'])
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
    return data

def rsi_strategy(data, window=14, overbought=70, oversold=30):
    """
    RSI Strategy: Buy when RSI is below oversold threshold, sell when above overbought threshold.
    """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['signal'] = 0
    data['signal'] = np.where(data['RSI'] < oversold, 1, data['signal'])
    data['signal'] = np.where(data['RSI'] > overbought, -1, data['signal'])
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
    return data

def moving_average_crossover_strategy(data, short_window=50, long_window=200):
    """
    Moving Average Crossover Strategy: Buy when short MA crosses above long MA, sell when it crosses below.
    """
    data['short_ma'] = data['Close'].rolling(window=short_window).mean()
    data['long_ma'] = data['Close'].rolling(window=long_window).mean()
    data['signal'] = 0
    data['signal'] = np.where(data['short_ma'] > data['long_ma'], 1, data['signal'])
    data['signal'] = np.where(data['short_ma'] < data['long_ma'], -1, data['signal'])
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
    return data

def bollinger_bands_strategy(data, window=20, num_std=2):
    """
    Bollinger Bands Strategy: Buy when price touches lower band, sell when it touches upper band.
    """
    data['ma'] = data['Close'].rolling(window).mean()
    data['std'] = data['Close'].rolling(window).std()
    data['upper_band'] = data['ma'] + (data['std'] * num_std)
    data['lower_band'] = data['ma'] - (data['std'] * num_std)
    data['signal'] = 0
    data['signal'] = np.where(data['Close'] < data['lower_band'], 1, data['signal'])
    data['signal'] = np.where(data['Close'] > data['upper_band'], -1, data['signal'])
    data['desired_position'] = data['signal'].shift().fillna(0)
    data['position'] = data['desired_position']
    data['strategy_returns'] = data['desired_position'] * data['Close'].pct_change().fillna(0)
    return data

# ============================
# 8. Live Data Fetching Function
# ============================

def fetch_latest_price(symbol):
    """
    Fetch the latest available price for a symbol using yfinance.
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='1m')  # 1-minute intervals
        if data.empty:
            logging.warning(f"No data fetched for symbol {symbol}.")
            return None
        latest = data.iloc[-1]
        latest_price = latest['Close']
        logging.debug(f"Fetched latest price for {symbol}: {latest_price}")
        return latest_price
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# ============================
# 9. Order Execution Function (Simulated)
# ============================

def execute_trade(symbol, action, quantity, price, current_time):
    """
    Simulate the execution of a trade by updating holdings and cash.
    """
    global available_cash
    if action == 'BUY':
        cost = quantity * price
        if available_cash >= cost:
            holdings[symbol] += quantity
            available_cash -= cost
            trade_logs.loc[len(trade_logs)] = [
                current_time, symbol, action, quantity, price,
                cost, available_cash
            ]
            logging.info(f"BUY: {quantity} shares of {symbol} at ${price:.2f} on {current_time}")
            print(f"BUY: {quantity} shares of {symbol} at ${price:.2f} on {current_time}")
        else:
            logging.warning(f"Insufficient cash to BUY {quantity} shares of {symbol} at ${price:.2f}")
            print(f"Warning: Insufficient cash to BUY {quantity} shares of {symbol} at ${price:.2f}")
    elif action == 'SELL':
        if holdings[symbol] >= quantity:
            revenue = quantity * price
            holdings[symbol] -= quantity
            available_cash += revenue
            trade_logs.loc[len(trade_logs)] = [
                current_time, symbol, action, quantity, price,
                revenue, available_cash
            ]
            logging.info(f"SELL: {quantity} shares of {symbol} at ${price:.2f} on {current_time}")
            print(f"SELL: {quantity} shares of {symbol} at ${price:.2f} on {current_time}")
        else:
            logging.warning(f"Insufficient holdings to SELL {quantity} shares of {symbol}")
            print(f"Warning: Insufficient holdings to SELL {quantity} shares of {symbol}")
    else:
        logging.error(f"Invalid action '{action}' for trade execution.")
        print(f"Error: Invalid action '{action}' for trade execution.")

# ============================
# 10. Live Trading Function
# ============================

def live_trading(poll_interval=60):
    """
    Main function to run the live paper trading algorithm.
    """
    global available_cash  # To modify the available_cash variable
    
    symbols = list(best_strategies.keys())
    print("Starting live paper trading...")
    logging.info("Starting live paper trading...")
    
    while True:
        current_time = datetime.now()
        portfolio_value = available_cash
        print(f"\n=== {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        logging.debug(f"=== {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        for symbol in symbols:
            latest_price = fetch_latest_price(symbol)
            if latest_price is None:
                continue  # Skip if no data
            
            # Create a DataFrame for the latest price
            data = pd.DataFrame({
                'Open': [latest_price],
                'High': [latest_price],
                'Low': [latest_price],
                'Close': [latest_price],
                'Volume': [0]
            }, index=[current_time])
            
            # Get strategy and parameters
            strategy = best_strategies[symbol]['strategy']
            params = best_strategies[symbol]['parameters']
            
            # Map strategy name to function name
            mapped_strategy = strategy_mapping.get(strategy)
            if not mapped_strategy:
                logging.error(f"Strategy '{strategy}' for symbol '{symbol}' is not mapped.")
                print(f"Error: Strategy '{strategy}' for symbol '{symbol}' is not mapped.")
                continue
            
            # Apply strategy
            strategy_func = globals()[mapped_strategy]
            strategy_result = strategy_func(data.copy(), **params)
            
            # Extract the latest strategy signals
            position = strategy_result['position'].iloc[-1]
           
            current_price = strategy_result['Close'].iloc[-1]
            
            # Determine action based on position
            current_holdings = holdings[symbol]
            action = None
            quantity = 0
            
            if position == 1 and current_holdings == 0:
                action = 'BUY'
                allocation_fraction = 0.1  # Allocate 10% of available cash per buy signal
                investment_amount = available_cash * allocation_fraction
                quantity = int(investment_amount // current_price)
                if quantity <= 0:
                    logging.warning(f"Insufficient cash to BUY {symbol}")
                    print(f"Warning: Insufficient cash to BUY {symbol}")
                    action = None
            elif position == -1 and current_holdings > 0:
                action = 'SELL'
                quantity = current_holdings  # Sell all holdings
            
            if action and quantity > 0:
                execute_trade(symbol, action, quantity, current_price, current_time)
            
            # Update portfolio value
            portfolio_value += holdings[symbol] * current_price
            print(f"{symbol}: Holdings={holdings[symbol]} Shares, Latest Price=${current_price:.2f}")
            logging.debug(f"{symbol}: Holdings={holdings[symbol]} Shares, Latest Price=${current_price:.2f}")
        
        # Log total portfolio value
        portfolio_history.loc[current_time] = portfolio_value
        print(f"Total Portfolio Value: ${portfolio_value:.2f}")
        logging.info(f"Total Portfolio Value: ${portfolio_value:.2f}, Available Cash: ${available_cash:.2f}")
        
        # Sleep until next poll
        time.sleep(poll_interval)

# ============================
# 11. Execute Live Trading
# ============================

if __name__ == "__main__":
    try:
        live_trading(poll_interval=60)  # Poll every 60 seconds
    except KeyboardInterrupt:
        logging.info("Live trading stopped by user.")
        print("Live trading stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error in live trading: {e}")
        print(f"Unexpected error in live trading: {e}")
