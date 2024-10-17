import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, skew, kurtosis
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Data Download & Preprocessing
# -------------------------
# Download the data from Yahoo Finance for the specified date range
data = yf.download('^NSEI', start='2013-01-21', end='2024-07-31', interval='1d')
data.to_csv("E:/ML Gate/data.csv", index=True)

# Ensure the index is a datetime object
data.index = pd.to_datetime(data.index)

# Function to calculate daily returns
def calculate_daily_returns(data):
    return data['Close'].pct_change()

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD Histogram
def calculate_macd_histogram(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_histogram

# Function to calculate VWAP
def calculate_vwap(data):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

# Function to calculate Bollinger Band Width
def calculate_bollinger_band_width(data, window=20, num_std_dev=2):
    middle_band = data['Adj Close'].rolling(window=window).mean()
    std_dev = data['Adj Close'].rolling(window=window).std()
    upper_band = middle_band + (num_std_dev * std_dev)
    lower_band = middle_band - (num_std_dev * std_dev)
    bollinger_band_width = upper_band - lower_band
    return bollinger_band_width

# Assuming 'Adj Close' is the column for the adjusted closing prices
def calculate_log_returns(data):
    # Log return formula applied to the 'Adj Close' price
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    return log_returns

# Calculate features on the entire dataset

data['Log Returns'] = calculate_log_returns(data)
data['RSI'] = calculate_rsi(data)
data['MACD_Histogram'] = calculate_macd_histogram(data)
data['VWAP'] = calculate_vwap(data)
data['Bollinger_Band_Width'] = calculate_bollinger_band_width(data)
data['50_day_MA'] = data['Close'].rolling(window=50).mean()
data['20_day_MA'] = data['Close'].rolling(window=20).mean()
data['9_day_MA'] = data['Close'].rolling(window=9).mean()
data['Skewness'] = data['Log Returns'].rolling(window=20).apply(lambda x: skew(x, bias=False))
data['Kurtosis'] = data['Log Returns'].rolling(window=20).apply(lambda x: kurtosis(x, bias=False))

# -------------------------
# 2. Volatility Calculation
# -------------------------
# Drop NaN values resulting from initial calculations
data = data.dropna(subset=['Log Returns']).copy()

# Function to calculate fixed window rolling volatility (standard deviation)
def calculate_fixed_window_volatility(data, window_size=20):
    return data.rolling(window=window_size).std()

# Function to determine dynamic window size based on volatility
def determine_dynamic_window_size(volatility, min_window=5, max_window=20, epsilon=1e-8):
    # Add epsilon to volatility to avoid division by zero or near-zero values
    inverse_volatility = 1 / (volatility + epsilon)
    
    # Normalize the inverse volatility to scale between min and max window sizes
    normalized_window_size = (inverse_volatility - inverse_volatility.min()) / (inverse_volatility.max() - inverse_volatility.min())
    
    # Scale the normalized window size to the specified window range
    dynamic_window_size = normalized_window_size * (max_window - min_window) + min_window
    
    # Fill any NaN values with the minimum window size and convert to integers
    return dynamic_window_size.fillna(min_window).astype(int)

# Calculate volatility and dynamic window sizes
data.loc[:, 'volatility'] = calculate_fixed_window_volatility(data['Log Returns'])
data.loc[:, 'dynamic_window_sizes'] = determine_dynamic_window_size(data['volatility'])

# Function to calculate rolling variance using exponential moving average (EMA)
def calculate_rolling_variance(data, window_size):
    return data.ewm(span=window_size).var()

# Initialize a list to store dynamic rolling variances, pre-filled with NaN values
dynamic_rolling_variances = [np.nan] * len(data)

# Calculate dynamic rolling variances for each data point
for idx, (_, row) in enumerate(data.iterrows()):
    window_size = int(row['dynamic_window_sizes'])  # Get dynamic window size for this row
    
    # Check if there are enough data points behind the current data point to create a window
    if idx < window_size - 1:
        continue  # Skip if not enough data points for rolling variance
    
    # Calculate rolling variance for the previous 'window_size' rows including current index
    start_idx = idx - window_size + 1
    end_idx = idx + 1
    data_window = data['Log Returns'].iloc[start_idx : end_idx]
    
    # Calculate rolling variance
    dynamic_rolling_variance = calculate_rolling_variance(data_window, window_size).iloc[-1]

    # Store the result in the list at the correct index
    dynamic_rolling_variances[idx] = dynamic_rolling_variance

# Add the dynamic rolling variances as a new column to your DataFrame
data.loc[:, 'dynamic_rolling_variances'] = dynamic_rolling_variances

# -------------------------
# 3. Bayesian Updates
# -------------------------
def update_posterior(x_i, mu_prior, kappa_prior, alpha_prior, beta_prior):
    """Update posterior parameters using the Normal-Inverse-Gamma conjugate prior."""
    # Update kappa
    kappa_posterior = kappa_prior + 1
    
    # Update mu
    mu_posterior = (kappa_prior * mu_prior + x_i) / kappa_posterior
    
    # Update alpha
    alpha_posterior = alpha_prior + 0.5
    
    # Update beta
    beta_posterior = beta_prior + (kappa_prior * (x_i - mu_prior) ** 2) / (2 * kappa_posterior)
    
    return mu_posterior, kappa_posterior, alpha_posterior, beta_posterior

def calculate_posterior_variance(kappa_posterior, alpha_posterior, beta_posterior):
    """Calculate the posterior variance of mu."""
    # Expected variance (sigma^2)
    expected_variance = beta_posterior / (alpha_posterior - 1)
    
    # Posterior variance of mu
    sigma_posterior_squared = expected_variance / kappa_posterior
    
    return sigma_posterior_squared

# Initial priors for Bayesian updates
mu_prior = 0  # Prior mean
kappa_prior = 1
alpha_prior = 3  # Prior alpha for Inverse-Gamma
beta_prior = 2  # Prior beta for Inverse-Gamma

# Initialize dictionaries or lists to store the results
updated_bayes_means = {}
updated_bayes_sds = {}
cdfs = {}

# Loop through the data for Bayesian updates
for i, row in data.iterrows():
    x_i = row['Log Returns']
    
    # Update posterior parameters
    mu_posterior, kappa_posterior, alpha_posterior, beta_posterior = update_posterior(
        x_i, mu_prior, kappa_prior, alpha_prior, beta_prior)
    
    # Calculate posterior variance of mu
    sigma_posterior_squared = calculate_posterior_variance(
        kappa_posterior, alpha_posterior, beta_posterior)
    
    # Store posterior mean and standard deviation
    updated_bayes_means[i] = mu_posterior
    updated_bayes_sds[i] = np.sqrt(sigma_posterior_squared)
    
    # Calculate CDF
    cdfs[i] = norm.cdf(x_i, mu_posterior, np.sqrt(sigma_posterior_squared))
    
    # Update priors for next iteration
    mu_prior, kappa_prior, alpha_prior, beta_prior = mu_posterior, kappa_posterior, alpha_posterior, beta_posterior

# Add Bayesian results to the dataset
data.loc[:, 'updated_bayes_means'] = data.index.map(updated_bayes_means)
data.loc[:, 'updated_bayes_sds'] = data.index.map(updated_bayes_sds)
data.loc[:, 'CDF'] = data.index.map(cdfs)


# -------------------------
# 4. Final Dataset Selection
# -------------------------

# Adding a separate column of Adj Close for normalization
data['Nor_Adj_Close'] = data['Adj Close']

# Select the columns you want to keep
selected_columns = [
    'Adj Close', 'Nor_Adj_Close', 'RSI', 'MACD_Histogram', 'VWAP', 'Bollinger_Band_Width',
    '50_day_MA', '20_day_MA', '9_day_MA', 'Skewness', 'Kurtosis',
    'dynamic_rolling_variances', 'CDF'
    # Add or remove columns as needed
]

# Select the desired columns and make a copy
data = data[selected_columns].copy()


# Drop rows with missing values in selected columns
data = data.dropna().copy()