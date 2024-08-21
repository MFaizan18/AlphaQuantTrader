![image](https://github.com/user-attachments/assets/cbe19caa-904a-4255-87f1-cabe30d99040)# AlphaQuantTrader

Welcome to the AlphaQuantTrader project! This powerful trading bot is designed to revolutionize the way you approach algorithmic trading by leveraging advanced machine learning techniques and Bayesian statistics.

## 1) What Exactly is AlphaQuantTrader?
AlphaQuantTrader is a sophisticated trading bot that utilizes reinforcement learning to automate and optimize stock trading decisions. It operates within a custom-built trading environment that simulates real-world market conditions, allowing the bot to make informed buy, hold, or sell decisions based on historical financial data. The bot is powered by a deep Q-network (DQN) model, which learns to maximize portfolio returns by continuously adapting to changing market conditions.

Additionally, AlphaQuantTrader incorporates Bayesian statistical methods to dynamically adjust risk and enhance decision-making, ensuring robust performance even in volatile markets.

## 2) Key Features

**2.1) Custom Trading Environment:**
AlphaQuantTrader is built on a custom Gym environment that accurately simulates trading activities, including transaction costs and portfolio management, offering a realistic trading experience.

**2.2) Deep Q-Network (DQN):**
At its core, the bot uses a DQN model to predict and execute the most profitable trading actions, learning from past market data to improve its strategies over time.

**2.3) Bayesian Risk Management:**
The bot integrates Bayesian updating techniques to adjust its risk management strategies dynamically, taking into account market volatility and uncertainty.

**2.4) Historical Data Processing:**
AlphaQuantTrader preprocesses and utilizes historical market data, including adjusted closing prices, daily returns, and volatility, to inform its trading decisions.

**2.5) Portfolio Optimization:**
Through reinforcement learning, the bot continuously seeks to optimize its portfolio by balancing risk and reward, aiming to maximize long-term gains.

## 3) Why Use AlphaQuantTrader?

AlphaQuantTrader is ideal for traders and developers looking to automate and enhance their trading strategies with cutting-edge AI. Whether you're aiming to reduce the manual effort in trading or seeking a robust system that adapts to market changes, AlphaQuantTrader provides the tools and intelligence to help you stay ahead in the financial markets. With its combination of deep learning and Bayesian techniques, AlphaQuantTrader offers a strategic edge that goes beyond traditional trading algorithms.

# 4) Model Overview

Let's start by importimg the necessary libraries

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import gym
from gym import spaces
import random
from collections import deque, namedtuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, TimeDistributed, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
```
## 5) Preprocessing

The preprocessing phase is crucial for preparing raw financial data into a form suitable for training the reinforcement learning model. This step involves acquiring historical market data, cleaning it, and engineering the necessary features to ensure that the model receives meaningful input for effective learning.

**5.1) Data Acquisition**

Historical stock price data for the National Stock Exchange of India (^NSEI) index is downloaded using the `yfinance` library. The dataset spans a 10-year period, from January 1, 2014, to July 31, 2024, and provides the foundation for training and testing the reinforcement learning model.

```python
data = yf.download('^NSEI', start='2014-01-01', end='2024-07-31', interval='1d')
```
Here's a glimpse of the data we're working with. The first 10 rows of the data are as follows:
```
| Date       | Open         | High         | Low          | Close        | Adj Close   | Volume |
|------------|--------------|--------------|--------------|--------------|-------------|--------|
| 02-01-2014 | 6301.25      | 6358.299805  | 6211.299805  | 6221.149902  | 6221.149902 | 158100 |
| 03-01-2014 | 6194.549805  | 6221.700195  | 6171.25      | 6211.149902  | 6211.149902 | 139000 |
| 06-01-2014 | 6220.850098  | 6224.700195  | 6170.25      | 6191.450195  | 6191.450195 | 118300 |
| 07-01-2014 | 6203.899902  | 6221.5       | 6144.75      | 6162.25      | 6162.25     | 138600 |
| 08-01-2014 | 6178.049805  | 6192.100098  | 6160.350098  | 6174.600098  | 6174.600098 | 146900 |
| 09-01-2014 | 6181.700195  | 6188.049805  | 6148.25      | 6168.350098  | 6168.350098 | 150100 |
| 10-01-2014 | 6178.850098  | 6239.100098  | 6139.600098  | 6171.450195  | 6171.450195 | 159900 |
| 13-01-2014 | 6189.549805  | 6288.200195  | 6189.549805  | 6272.75      | 6272.75     | 135000 |
| 14-01-2014 | 6260.25      | 6280.350098  | 6234.149902  | 6241.850098  | 6241.850098 | 110200 |
| 15-01-2014 | 6265.950195  | 6325.200195  | 6265.299805  | 6320.899902  | 6320.899902 | 145900 |
```
And the last 10 rows of the data are as follows:

```
| Date       | Open         | High         | Low          | Close        | Adj Close   | Volume |
|------------|--------------|--------------|--------------|--------------|-------------|--------|
| 16-07-2024 | 24615.90039  | 24661.25     | 24587.65039  | 24613        | 24613       | 283200 |
| 18-07-2024 | 24543.80078  | 24837.75     | 24504.44922  | 24800.84961  | 24800.84961 | 350900 |
| 19-07-2024 | 24853.80078  | 24854.80078  | 24508.15039  | 24530.90039  | 24530.90039 | 343800 |
| 22-07-2024 | 24445.75     | 24595.19922  | 24362.30078  | 24509.25     | 24509.25    | 324200 |
| 23-07-2024 | 24568.90039  | 24582.55078  | 24074.19922  | 24479.05078  | 24479.05078 | 436400 |
| 24-07-2024 | 24444.94922  | 24504.25     | 24307.25     | 24413.5      | 24413.5     | 366600 |
| 25-07-2024 | 24230.94922  | 24426.15039  | 24210.80078  | 24406.09961  | 24406.09961 | 391800 |
| 26-07-2024 | 24423.34961  | 24861.15039  | 24410.90039  | 24834.84961  | 24834.84961 | 383800 |
| 29-07-2024 | 24943.30078  | 24999.75     | 24774.59961  | 24836.09961  | 24836.09961 | 355000 |
| 30-07-2024 | 24839.40039  | 24971.75     | 24798.65039  | 24857.30078  | 24857.30078 | 385000 |
```











