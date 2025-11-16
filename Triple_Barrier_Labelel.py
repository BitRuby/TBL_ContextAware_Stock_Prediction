#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.optimize import minimize
from Calculate_Returns import Calculate_Returns


# In[ ]:


LABEL_BULLISH = "Bullish"
LABEL_BEARISH = "Bearish"
LABEL_NEUTRAL = "Neutral"

param_grid = {
    'volatility_period': [8, 9, 10, 11, 12, 13, 14, 15],
    'upper_barrier_factor': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    'lower_barrier_factor': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    'vertical_barrier': [8, 9, 10, 11, 12, 13, 14, 15],
}

label_map = {
    LABEL_BEARISH: 0,
    LABEL_NEUTRAL: 1,
    LABEL_BULLISH: 2
}       

class Triple_Barrier_Labelel():
    def calculate_barriers(self, price, volatility, Fu, Fl):
        """
        Calculates the upper and lower barriers for the Triple Barrier Labeling technique.

        The Triple Barrier Labeling method is used to assign labels to price data based on upper and lower barriers. 
        These barriers are computed using the current price, historical volatility, and two user-defined parameters 
        that represent scaling factors for the upper and lower barriers.

        Parameters:
            price (float): The current price of the asset.
            volatility (float): The historical volatility of the asset, typically calculated as the standard deviation of returns.
            Fu (float): A scaling factor for the upper barrier. It represents the multiplier of volatility added to the price.
            Fl (float): A scaling factor for the lower barrier. It represents the multiplier of volatility subtracted from the price.

        Returns:
            tuple: A tuple containing two values:
                - `upper` (float): The computed upper barrier.
                - `lower` (float): The computed lower barrier.

        Example:
            # Assuming `price` is 100, `volatility` is 0.02 (2%), `Fu` is 0.05, and `Fl` is 0.05.
            upper, lower = calculate_barriers(100, 0.02, 0.05, 0.05)
            # The upper barrier will be 105.0, and the lower barrier will be 95.0.

        Notes:
            - The upper and lower barriers are typically used in the context of the Triple Barrier Labeling method 
              to classify price movements as "up," "down," or "neutral" based on whether the price crosses 
              the upper or lower barrier.
            - The `Fu` and `Fl` parameters allow for flexible adjustment of the barrier levels based on the asset's 
              volatility and the desired sensitivity of the labeling.
        """
        upper = price + price * volatility * Fu
        lower = price - price * volatility * Fl
        return upper, lower
    
    def assign_labels(self, data, fu, fl, vt):
        """
        Assigns labels to the stock data based on the Triple Barrier Labeling method.

        This method processes each row of stock data, calculating upper and lower barriers using historical volatility 
        and scaling factors (Fu, Fl). It assigns labels ("bullish", "bearish", or "neutral") depending on whether the 
        price crosses the upper or lower barriers within a specified time window (vt). If neither barrier is hit 
        within the window, a "neutral" label is assigned. 

        Parameters:
            data (pd.DataFrame): A DataFrame containing stock data. It should have columns like 'close', 'high', 'low', and 'volatility'.
            fu (float): A scaling factor for the upper barrier (multiplication with volatility).
            fl (float): A scaling factor for the lower barrier (multiplication with volatility).
            vt (int): The volatility time window (in days) within which the price should hit a barrier for labeling.

        Returns:
            list: A list of dictionaries, each containing information about the calculated barriers, the start and end date of the window,
                  and the assigned label. The possible labels are:
                  - LABEL_BULLISH: If the price crosses the upper barrier.
                  - LABEL_BEARISH: If the price crosses the lower barrier.
                  - LABEL_NEUTRAL: If the price does not cross any barrier within the given time window.
                  Each dictionary also includes the previous label ('prev_label').

        Example:
            # Assuming `data` is a DataFrame with stock data and the necessary columns:
            labels = assign_labels(data, fu=0.4, fl=0.5, vt=5)
            # The function will return labels for each price point, indicating whether it's bullish, bearish, or neutral.

        Notes:
            - The method uses `calculate_barriers` to get the upper and lower barriers for each row in the data.
            - The label assignment is based on whether the stock price crosses the upper or lower barrier within the given time window.
            - The method uses `prev_label` to keep track of the previous label assigned, ensuring continuity in labeling.
            - The function skips creating new labels if the time window is too short (less than 3 days) or if there are fewer than 2 days left in the dataset.

        """
        barriers = None
        prev_label = None
        labels = []
        # loop over all data rows
        for index, day in data.iterrows():
            # check if barriers are calculated if not, calculate and set starting day
            if barriers == None:
                barriers = self.calculate_barriers(day["close"], day["volatility"], fu, fl)
                start_day = index
            days_last = abs((start_day - index).days)
            # If window is shorter than 3 days or only 2 days left to end of dataset, skip creating new labels
            if days_last < 3 and abs((start_day - data.index[-1]).days) > 2:
                continue
            # If price dont hit upper or lower barrier in specific vt period, assign neutral label
            if days_last == vt:
                labels.append({"lower_barrier": barriers[1], "higher_barrier": barriers[0], "start": start_day, "end": index, "label": LABEL_NEUTRAL, "prev_label": prev_label})
                barriers = None
                prev_label = LABEL_NEUTRAL
            # If price hit upper barrier assign bullish label
            elif day["high"] > barriers[0]:
                labels.append({"lower_barrier": barriers[1], "higher_barrier": barriers[0], "start": start_day, "end": index, "label": LABEL_BULLISH, "prev_label": prev_label})
                barriers = None
                prev_label = LABEL_BULLISH
            # If price hit lower barrier assign bearish label
            elif day["low"] < barriers[1]:
                labels.append({"lower_barrier": barriers[1], "higher_barrier": barriers[0], "start": start_day, "end": index, "label": LABEL_BEARISH, "prev_label": prev_label})
                barriers = None
                prev_label = LABEL_BEARISH
        return labels
    
    def get_daily_volatility(self, df, span=14, std_coef=(-1.0, 0.5)):
        """
        Calculates the daily volatility of a stock based on the percentage change in price relative to the previous day.

        The method computes the daily returns (percentage change) by aligning each current day's price with the previous day's price, 
        even when the index has gaps (e.g., weekends or missing days). It then applies an exponentially weighted moving standard deviation 
        (EWMA) over the specified time window (`span`) to estimate the daily volatility.

        Parameters:
            df (pd.DataFrame): A DataFrame containing stock price data. It should have a 'close' column representing the closing prices.
            span (int, optional): The span for the exponentially weighted moving standard deviation. Default is 14 days.
            std_coef (Tuple[float, float]):
                Threshold multipliers relative to rolling volatility mean:
                    - Negative coef = lower bound (bullish if below this)
                    - Positive coef = upper bound (bearish if above this)
                Default (-1.0, 0.5).
    
        Returns:
            pd.Series: A Series containing the daily volatility for each day, based on the exponentially weighted moving standard deviation 
                       of the daily returns.

        Example:
            # Assuming `df` is a DataFrame with stock price data and a 'close' column:
            volatility = get_daily_volatility(df, span=14)
            # `volatility` will contain the daily volatility values for the stock.

        Notes:
            - The method accounts for missing data or gaps in the time series (e.g., weekends or holidays).
            - The `span` parameter controls the smoothing of the volatility estimate. A smaller span gives more weight to recent data.
            - The method uses an exponentially weighted moving average (EWMA) to compute the standard deviation of returns, which gives more 
              weight to recent observations.
        """
        prev_day_start = df.close.index.searchsorted(df.close.index - pd.Timedelta(days=1))
        prev_day_start = prev_day_start[prev_day_start > 0]
        prev_day_start = pd.Series(df.close.index[prev_day_start - 1], index=df.close.index[df.close.shape[0] - prev_day_start.shape[0]:])
        daily_returns = df.close.loc[prev_day_start.index] / df.close.loc[prev_day_start.values].values - 1
        vol = daily_returns.ewm(span=span).std()
        
        rolling_mean = vol.rolling(window=span).mean()
        rolling_std = vol.rolling(window=span).std()

        lower = rolling_mean + std_coef[0] * rolling_std
        upper = rolling_mean + std_coef[1] * rolling_std
        
        description = pd.Series(index=vol.index, dtype="object")
        description[vol > upper] = "bearish"
        description[vol < lower] = "bullish"
        description[(vol <= upper) & (vol >= lower)] = "neutral"
        return vol, description
    
    def map_data(self, labels, day, key):
        day = pd.to_datetime(day)
        for label in labels:
            start = pd.to_datetime(label['start'])
            end = pd.to_datetime(label['end'])
            if start <= day <= end:
                return label[key]
        return None  # If day is outside all defined windows
    
    def map_start_window(self, labels, day):
        day = pd.to_datetime(day)
        return any(pd.to_datetime(label['start']) == day for label in labels)
    
    def transform(self, df, vol, fu, fl, vt):
        """
        Transforms stock price data by calculating features such as daily volatility, trading signals, and barriers 
        based on predefined thresholds and labels, which are used to support trading strategies.

        The method performs the following transformations:
        1. Calculates the daily volatility using the `get_daily_volatility` method.
        2. Assigns labels to the stock price data based on the upper and lower barrier thresholds and the volatility.
        3. Maps labels and associated barriers (upper and lower) to the DataFrame.
        4. Creates signals (buy, sell, or neutral) based on the labels.
        5. Computes the percentage changes for the target price (take profit) and stop loss levels.
        6. Ensures that negative values for the target price and stop loss are replaced with 0, to maintain logical consistency.

        Parameters:
            df (pd.DataFrame): A DataFrame containing stock price data. It should have columns such as 'close' for the closing prices.
            vol (int): The span parameter used for calculating volatility in the `get_daily_volatility` method.
            fu (float): The factor used to calculate the upper barrier.
            fl (float): The factor used to calculate the lower barrier.
            vt (float): An additional threshold factor used in label assignment.

        Returns:
            pd.DataFrame: A transformed DataFrame containing the following new columns:
                - 'volatility': The daily volatility of the stock price.
                - 'label': A label indicating the market trend (e.g., bullish, bearish).
                - 'lower_barriers': The calculated lower barrier for the stock price.
                - 'upper_barriers': The calculated upper barrier for the stock price.
                - 'previous_label': The label from the previous trading day.
                - 'window_start': A boolean column indicating if the current day marks the start of a new window.
                - 'signals': A column indicating trading signals (buy, sell, or neutral).
                - 'tp_stop': The percentage change for the take profit level.
                - 'sl_stop': The percentage change for the stop loss level.

        Example:
            # Assuming `df` is a DataFrame with stock price data and a 'close' column:
            transformed_data = transform(df, vol=14, fu=0.02, fl=0.02, vt=0.1)
            # `transformed_data` will contain the enhanced features, including volatility, labels, barriers, and signals.

        Notes:
            - The method uses `get_daily_volatility` to estimate daily price volatility and applies a smoothing technique (EWMA) over the specified `span` to calculate volatility.
            - The `assign_labels` method is used to determine the market condition (e.g., bullish or bearish) based on the barriers (`fu` and `fl`), volatility (`vol`), and the additional threshold (`vt`).
            - The 'signals' column provides actionable trading decisions: 1 for "buy", 2 for "sell", and 0 for "neutral".
            - The method ensures that invalid values (negative target prices or stop loss levels) are replaced with 0 to avoid unrealistic trading decisions.

        """
        copy = df.copy()
        copy["volatility"], copy['volatility_label'] = self.get_daily_volatility(copy, vol)
        labels = self.assign_labels(copy, fu, fl, vt)
        copy['label'] = copy.apply(lambda x: self.map_data(labels, x.name, 'label'), axis=1)
        copy['lower_barriers'] = copy.apply(lambda x: self.map_data(labels, x.name, 'lower_barrier'), axis=1)      
        copy['upper_barriers'] = copy.apply(lambda x: self.map_data(labels, x.name, 'higher_barrier'), axis=1)      
        copy['label'] = copy['label'].map(label_map)
        copy["window_start"] = copy.apply(lambda x: self.map_start_window(labels, x.name), axis=1)      
        copy['signals'] = copy["label"].apply(lambda x: 1 if x == 0 else 2 if x == 2 else 0)
        # Calculate the percentage changes for TP and SL
        copy['tp_stop'] = (copy['upper_barriers'] - copy['close']) / copy['close']
        copy['sl_stop'] = (copy['close'] - copy['lower_barriers']) / copy['close']
        # Replace negative values with 0
        copy['tp_stop'] = copy['tp_stop'].apply(lambda x: max(x, 0))
        copy['sl_stop'] = copy['sl_stop'].apply(lambda x: max(x, 0))
        return copy
    
    def optimize(self, data, num_starts=10, initial_cash=100000, fee=0.006):
        """
        Optimizes trading strategy parameters over specified intervals using the Sharpe ratio as the optimization criterion.

        This method performs parameter optimization by simulating trading performance across multiple intervals of the provided data. For each interval, the optimization process aims to find the best combination of four parameters (`volatility_period`, `upper_barrier_factor`, `lower_barrier_factor`, and `vertical_barrier`) based on the Sharpe ratio of the resulting strategy.

        The process involves the following steps:
        1. Generate interval periods based on the frequency (every 6 months in this case).
        2. For each period, simulate trading with different combinations of parameters.
        3. Optimize the parameters to maximize the Sharpe ratio using a numerical optimization method (Sequential Least Squares Quadratic Programming - SLSQP).
        4. Track the best parameter set for each period and store the corresponding Sharpe ratio and net profit.

        Parameters:
            data (pd.DataFrame): A DataFrame containing historical stock price data, with a 'close' column representing closing prices.
            num_starts (int, optional): The number of random initial guesses to start the optimization. Default is 10.
            initial_cash (float, optional): The initial cash amount for simulating trades. Default is 100,000.
            fee (float, optional): The transaction fee to apply to each trade. Default is 0.006 (0.6%).

        Returns:
            pd.DataFrame: A DataFrame with the optimized parameters for each interval, including:
                - 'start': The start date of the interval.
                - 'end': The end date of the interval.
                - 'params': The optimized parameters for the trading strategy (volatility period, upper and lower barrier factors, vertical barrier).
                - 'sharpe_ratio': The Sharpe ratio of the optimized strategy for the interval.

        Example:
            # Assuming `data` is a DataFrame with historical stock price data:
            optimizer = YourClass()
            optimized_params_df = optimizer.optimize(data)
            # `optimized_params_df` will contain the optimized parameters and Sharpe ratios for each 6-month interval.

        Notes:
            - The function divides the data into 6-month intervals to simulate and optimize the strategy for each period.
            - The `objective_wrapper` function calculates the Sharpe ratio for a given set of parameters and is used in the optimization.
            - The optimization is performed using `scipy.optimize.minimize` with the `SLSQP` method. The parameter bounds are defined by the length of the parameter space.
            - The best parameter set for each interval is stored, and the final DataFrame contains these results.

        """
        optimized_params = []
        # Generate interval periods based on frequency and provided timespan
        intervals = pd.date_range(start=data.index.min(), end=data.index.max(), freq="6M")

        # For each period optimize vol, fu, fl, vt values by simulating returns and optimizing based on Sharpe ratio
        for start, end in zip(intervals[:-1], intervals[1:]):
            best_sharpe_ratio, best_net_profit, best_params = -np.inf, -np.inf, None
            interval_df = data.loc[start:end]

            # Parameter space for optimization (volatility_period, upper_barrier_factor, lower_barrier_factor, vertical_barrier)
            param_space = [
                param_grid['volatility_period'],
                param_grid['upper_barrier_factor'],
                param_grid['lower_barrier_factor'],
                param_grid['vertical_barrier'],
            ]

            def bounds_to_params(x):
                """Converts a list of indices to actual parameter values"""
                return {key: space[int(idx)] for key, space, idx in zip(param_grid.keys(), param_space, x)}

            def objective_wrapper(x):
                """Wrapper for the optimization objective: calculates negative Sharpe ratio"""
                params = bounds_to_params(x)
                transformed = self.transform(interval_df, params['volatility_period'], params['upper_barrier_factor'], params['lower_barrier_factor'], params['vertical_barrier'])
                cr = Calculate_Returns(
                    initial_cash=initial_cash,
                    fee=fee,
                    prices=transformed['close'],
                    signals=transformed['signals'],
                    tp_stop=transformed['tp_stop'],
                    sl_stop=transformed['sl_stop']
                )
                cr.from_signals()
                return -cr.sharpe()  # Minimize the negative Sharpe ratio to maximize it

            # Define the bounds for each parameter based on the parameter space
            bounds = [(0, len(space) - 1) for space in param_space]

            for _ in tqdm(range(num_starts), desc="Optimizing"):
                # Random initial guess
                initial_guess = [np.random.randint(len(space)) for space in param_space]

                # Run the optimization using the SLSQP method
                result = minimize(objective_wrapper, initial_guess, method='SLSQP', bounds=bounds)

                # Check if the result is successful and if it gives a better Sharpe ratio
                if result.success and -result.fun > best_sharpe_ratio:
                    best_sharpe_ratio = -result.fun
                    best_params = bounds_to_params(result.x)

            # Store the results for this period
            optimized_params.append({
                'start': start,
                'end': end,
                'params': best_params,
                'sharpe_ratio': best_sharpe_ratio,
            })

        # Return the results as a DataFrame
        return pd.DataFrame(optimized_params)

