#!/usr/bin/env python
# coding: utf-8

# In[4]:


from decimal import Decimal, getcontext, ROUND_HALF_UP
import yfinance as yf
import pandas as pd
import numpy as np
import math
from tqdm.auto import tqdm
from scipy.optimize import minimize

getcontext().prec = 10
getcontext().rounding = ROUND_HALF_UP  # financial round


# In[5]:


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
    def __init__(self):
        super()
        
    def calculate_barriers(self, price, volatility, Fu, Fl):
        upper = price + price * volatility * Fu
        lower = price - price * volatility * Fl
        return upper, lower
    
    def assign_labels(self, data, fu, fl, vt):
        barriers = None
        prev_label = None
        labels = []
        for index, day in data.iterrows():
            if barriers == None:
                barriers = self.calculate_barriers(day["close"], day["volatility"], fu, fl)
                start_day = index
            days_last = abs((start_day - index).days)
            if days_last < 3 and abs((start_day - data.index[-1]).days) > 2:
                continue
            if days_last == vt:
                labels.append({"lower_barrier": barriers[1], "higher_barrier": barriers[0], "start": start_day, "end": index, "label": LABEL_NEUTRAL, "prev_label": prev_label})
                barriers = None
                prev_label = LABEL_NEUTRAL
            elif day["high"] > barriers[0]:
                labels.append({"lower_barrier": barriers[1], "higher_barrier": barriers[0], "start": start_day, "end": index, "label": LABEL_BULLISH, "prev_label": prev_label})
                barriers = None
                prev_label = LABEL_BULLISH
            elif day["low"] < barriers[1]:
                labels.append({"lower_barrier": barriers[1], "higher_barrier": barriers[0], "start": start_day, "end": index, "label": LABEL_BEARISH, "prev_label": prev_label})
                barriers = None
                prev_label = LABEL_BEARISH
        return labels
    
    def get_daily_volatility(self, df, span = 14):
        prev_day_start = df.close.index.searchsorted(df.close.index - pd.Timedelta(days=1))
        prev_day_start = prev_day_start[prev_day_start > 0]
        prev_day_start = pd.Series(df.close.index[prev_day_start - 1], index=df.close.index[df.close.shape[0] - prev_day_start.shape[0]:])
        daily_returns = df.close.loc[prev_day_start.index] / df.close.loc[prev_day_start.values].values - 1
        return daily_returns.ewm(span=span).std()
    
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
        copy = df.copy()
        copy["volatility"] = self.get_daily_volatility(copy, vol)
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
    
    def optimize(self, data, num_starts = 10, initial_cash=100000, fee=0.006):
        optimized_params = []
        intervals = pd.date_range(start=data.index.min(), end=data.index.max(), freq="6M")
        for start, end in zip(intervals[:-1], intervals[1:]):
            best_sharpe_ratio, best_net_profit, best_params = -np.inf, -np.inf, None
            interval_df = data.loc[start:end]
            param_space = [
                param_grid['volatility_period'],
                param_grid['upper_barrier_factor'],
                param_grid['lower_barrier_factor'],
                param_grid['vertical_barrier'],
            ]
            
            def bounds_to_params(x):
                return {key: space[int(idx)] for key, space, idx in zip(param_grid.keys(), param_space, x)}
            
            def objective_wrapper(x):
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
                return -cr.sharpe()
            bounds = [(0, len(space) - 1) for space in param_space]
            for _ in tqdm(range(num_starts), desc="Optimizing"):
                initial_guess = [np.random.randint(len(space)) for space in param_space]
                result = minimize(objective_wrapper, initial_guess, method='SLSQP', bounds=bounds)
                if result.success and -result.fun > best_sharpe_ratio:
                    best_sharpe_ratio = -result.fun
                    best_params = bounds_to_params(result.x)
            optimized_params.append({
                'start': start,
                'end': end,
                'params': best_params,
                'sharpe_ratio': best_sharpe_ratio,
            })
        return pd.DataFrame(optimized_params)


# In[6]:


class Calculate_Returns():
    def __init__(self, initial_cash, fee, prices, signals, tp_stop, sl_stop):
        self.cash = Decimal(initial_cash) if initial_cash is not None else Decimal(0)
        self.investing_cash = Decimal(initial_cash) if initial_cash is not None else Decimal(0)
        self.fee = Decimal(fee)
        self.prices = prices.apply(Decimal)
        self.signals = signals
        self.tp_stop = tp_stop.apply(Decimal)
        self.sl_stop = sl_stop.apply(Decimal)
        self.context = 0 # 0-hold, 1-short, 2-long
        self.returns = [Decimal(0)]
        self.records = []
        self.entry_data = {
            "price": None,
            "date": None,
            "tp": None,
            "sl": None,
            "size": Decimal(1)
        }
        
    def add_record(self, direction, exit_price, exit_date):
        if direction != "Long" and direction != "Short":
            print('Error. Direction must be "Long" or "Short"')
            return 0
        entry_value = self.entry_data['size'] * self.entry_data['price']
        exit_value = self.entry_data['size'] * exit_price
        entry_fees = self.fee * self.entry_data['size'] * self.entry_data['price']
        exit_fees = self.fee * self.entry_data['size'] * exit_price
        total_fees = entry_fees + exit_fees
        pnl = exit_value - entry_value - total_fees if direction == "Long" else entry_value - exit_value - total_fees
        self.records.append({
            "Size": self.entry_data['size'],
            "Entry Timestamp": self.entry_data['date'],
            "Avg Entry Price": self.entry_data['price'],
            "Entry fees": entry_fees,
            "Exit Timestamp": exit_date,
            "Avg Exit Price": exit_price,
            "Exit Fees": exit_fees,
            "PnL": pnl,
            "Return": pnl / entry_value if entry_value != 0 else Decimal(0),
            "Direction": direction,
            "Status": "Closed",
        })
        
    def sharpe(self, period = 365):
        returns_array = np.array([float(r) for r in self.returns])
        if returns_array.std() == 0:
            return 0
        return (returns_array.mean() / returns_array.std()) * math.sqrt(period)
        
    def is_position_opened(self):
        return (
            all(self.entry_data[k] is not None for k in self.entry_data if k != "size")
            and self.entry_data["size"] == Decimal(1)
        )
    
    def save_entry_position(self, price, date, tp, sl, size):
        self.entry_data.update({
            "price": price,
            "date": date,
            "tp": tp,
            "sl": sl,
            "size": size
        })
    
    def drop_entry_position(self):
        self.entry_data = {
            key: None if key != "size" else Decimal(1)
            for key in self.entry_data
        }
        self.context = 0

    def get_short_return_value(self, prev_price, curr_price):
        return prev_price - curr_price

    def get_long_return_value(self, prev_price, curr_price):
        return (curr_price - prev_price) * self.entry_data['size']

    def get_fee_value(self, curr_price):
        return -(curr_price * self.fee * self.entry_data['size'])
    
    def update_cash(self, ret, fee):
        self.cash += ret + fee
    
    def add_return(self, ret, fee):
        if self.cash == 0:
            self.returns.append(Decimal(0))
        else:
            self.returns.append((ret + fee)/self.cash)
        
    def open_short(self, i, closing_ret = Decimal(0), closing_fee = Decimal(0)):
        # Abort operation if there is no cash
        # If there is a cash, update context, then save entry position
        # Calculate fee then add previous position returns and fees if exits,
        # Finally combine all costs to retrieve return and update cash
        if self.investing_cash < Decimal(0):
            return None
        self.context = 1 # Change context to short
        self.save_entry_position(self.prices[i], self.prices.index[i], self.tp_stop[i], self.sl_stop[i], Decimal(1))
        fee_opening = self.get_fee_value(self.prices[i])
        self.add_return(fee_opening, closing_ret + closing_fee)
        self.update_cash(fee_opening, closing_ret + closing_fee)
    
    def close_short(self, i):
        # Update investing cash, calculate return and fee then save to logs,
        # Clean open position then return calculated return and fee
        self.investing_cash += self.entry_data['price'] - self.prices[i] - (self.prices[i] * self.fee) - (self.entry_data['price'] * self.fee)
        closing_ret = self.get_short_return_value(self.prices[i - 1], self.prices[i])
        closing_fee = self.get_fee_value(self.prices[i])
        self.add_record("Short", self.prices[i], self.prices.index[i])
        self.drop_entry_position()
        return closing_ret, closing_fee
    
    def open_long(self, i, closing_ret = Decimal(0), closing_fee = Decimal(0)):
        # Retrieve size of asset possible to purchase based on investing cash
        # update context, then save entry position
        # Calculate fee then add previous position returns and fees if exits,
        # Finally combine all costs to retrieve return and update cash
        position_size = Decimal(min(Decimal(1), (self.investing_cash - (self.investing_cash * self.fee)) / self.prices[i]))
        # No cash, so aboart
        if position_size <= Decimal(0):
            return None
        # Purchase 1 asset
        elif position_size == Decimal(1):
            self.investing_cash -= (self.prices[i] - (self.prices[i] * self.fee))
        # Purchase part of asset, more than 0.0 but less than 1.0
        else:
            self.investing_cash = Decimal(0)
        self.context = 2 # Change context to long
        self.save_entry_position(self.prices[i], self.prices.index[i], self.tp_stop[i], self.sl_stop[i], position_size)
        fee_opening = self.get_fee_value(self.prices[i])
        self.add_return(fee_opening, closing_ret + closing_fee)
        self.update_cash(fee_opening, closing_ret + closing_fee)
    
    def close_long(self, i):
        # Update investing cash, calculate return and fee then save to logs,
        # Clean open position then return calculated return and fee
        self.investing_cash += (self.entry_data['size'] * self.prices[i]) - (self.entry_data['size'] * self.prices[i] * self.fee)
        closing_ret = self.get_long_return_value(self.prices[i - 1], self.prices[i])
        closing_fee = self.get_fee_value(self.prices[i])
        self.add_record("Long", self.prices[i], self.prices.index[i])
        self.drop_entry_position()
        return closing_ret, closing_fee
        
    def check_tp_sl_barriers(self, i):
        if self.is_position_opened():
            if self.context == 1:
                # If price hits Take Profit (TP) price or Stop Loss (TL) price, close short position 
                if self.prices[i] <= self.entry_data['price'] * (Decimal(1) - self.entry_data['tp']) or \
                self.prices[i] >= self.entry_data['price'] * (Decimal(1) + self.entry_data['sl']):
                    closing_ret, closing_fee = self.close_short(i)
                    self.add_return(closing_ret, closing_fee)
                    self.update_cash(closing_ret, closing_fee)
                    return True
            else:
                # If price hits Take Profit (TP) price or Stop Loss (TL) price, close long position 
                if self.prices[i] >= self.entry_data['price'] * (Decimal(1) + self.entry_data['tp']) or \
                self.prices[i] <= self.entry_data['price'] * (Decimal(1) - self.entry_data['sl']):
                    closing_ret, closing_fee = self.close_long(i)
                    self.add_return(closing_ret, closing_fee)
                    self.update_cash(closing_ret, closing_fee)
                    return True
        return False
    
    def from_signals(self):
        for i in range(1, len(self.prices)):
            action = self.signals[i]
            if self.check_tp_sl_barriers(i):
                continue
            # --- Action Logic ---
            if action == 0:  # Hold
                if self.context == 0: # Keep holding
                    self.returns.append(Decimal(0))
                elif self.context == 1:  # Hold while short so Continue short
                    ret = self.get_short_return_value(self.prices[i - 1], self.prices[i])
                    self.add_return(ret, Decimal(0))
                    self.update_cash(ret, Decimal(0))
                elif self.context == 2:  # Hold while long so Continue long
                    ret = self.get_long_return_value(self.prices[i - 1], self.prices[i])
                    self.add_return(ret, Decimal(0))
                    self.update_cash(ret, Decimal(0))
            elif action == 1:  # Short
                if self.context == 0:  # Opening short
                    self.open_short(i)
                elif self.context == 1:  # Continue short
                    ret = self.get_short_return_value(self.prices[i - 1], self.prices[i])
                    self.add_return(ret, Decimal(0))
                    self.update_cash(ret, Decimal(0))
                elif self.context == 2:  # Close long, open short
                    closing_ret, closing_fee = self.close_long(i)
                    self.open_short(i, closing_ret, closing_fee)
            elif action == 2:  # Long
                if self.context == 0:  # Opening long
                    self.open_long(i)
                elif self.context == 1:  # Close short, open long
                    closing_ret, closing_fee = self.close_short(i)
                    self.open_long(i, closing_ret, closing_fee)
                elif self.context == 2:  # Continue long
                    ret = self.get_long_return_value(self.prices[i - 1], self.prices[i])
                    self.add_return(ret, Decimal(0))
                    self.update_cash(ret, Decimal(0))
    def test(self):
        data = yf.download("BTC-USD", start="2020-01-01", end="2025-01-01").reset_index()
        data.columns = data.columns.get_level_values(0)
        data = data.rename(columns={"Date": 'date', 'Close': 'close', 'Low': 'low', 'High': 'high', 'Open': 'open'})
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
        tbl = Triple_Barrier_Labelel(data)
        transformed = tbl.transform(data, 15, 1.5, 1.1, 12)
        transformed = pd.DataFrame(transformed)
        cr = Calculate_Returns(
            initial_cash=10000,
            fee=0.005,
            prices=transformed['close'],
            signals=transformed['signals'],
            tp_stop=transformed['tp_stop'],
            sl_stop=transformed['sl_stop']
        )
        cr.from_signals()
        return tbl.optimize(num_starts=10, initial_cash=100000, fee=0.006)

