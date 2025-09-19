#!/usr/bin/env python
# coding: utf-8
 
# In[ ]:
 
 
from decimal import Decimal, getcontext, ROUND_HALF_UP
import yfinance as yf
import pandas as pd
import numpy as np
import math
 
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_UP  # financial round
 
 
# In[ ]:
 
 
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
        self.save_entry_position(self.prices[i], self.prices.index[i], self.tp_stop[i] if i < len(self.tp_stop) else 0, self.sl_stop[i] if i < len(self.sl_stop) else 0, Decimal(1))
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
        self.save_entry_position(self.prices[i], self.prices.index[i], self.tp_stop[i] if i < len(self.tp_stop) else 0, self.sl_stop[i] if i < len(self.sl_stop) else 0, position_size)
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
 
    def perform_action(self, action, i):
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
 
    # Based on following parameters: initial_cash, fee, prices, signals, tp_stop, sl_stop
    # perform trading simulation and retrieve returns 
    def from_signals(self):
        """
        Simulates a trading strategy based on given signals and updates the position, returns, and cash balance.
 
        The method iterates over the provided price data, using signals to decide whether to hold, short, or go long. 
        It calculates returns based on the price change and updates the cash balance accordingly. It also checks if 
        any stop-loss or take-profit barriers are hit before proceeding with the trade action. The simulation is based 
        on different market contexts (e.g., holding a short or long position).
 
        Parameters:
            None (Relies on class attributes such as `self.prices`, `self.signals`, `self.context`, etc.)
 
        Returns:
            None: The function updates the position and returns for each trading day, modifying the cash balance.
 
        Example:
            # Assuming `self.prices` and `self.signals` are populated with stock price data and trading signals:
            from_signals()
            # The method will process each signal, update the returns, and adjust the cash balance accordingly.
 
        Notes:
            - The method checks for take-profit and stop-loss barriers at each iteration, skipping the current trade 
              if any barriers are hit (`check_tp_sl_barriers(i)`).
            - The `self.context` variable tracks the current position of the agent (neutral, short, or long).
            - The function handles three actions: "Hold" (0), "Short" (1), and "Long" (2).
            - For each action, it checks the current context and determines whether to open, close, or continue a position.
            - The returns for each trade are calculated based on the price changes between consecutive days.
            - The method uses `self.add_return()` to append calculated returns to the list of returns and `self.update_cash()` 
              to adjust the agent's cash balance accordingly.
 
        Action Flow:
            - **Hold (0):** If the position is neutral, it holds the current position (no return). If already short or long, 
              it continues with the existing position.
            - **Short (1):** Opens a new short position or continues a short position. If in a long position, it closes the 
              long position and opens a short.
            - **Long (2):** Opens a new long position or continues a long position. If in a short position, it closes the 
              short position and opens a long.
 
        """
        for i in range(1, len(self.prices)):
            action = self.signals[i]
            if self.check_tp_sl_barriers(i):
                continue
            self.perform_action(action, i)
 
 