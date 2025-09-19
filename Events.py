 
#!/usr/bin/env python
# coding: utf-8
 
# In[176]:
 
 
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
 
 
# In[184]:
 
 
class Events():
    def get_common_events(self):
        """
        The function returns start date + 10 days postpone of significant events impacting stocks last years
        """
        event_list = [
        #         {"date": "11-03-2020", "description": "COVID-19 pandemic declared, lockdowns and market crash"},
        #         {"date": "20-03-2020", "description": "COVID Crash Black-thursday"},
        {"date": "01-01-2022", "description": "Global bear market"},
        {"date": "24-02-2022", "description": "Russian Invasion on Ukraine"},
        {"date": "10-03-2023", "description": "Silicon Valley Bank collapse"},
        {"date": "01-12-2023", "description": "Start rise of Magnificent Seven (tech-dominated rally)"},
        {"date": "29-01-2024", "description": "Evergrande liquidation intensifies China market crisis"},
        {"date": "04-01-2025", "description": "Early Jan China market crash (down Asian equities)"},
        {"date": "02-04-2025", "description": "Trump's tarrifs announced"},
        ]
        dates = [event["date"] for event in event_list]
        new_dates = [(datetime.strptime(d, "%d-%m-%Y") + timedelta(days=10)).strftime("%d-%m-%Y") for d in dates]
        sort = sorted(dates + new_dates, key=lambda date: datetime.strptime(date, "%d-%m-%Y"))
        return [pd.to_datetime(day, format="%d-%m-%Y") for day in sort]
 
    def get_t_events(self, raw_price, volatility, base_threshold):
        """
        :param raw_price: (series) of close prices.
        :param volatility: (series) of volatility values.
        :param base_threshold: (float) base level for the threshold.
        :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.
        """
        print('Applying Symmetric CUSUM filter.')
        t_events, s_pos, s_neg = [], 0, 0
        # log returns
        diff = np.log(raw_price).diff().dropna()
 
        # Get event time stamps for the entire series
        for i in tqdm(diff.index[1:]):
            # Adjust the threshold based on volatility

            if isinstance(volatility.loc[i], pd.Series):
                vol = volatility.loc[i].iloc[0]  # Get the first value if it's a Series
            else:
                vol = volatility.loc[i]

            threshold = base_threshold * vol

            if isinstance(diff.loc[i], pd.Series):
                val = diff.loc[i].iloc[0]  # Get the first value if it's a Series
            else:
                val = diff.loc[i]

            s_pos = max(0.0, float(s_pos + val))
            s_neg = min(0.0, float(s_neg + val))

            if s_neg < -threshold:
                s_neg = 0
                t_events.append(i)
 
            elif s_pos > threshold:
                s_pos = 0
                t_events.append(i)
 
        event_timestamps = pd.DatetimeIndex(t_events)
        return event_timestamps
 
    def plot_close_and_events(self, price_df, event_columns):
        # Create a new figure and set the size
        plt.figure(figsize=(14, 7))
 
        # Plot the close price
        plt.plot(price_df.index, price_df['close'], label='Close Price')
 
        # Define a list of colors for the events
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
 
        # Plot each event column with a different color
        for i, event_column in enumerate(event_columns):
            event_indices = price_df[price_df[event_column] == True].index
            event_values = price_df.loc[event_indices, 'close']
            plt.scatter(event_indices, event_values, color=colors[i % len(colors)], label=event_column)
 
        # Set the title and labels
        plt.title('Close Price and Events')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        # Add a legend
        plt.legend()
        # Show the plot
        plt.show()
 
    def add_events(self, labeled_df, base_threshold = 5):
        events = self.get_t_events(labeled_df['close'], labeled_df['volatility'], base_threshold)
        main_events = self.get_common_events()
        labeled_df["common_events"] = labeled_df.index.isin(main_events)
        labeled_df["event"] = labeled_df.index.isin(events)
#         e.plot_close_and_events(transformed, ["event", "common_events"])
        return labeled_df