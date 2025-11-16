#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from Data_Preprocessing import Data_Preprocessing
from Triple_Barrier_Labelel import Triple_Barrier_Labelel, LABEL_BULLISH, LABEL_BEARISH, LABEL_NEUTRAL

import pandas_ta as ta
import warnings
from scipy.optimize import minimize
import math 

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

dp = Data_Preprocessing()
encoder = OneHotEncoder(sparse=False)
tbl = Triple_Barrier_Labelel()

def id_to_label(x):
    return LABEL_BULLISH if x == 2 else LABEL_NEUTRAL if x == 1 else LABEL_BEARISH

STARTING_DATE = "2014-01-01"
ENDING_DATE ="2021-06-01"

# In[119]:


def quantile_labels(series: pd.Series, lower, upper):
    labels = pd.Series(index=series.index, dtype='object')
    # Convert scalar thresholds to series if needed
    if np.isscalar(lower):
        lower = pd.Series(lower, index=series.index)
    if np.isscalar(upper):
        upper = pd.Series(upper, index=series.index)
    labels[series < lower] = LABEL_BULLISH
    labels[series > upper] = LABEL_BEARISH
    labels[(series >= lower) & (series <= upper)] = LABEL_NEUTRAL
    return labels

def label_by_zscore(series: pd.Series, window: int = 90, std_coef=(-1.0, 0.5)):
    roll_mean = series.rolling(window=window, min_periods=1).mean()
    roll_std = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    lower = roll_mean + std_coef[0] * roll_std
    upper = roll_mean + std_coef[1] * roll_std
    z = (series - roll_mean) / roll_std
    labels = quantile_labels(z, lower, upper)
    return z, labels

def label_sma_crossover(series: pd.Series, short=10, long=50, z_low=-0.2, z_high=0.2):
    s_short = series.rolling(short).mean()
    s_long = series.rolling(long).mean()
    sma_diff = (s_short - s_long) / s_long  # relative gap
    # Rolling z-score of sma_diff
    roll_mean = sma_diff.rolling(long).mean()
    roll_std = sma_diff.rolling(long).std().replace(0, np.nan)
    z = (sma_diff - roll_mean) / roll_std
    labels = quantile_labels(z, z_low, z_high)
    return s_short, s_long, z, sma_diff, labels

def label_macd(series: pd.Series, fast=12, slow=26, signal=9, q_low=0.33, q_high=0.67):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    # quantile thresholds
    ql, qh = macd_hist.quantile([q_low, q_high])
    labels = quantile_labels(macd_hist, ql, qh)
    return macd, macd_signal, macd_hist, labels

def calc_slopes(series: pd.Series, window):
    slopes = pd.Series(index=series.index, dtype='float64')
    for i in range(window, len(series) + 1):
        window_series = series.iloc[i - window:i].dropna()
        if len(window_series) < window // 2:
            slopes.iloc[i - 1] = np.nan
            continue
        x = np.arange(len(window_series)).reshape(-1, 1)
        y = np.log(window_series.values).reshape(-1, 1)
        lr = LinearRegression().fit(x, y)
        slopes.iloc[i - 1] = float(lr.coef_[0])
    return slopes

def label_trend_slope(series: pd.Series, method=None, window=10, q_low=0.33, q_high=0.67):
    if method == 'pct':
        momentum = series.pct_change(periods=window)
    else:
        momentum = calc_slopes(series, window)
    ql, qh = momentum.quantile([q_low, q_high])
    labels = quantile_labels(momentum, ql, qh)
    return momentum, labels


# In[120]:

def create_tweet_dataset(dataset="Datasets/btc_classified_sentiments"): # investing_classified_sentiments
    tweets = pd.read_csv(dataset)
    tweets = tweets.set_index("index")
    tweets.index = pd.to_datetime(tweets.index)
    tweets_df = tweets.drop('Unnamed: 0', axis=1)
    tweets_df["sen_compound"] = (tweets_df['bullish_mean'] - tweets_df['bearish_mean'])
    tweets_df["sen_compound_increasing"] = (tweets_df["sen_compound"].shift(1) < tweets_df["sen_compound"]).astype(int)
    tweets_df["sen_positive_increasing"] = (tweets_df['bullish_mean'].shift(1) < tweets_df['bullish_mean']).astype(int)
    tweets_df["sen_negative_increasing"] = (tweets_df['bearish_mean'].shift(1) < tweets_df['bearish_mean']).astype(int)
    return tweets_df.dropna()


# In[ ]:

param_grid_crossover = {
    'Low_Window': list(range(5,30)),
    'High_Window': list(range(31,55))
}

def crossover_signal_gen(df, params):
    df["SMA_Short"] = df["close"].rolling(window=params['Low_Window']).mean()
    df["SMA_Long"] = df["close"].rolling(window=params['High_Window']).mean()
    df["signals"] = np.where(df["SMA_Short"] > df["SMA_Long"], 1, -1)
    return df

params_macd = {
    'High_Window': list(range(1, 20))
}

def macd_signal_gen(df, params):
    df["MACD_Val"] = ta.macd(df["close"], length=params["High_Window"])['MACD_12_26_9']
    df["signals"] = np.where(((df["MACD_Val"].shift(1) < df["MACD_Val"])).astype(int), 1, -1)
    return df

params_grid_momentum = {
    'High_Window': list(range(1, 20))
}

def momentum_signal_gen(df, params):
    df["Momentum_Val"] = df["close"] - df["close"].diff(params["High_Window"])
    df["signals"] = np.where((df["Momentum_Val"].shift(1) < df["Momentum_Val"]).astype(int), 1, -1)
    return df

params_grid_close_increasing = {
    'High_Window': list(range(1, 10))
}

def close_increasing_signal_gen(df, params):
    df["signals"] = np.where(df["close"].shift(params["High_Window"]) < df["close"], 1, -1)
    return df

param_grid_roc = {
    'High_Window': list(range(5,30))
}

def roc_signal_gen(df, params):
    _, df["roc"] = dp.calculate_returns(df.close, params["High_Window"])
    df["signals"] = df["roc"].apply(lambda x: -1 if x == "bearish" else 1 if x == "bullish" else 0)
    return df

param_grid_rsi = {
    'RSI_Window': list(range(5,30)),
    'Low_Window': list(range(55,80)),
    'High_Window': list(range(25,50))
}

def rsi_signal_gen(df, params):
    _, df["rsi"] = dp.calculate_rsi(df["close"], params["RSI_Window"], (params["Low_Window"], params["High_Window"]))
    df["signals"] = df["rsi"].apply(lambda x: -1 if x == "bearish" else 1 if x == "bullish" else 0)
    return df

param_grid_volatility = {
    'High_Window': list(range(5,30))
}

def volatility_signal_gen(df, params):
    _, df['volatility_label'] = tbl.get_daily_volatility(df, params['High_Window'])
    df["signals"] = df["volatility_label"].apply(lambda x: -1 if x == "bearish" else 1 if x == "bullish" else 0)
    return df

def optimize_param(data, param_grid, signal_generator, freq='6M', timedelta_value=0):
    optimized_params = []
    param_space = [param_grid[key] for key in param_grid.keys()]
    bounds = [(0, len(space) - 1) for space in param_space]
    
    intervals = pd.date_range(start=data.index.min() + pd.Timedelta(days=timedelta_value),
                              end=data.index.max(), freq=freq)

    for start, end in zip(intervals[:-1], intervals[1:]):
        interval_df = data.loc[start - pd.Timedelta(days=timedelta_value):end].copy()

        def bounds_to_params(x):
            return {key: space[int(round(idx))] for key, space, idx in zip(param_grid.keys(), param_space, x)}

        def objective_wrapper(x):
            params = bounds_to_params(x)
            df = interval_df
            df["pct_change"] = df["close"].pct_change()
            df = signal_generator(df, params)
            df["returns"] = df["signals"].shift(1) * df["pct_change"]
            df = df.dropna(subset=["returns"])
            if df["returns"].dropna().empty:
                return 1e6
            std = df["returns"].std()
            if std == 0 or np.isnan(std):
                return 1e6
            sharpe = (df["returns"].mean() / std) * math.sqrt(365)
            return -sharpe

        best_result = None
        for _ in range(10):
            initial_guess = [np.random.randint(len(space)) for space in param_space]
            result = minimize(objective_wrapper, initial_guess, method='SLSQP', bounds=bounds)
            if not best_result or result.fun < best_result.fun:
                best_result = result

        best_params = bounds_to_params(best_result.x)
        optimized_params.append({
            "start": start,
            "end": end,
            "params": best_params,
            "sharpe": -best_result.fun,
        })

    return optimized_params

def calculate_all_periods(merged_df, optimized_params, signal_generator):
    data = pd.DataFrame()
    for _, row in optimized_params.iterrows():
        start, end, params = row['start'], row['end'], row['params']
        if 'High_Window' in params:
            interval_df = merged_df.loc[start - pd.Timedelta(days=params['High_Window']):end].copy()
        else:
            interval_df = merged_df.loc[start:end].copy()
        interval_df = signal_generator(interval_df, params)
        interval_df = interval_df.loc[start:end]
        data = pd.concat([data, interval_df["signals"]])
    return data

def insert_optimized_parameter(price_df, param_grid, signal_generator, name):
    optimized_params = pd.DataFrame(optimize_param(price_df, param_grid, signal_generator, timedelta_value=np.array(param_grid['High_Window']).max()))
    calculated_periods = calculate_all_periods(price_df, optimized_params, signal_generator)
    calculated_periods = calculated_periods[~calculated_periods.index.duplicated(keep='first')].rename(columns={0: name})
    price_df = pd.concat([price_df, calculated_periods], axis=1)
    return price_df

def create_optimized_dataset(market = 'BTC-USD', starting_date=STARTING_DATE, ending_date=ENDING_DATE):
    price_df = dp.load_stock_data(market, starting_date, ending_date)
    price_df = insert_optimized_parameter(price_df, param_grid_crossover, crossover_signal_gen, "Crossover")
    price_df = insert_optimized_parameter(price_df, params_macd, macd_signal_gen, "MACD")
    price_df = insert_optimized_parameter(price_df, params_grid_momentum, momentum_signal_gen, "Momentum")
    price_df = insert_optimized_parameter(price_df, params_grid_close_increasing, close_increasing_signal_gen, "Close_Increasing")
    price_df = insert_optimized_parameter(price_df, param_grid_roc, roc_signal_gen, "ROC")
    price_df = insert_optimized_parameter(price_df, param_grid_rsi, rsi_signal_gen, "RSI")
    price_df = insert_optimized_parameter(price_df, param_grid_volatility, volatility_signal_gen, "Volatility")
    return price_df.dropna()

def create_continous_dataset(market = 'BTC-USD', starting_date=STARTING_DATE, ending_date=ENDING_DATE):
    price_df = dp.load_stock_data(market, starting_date, ending_date)
    price_df["close_pct_change"] = price_df["close"].pct_change()
    price_df["trade_profitable"] = (price_df["close"].shift(-1) > price_df["close"]).astype(int)
    price_df["close_increasing"] = (price_df["close"].shift(1) < price_df["close"]).astype(int)
    price_df["SMA_50"] = price_df["close"].rolling(window=50).mean()
    price_df["SMA_10"] = price_df["close"].rolling(window=10).mean()
    price_df["crossover"] = (price_df["SMA_10"] > price_df["SMA_50"]).astype(int)
    # price_df["RSI_Basic"] = ta.rsi(price_df["close"], length=50)
    # price_df["RSI_50"] = price_df["RSI_Basic"].rolling(window=50).mean()
    # price_df["RSI_Above_50"] = (price_df["RSI_Basic"] > price_df["RSI_50"]).astype(int)
    # price_df["RSI_Increasing"] = (price_df["RSI_Basic"].shift(1) < price_df["RSI_Basic"]).astype(int)
    price_df["Momentum_Val"] = price_df["close"] - price_df["close"].diff(5)
    price_df["Momentum_Increasing"] = (price_df["Momentum_Val"] > price_df["Momentum_Val"]).astype(int)
    # price_df["ROC_Basic"] = ((price_df["close"] - price_df["close"].shift(5)) / price_df["close"].shift(5)) * 100
    # price_df["ROC_Increasing"] = (price_df["ROC_Basic"].shift(1) < price_df["ROC_Basic"]).astype(int)
    price_df["MACD_Val"] = ta.macd(price_df["close"], length=14)['MACD_12_26_9']
    # price_df["MACD_50"] = price_df["MACD_Val"].rolling(window=50).mean()
    # price_df["MACD_10"] = price_df["MACD_Val"].rolling(window=10).mean()
    # price_df["MACD_Above_50"] = (price_df["MACD_Val"] > price_df["MACD_50"]).astype(int)
    # price_df["MACD_Above_10"] = (price_df["MACD_Val"] > price_df["MACD_10"]).astype(int)
    price_df["MACD_Increasing"] = (price_df["MACD_Val"].shift(1) < price_df["MACD_Val"]).astype(int)
    return price_df.dropna()

def create_triple_barrier_labeling(price_df, length = 8):
    labeled_df = dp.add_optimized_labels(price_df)
    labeled_df = dp.add_target_label(labeled_df)
    # labeled_df = dp.add_technical_indicators(price_df).dropna()
    labeled_df["RSI_Value"], labeled_df["RSI"] = dp.calculate_rsi(labeled_df.close, length, (30, 70))
    labeled_df["ROC_Value"], labeled_df["ROC"] = dp.calculate_returns(labeled_df.close, length)
    
    # labeled_df['previous_label'] = labeled_df['previous_label'].map(id_to_label)
    # labeled_df['next_day_label'] = labeled_df['next_day_label'].map(id_to_label)  
    # z1, labeled_df['volatility_label'] = label_by_zscore(labeled_df['volatility'], 14)
    # s_short, s_long, z, sma_diff,  labeled_df['sma_label'] = label_sma_crossover(labeled_df['close'])
    # m1, labeled_df['momentum_label'] = label_trend_slope(labeled_df['close'], method='pct')
    # m2, labeled_df['close_label'] = label_trend_slope(labeled_df['close'])
    # m3, labeled_df['volume_label'] = label_trend_slope(labeled_df['volume'])
    # m4, labeled_df['tp_stop_label'] = label_by_zscore(labeled_df['tp_stop'], 14)
    # m5, labeled_df['sl_stop_label'] = label_by_zscore(labeled_df['sl_stop'], 14)
    # macd, macd_signal, macd_hist, labeled_df['macd_label'] = label_macd(labeled_df['close'])
    return labeled_df.dropna()

# In[122]:

def extend_time_columns(df, skip_cols=["signals", "close_pct_change", "previous_label", "next_day_label", "close_label", "trade_profitable", "close_increasing"], t = 7):
    for col in df.columns:
        if col in skip_cols:
            continue
        df[f"{col}_t-{t}"] = df[col].shift(t)
    return df.dropna()

# In[ ]:

def create_categorize_dataset(df, suffix_vals=["0", "1", "Bullish", "Bearish", "Neutral"], skip_cols=[]):
    df = df.copy()
    for col in df.columns:
        if (col not in skip_cols) and (df[col].nunique() <= 3):
            one_hot_encoded_col = pd.DataFrame(
                encoder.fit_transform(df[[col]]),
                columns=encoder.get_feature_names_out([col]),
                index=df.index
            )
            
            # Convert values to integers
            one_hot_encoded_col = one_hot_encoded_col.astype(int)
            
            # Rename columns to remove ".0" if present
            one_hot_encoded_col.columns = [c.replace('.0', '') for c in one_hot_encoded_col.columns]
            
            # Keep only desired suffixes
            suffix_pattern = '|'.join([fr'_{v}$' for v in suffix_vals])
            one_hot_encoded_col = one_hot_encoded_col.filter(regex=suffix_pattern)
            
            df = pd.concat([df, one_hot_encoded_col], axis=1)
            df = df.drop(col, axis=1)
        else:
            df = df.drop(col, axis=1)
    return df

# In[124]:


def perform_apriori(dataset, min_support = 0.1, min_confidence = 0.5):
    dataset = dataset.dropna()
    frequent_itemsets = apriori(dataset, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules


# In[125]:


def filter_apriori(rules, antecedents=None, min_lift=None, max_lift=None, target_label = 'close', min_antecedents=5, max_antecedents=20, selected_features=['antecedents', 'support', 'confidence', 'lift']):
    # Normalize target_label to always be a list
    if not isinstance(target_label, (list, tuple, set)):
        target_label = [target_label]
    
    results = []

    for label in target_label:
        condition = (
            (rules['antecedents'].map(len) >= min_antecedents) &
            (rules['antecedents'].map(len) < max_antecedents) &
            (rules['consequents'] == {label})
        )

        if min_lift is not None:
            condition &= (rules['lift'] >= min_lift)
            
        if max_lift is not None:
            condition &= (rules['lift'] <= max_lift)

        if antecedents is not None:
            condition &= (rules['antecedents'] == {antecedents})

        filtered = rules[condition].sort_values(['lift', 'support'], ascending=False)

        if not filtered.empty:
            filtered = filtered[selected_features].copy()
            filtered['target_label'] = label  # Optional: track which label it came from
            results.append(filtered)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=selected_features + ['target_label'])


# In[126]:


def plot(data, title='', x_label='', y_label='', color='skyblue'):
    # Example data
    # Names (x-axis) and values (y-axis)
    names = list(data.keys())
    values = list(data.values())

    # Create bar chart
    plt.bar(names, values, color=color)
    bars = plt.bar(names, values, color=color)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')  # va='bottom' places text above bar

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=60, ha='right', )

    # Show the plot
    plt.show()


# In[127]:


def normalize_values(dataset):
    X = dataset.copy()
    X = X.fillna(method='ffill').fillna(method='bfill')
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    return X_scaled


# In[128]:


def perform_PCA(X_scaled, y, n_components=5, top_k=3, corr_threshold=0.1):
    # --- 1. Compute Pearson correlation with target
    corr_with_target = X_scaled.corrwith(pd.Series(y), method='pearson')
    filtered_features = corr_with_target[abs(corr_with_target) >= corr_threshold].index.tolist()
    print(f"Features kept after correlation filter ({len(filtered_features)}): {filtered_features}")
    if len(filtered_features) < 2:
        print("⚠️ Too few features after correlation filter — skipping PCA.")
        return []
    X_filtered = X_scaled[filtered_features]
    pca = PCA(n_components=min(n_components, len(filtered_features)))
    pca.fit(X_filtered)
    explained = pca.explained_variance_ratio_.cumsum()
    print("Explained variance (cumulative):", explained.round(3))
    loadings = pd.DataFrame(pca.components_.T, 
                            index=filtered_features,
                            columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    abs_loadings = loadings.abs().max(axis=1).sort_values(ascending=False)
    pca_selected_features = list(abs_loadings.head(top_k).index)
    results_df = pd.DataFrame({
        'Correlation_with_target': corr_with_target[pca_selected_features],
        'Max_abs_loading': abs_loadings[pca_selected_features]
    }).sort_values('Max_abs_loading', ascending=False)

    print("\nPCA + Correlation selected features:")
    print(results_df)

# In[129]:


def perform_LASSO(X_scaled, df, target='close', n_splits = 5, n =30):
    # Example target: next-day return (continuous)
    y = df[target].pct_change().fillna(0)#.shift(-1).fillna(0)  # shift so X -> y_next

    # Align X_scaled and y
    X_lasso = X_scaled.loc[y.index].copy()
    y_lasso = y.loc[X_lasso.index]

    # TimeSeries CV
    tscv = TimeSeriesSplit(n_splits)
    lasso = LassoCV(cv=tscv, n_alphas = n, max_iter = 5000).fit(X_lasso, y_lasso)

    # coefficients
    coef = pd.Series(lasso.coef_, index=X_lasso.columns)
    lasso_selected = list(coef[coef != 0].index)
    print("LASSO selected features:", lasso_selected)


# In[ ]:


def generate_plot_data(dataset_for_apriori, target_label, all_scores):
    lift_results = {}
    for col in dataset_for_apriori.columns[3:]:
        el = filter_apriori(all_scores, min_lift = 1.03, antecedents=col, min_antecedents=1, target_label=target_label)
        if len(el) > 0:
            lift = el["lift"].iloc[0]
            lift_results[f"{col}"] = lift
    return dict(sorted(lift_results.items(), key=lambda item: item[1], reverse=True)[:7])


# In[ ]:

def test_apriori_on_labels():
    continous_df = create_continous_dataset()
    columns = ['close_increasing', 'close_label', 'trade_profitable']  #["close_label", 'RSI', 'ROC', 'volatility_label', 'momentum_label', 'macd_label', 'crossover'] # 'close_increasing', 'trade_profitable', 'SMA_50_Increasing', 'RSI_Above_50', 'crossover', 'RSI_Increasing', 'Momentum_Increasing', 'ROC_Increasing', 'macd_label', 'momentum_label', 'MACD_Increasing', 'volatility_label']
    selected_cols_df = continous_df[columns]
    # extended_df = extend_time_columns(selected_cols_df, t=5, skip_cols=["close_label"])
    cat_df = create_categorize_dataset(selected_cols_df.dropna())
    # selected_cols_tweets_df = tweets_df[['tweets_label', 'compound_label']]
    # extended_tweets_df = extend_time_columns(selected_cols_tweets_df, t=2)
    # cat_tweets_df = create_categorize_dataset(extended_tweets_df.dropna())
    # dataset_for_apriori = pd.concat([cat_df, cat_tweets_df], axis=1).dropna()
    dataset_for_apriori = cat_df.dropna()
    all_scores = perform_apriori(dataset_for_apriori, min_support = 0.1)
    plot(generate_plot_data("close_In", all_scores), "Lift Correlations Apriori with close_label_Bullish", "", "Lift")


# In[ ]:

def test_pca_lasso_on_continuous():
    columns_2 = ["close_pct_change", "close", "RSI_Basic", "RSI_Value", "ROC_Basic", "ROC_Value", "volatility", "SMA_10", "MACD_Val", "Momentum_Val", "crossover"]
    selected_cols_df_2 = continous_df[columns_2]
    extended_df_2 = extend_time_columns(selected_cols_df_2, t=2, skip_cols=['close'])
    selected_cols_tweets_df_2 = tweets_df[['compound_score']]
    extended_tweets_df_2 = extend_time_columns(selected_cols_tweets_df_2, t=2)
    dataset_for_lasso_and_pca = pd.concat([extended_tweets_df_2, extended_df_2], axis=1).dropna()
    X_scaled = normalize_values(dataset_for_lasso_and_pca)
    perform_PCA(X_scaled)
    perform_LASSO(aX_scaled, dataset_for_lasso_and_pca, target="close")


# In[ ]:


def test_pearson():
    from scipy.stats import pearsonr
    for col in ['RSI_Value', 'ROC_Value', 'MACD_Val', 
                'RSI_Value_t-1', 'ROC_Value_t-1', 'MACD_Val_t-1']:
        corr, p = pearsonr(dataset_for_lasso_and_pca[col], dataset_for_lasso_and_pca['close'])
        print(col, corr, p)


# In[ ]:


def test_granger():
        from statsmodels.tsa.stattools import grangercausalitytests
        grangercausalitytests(dataset_for_lasso_and_pca[['close', 'RSI_Value']], maxlag=20)

