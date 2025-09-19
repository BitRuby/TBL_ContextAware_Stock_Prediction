#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
from functools import partial
from Triple_Barrier_Labelel import Triple_Barrier_Labelel
from datasets import Dataset as HuggingfaceDataset
from transformers import (
    AutoTokenizer
)
import re
import string


# In[2]:


class Data_Preprocessing():    
    def load_stock_data(self, stock_name, starting_date, ending_date):
        """
        Downloads historical stock or cryptocurrency data using the `yfinance` library
        and applies basic preprocessing (renaming and indexing).

        Parameters:
            stock_name (str): The ticker symbol of the asset to download.
            starting_date (str): The start date for the time range (format: 'YYYY-MM-DD').
            ending_date (str): The end date for the time range (format: 'YYYY-MM-DD').

        Returns:
            pd.DataFrame: Preprocessed price data with columns ['open', 'high', 'low', 'close', 'volume']
                          and the index set to datetime timestamps.

        Notes:
            - Column names are standardized to lowercase financial terms.
            - The 'timestamp' column is used as the datetime index.
            - Only daily historical OHLCV data is returned.
        """
        price_df = yf.download(stock_name, start=starting_date, end=ending_date).reset_index()
        price_df.columns = price_df.columns.get_level_values(0)
        price_df = price_df.rename(columns={
            'Date': 'timestamp',
            'Close': 'close',
            'Low': 'low',
            'High': 'high',
            'Open': 'open',
            'Volume': 'volume'
        })
        price_df = price_df.set_index("timestamp")
        price_df.index = pd.to_datetime(price_df.index)
        return price_df
    
    def add_optimized_labels(self, price_df):
        """
        Applies the Triple Barrier Labeling (TBL) technique with hyperparameter optimization 
        to generate labels (bearish, bullish, neutral) for the given price data.

        The method:
            - Runs optimization (e.g. grid/random search) on different time intervals.
            - Applies the TBL algorithm to each interval using the best parameters.
            - Concatenates labeled intervals into one labeled DataFrame.
            - Prints class distribution and Sharpe ratios of each optimized window.

        Parameters:
            price_df (pd.DataFrame): A DataFrame containing historical OHLCV data with 
                                     a datetime index and columns like 'close', 'open', etc.

        Returns:
            labeled_df (pd.DataFrame): A DataFrame with the original price data and an additional
                                       'label' column indicating the market condition class:
                                       0 - bearish, 1 - neutral, 2 - bullish (or similar schema).

        Notes:
            - Uses multiple intervals with independent optimization to increase label robustness.
            - The optimization is repeated `num_starts` times per interval for better parameter search.
            - Outputs a class distribution summary and Sharpe ratio for each window.
        """
        tbl = Triple_Barrier_Labelel()
        optimized_params_df = tbl.optimize(price_df, num_starts=5)

        labeled_df = pd.DataFrame()

        for _, row in optimized_params_df.iterrows():
            start, end, params = row['start'], row['end'], row['params']
            interval_df = price_df.loc[start:end]
            transformed = tbl.transform(
                df=interval_df,
                vol=params['volatility_period'],
                fu=params['upper_barrier_factor'],
                fl=params['lower_barrier_factor'],
                vt=params['vertical_barrier']
            )
            labeled_df = pd.concat([labeled_df, transformed])

        print("Label distribution: ")
        print(labeled_df.label.value_counts(), optimized_params_df.sharpe_ratio)

        return labeled_df
    
    def load_tweet_data(self, data_path="phd.tweets.csv", tag_id="67082fb60891532d63a3ed67", engagement_posts: bool = False, random_sample: bool = False, daily_sample_size: int = 100, random_seed: int = 42):
        """
        Loads tweet data from a CSV file, filters tweets by a specific tag ID,
        and prepares the data with appropriate indexing and date grouping.

        Parameters:
            data_path (str): Path to the CSV file containing tweet data. 
                             Must include 'content', 'date', and 'tag_id' columns.
            tag_id (str): The unique tag identifier used to filter relevant tweets 
                          (e.g., those related to a specific cryptocurrency).
            engagement_posts (bool): If True select only posts having > 20 likes.
            random_sample (bool): Whether to sample a fixed number of tweets per day.
            daily_sample_size (int): Max number of tweets to sample per day (if enabled).
            random_seed (int): Seed for reproducibility in sampling.

        Returns:
            pd.DataFrame: A filtered and preprocessed DataFrame of tweets containing:
                          - 'content': the tweet text,
                          - 'day': extracted date (used for grouping),
                          - datetime index normalized to date level.

        Processing steps:
            - Filters rows by given `tag_id`.
            - Sorts tweets chronologically.
            - Renames the 'date' column to 'timestamp'.
            - Converts timestamps from nanoseconds to datetime.
            - Extracts day-level dates into a separate 'day' column.
            - Normalizes and converts index to date-only format for daily aggregation.

        Notes:
            - This function assumes the timestamp in the CSV is in nanoseconds.
            - The resulting DataFrame is sorted by date for easier time-series alignment.
        """
        tweet_df = pd.read_csv(data_path)
        if engagement_posts == True:
            tweet_df = tweet_df[tweet_df["likes"] > 20]
        tweet_df = tweet_df[tweet_df["tag_id"] == tag_id].sort_values(by='date')[["content", "date"]]
        tweet_df = tweet_df.rename(columns={'date': 'timestamp'})
        tweet_df = tweet_df.set_index("timestamp")
        tweet_df.index = pd.to_datetime(tweet_df.index, unit="ns")
        tweet_df["day"] = tweet_df.index
        tweet_df["day"] = tweet_df.day.apply(lambda x: x.date())
        tweet_df.index = tweet_df.index.normalize()
        tweet_df.index = tweet_df.index.date
        if random_sample:
            tweet_df = tweet_df.groupby('day', group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), daily_sample_size), random_state=random_seed)
            ) 
        tweet_df = tweet_df.sort_index()
        return tweet_df
    
    def add_target_label(self, labeled_df):
        """
        Adds a target label for prediction by shifting the current label to the next day.

        This function prepares the supervised learning target variable (`next_day_label`) 
        by using the current day's label (renamed to `previous_label`) and shifting it by one day.

        Parameters:
            labeled_df (pd.DataFrame): A DataFrame containing at least a 'label' column 
                                       and 'window_start' column, typically the output of 
                                       the Triple Barrier Labeling (TBL) process.

        Returns:
            pd.DataFrame: The modified DataFrame with:
                - 'previous_label': the original label for the current day.
                - 'next_day_label': the label of the following day (used as the prediction target).
                - 'next_day_window_start': the start flag shifted by one day to align with the target label.

        Notes:
            - The first row's 'next_day_window_start' is set manually to `True` for consistency.
            - Final row's 'next_day_label' will be NaN due to the shift.
            - Useful in models where prediction of next day’s label is based on today's input.
        """
        labeled_df.rename(columns={'label': 'previous_label'}, inplace=True)
        labeled_df["next_day_label"] = labeled_df.previous_label.shift(-1)
        labeled_df["next_day_window_start"] = labeled_df.window_start.shift(-1)
        labeled_df.loc[labeled_df.iloc[0].name, 'next_day_window_start'] = True
        return labeled_df
    
    def calculate_rsi(self, close_series, length, threshold=(30, 70)):
        """
        Calculates the Relative Strength Index (RSI) and returns descriptive labels 
        based on predefined thresholds.

        The RSI is computed using a simplified approach (based on rolling mean of 
        gains/losses rather than Wilder's smoothing). It helps to identify potential 
        overbought or oversold conditions in the price series.

        Parameters:
            close_series (pd.Series): Series of closing prices.
            length (int): Number of periods to use for rolling average calculations.
            threshold (Tuple[int, int], optional): Lower and upper RSI thresholds 
                for labeling trends. Default is (30, 70), where:
                - RSI < 30 → 'bullish'
                - RSI > 70 → 'bearish'
                - Otherwise → 'neutral'

        Returns:
            Tuple[pd.Series, pd.Series]:
                - rsi (pd.Series): The computed RSI values.
                - description (pd.Series): Textual interpretation of RSI values 
                  as 'bullish', 'bearish', or 'neutral'.

        Notes:
            - This implementation uses simple moving averages (SMA) instead of 
              exponential moving averages (EMA), so it approximates but does not 
              replicate Wilder’s original formula.
            - Can be used as a feature or label in financial ML tasks.
        """
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        description = pd.Series(index=close_series.index, dtype='object')
        # Apply dynamic thresholds if provided
        if threshold is not None:
            lower_threshold, upper_threshold = threshold
            description[rsi > upper_threshold] = 'bearish'
            description[rsi < lower_threshold] = 'bullish'
            description[(rsi <= upper_threshold) & (rsi >= lower_threshold)] = 'neutral'

        return rsi, description

    def calculate_returns(self, close_series, length, std_coef = (-0.5, 0.4)):
        """
        Calculates rolling percentage returns and classifies market sentiment 
        using adaptive volatility-aware thresholds.

        This function computes daily percentage returns from closing prices and uses 
        the rolling standard deviation to define dynamic thresholds for labeling:
            - 'bearish' if return < −0.5 × rolling_std (default)
            - 'bullish' if return > +0.4 × rolling_std (default)
            - 'neutral' otherwise

        Parameters:
            close_series (pd.Series): Series of closing prices.
            length (int): Rolling window size (in days) for computing return volatility.
            std_coef (Tuple[float, float], optional): Coefficients for setting dynamic thresholds:
                - First value (negative) sets bearish threshold (default: -0.5).
                - Second value (positive) sets bullish threshold (default: +0.4).

        Returns:
            Tuple[pd.Series, pd.Series]:
                - returns (pd.Series): The computed daily percentage returns.
                - description (pd.Series): Sentiment labels ('bullish', 'bearish', or 'neutral') 
                  based on adaptive thresholds.

        Notes:
            - Adaptive thresholds make the labeling sensitive to recent volatility.
            - Useful for incorporating short-term momentum into trading signals or 
              supervised learning models.
            - The first few entries may be NaN due to rolling window effects.
        """
        returns = close_series.pct_change()
        neg_std_coef, pos_std_coef = std_coef
        rolling_std = returns.rolling(window=length).std()
        rolling_neg_std_effect = rolling_std * neg_std_coef
        rolling_pos_std_effect = rolling_std * pos_std_coef

        # Descriptive sentiment labels based on adaptive thresholds
        description = pd.Series(index=close_series.index, dtype='object')
        description[returns < rolling_neg_std_effect] = 'bearish'
        description[returns > rolling_pos_std_effect] = 'bullish'
        description[(returns <= rolling_pos_std_effect) & (returns >= rolling_neg_std_effect)] = 'neutral'

        return returns, description
    
    def add_technical_indicators(self, labeled_df):
        """
        Adds technical indicators RSI (Relative Strength Index) and ROC (Rate of Change) 
        to the labeled DataFrame using an 8-day window.

        - RSI is computed using a predefined helper function (`calculate_rsi`) and reflects
          recent price strength relative to historical losses/gains.
        - ROC is computed using a wrapper around percentage returns (`calculate_returns`), 
          indicating momentum as a percent change over the window.

        Parameters:
            labeled_df (pd.DataFrame): Input DataFrame that must include a 'close' price column.

        Returns:
            pd.DataFrame: DataFrame with two new columns added:
                - "RSI": Relative Strength Index values (float).
                - "ROC": Rate of Change percentage returns (float).

        Notes:
            - RSI thresholds (e.g. 30/70) are typically used for oversold/overbought signals.
            - Uses an 8-period window for both RSI and ROC by default.
            - The function prints the distribution of RSI labels and ROC values for quick inspection.
            - Any NaNs introduced by rolling computations should be handled downstream.
        """
        labeled_df["RSI_Value"], labeled_df["RSI"] = self.calculate_rsi(labeled_df.close, 8, (30, 70))
        labeled_df["ROC_Value"], labeled_df["ROC"] = self.calculate_returns(labeled_df.close, 8)
        print("ROC and RSI label distribution: ")
        print(labeled_df["RSI"].value_counts())
        labeled_df["ROC"].value_counts()  # This line could be printed or stored if needed
        return labeled_df

    def merge_data(self, labeled_df, tweet_df):
        """
        Merges stock data with tweet data based on their datetime index to align 
        sentiment/text features with market labels and technical indicators.

        Specifically, it performs a left join of tweet data with selected columns from 
        the labeled stock DataFrame, ensuring alignment by date (index-based merge). 
        Rows with missing values are removed after merging to ensure clean training data.

        Parameters:
            labeled_df (pd.DataFrame): DataFrame containing stock labels and technical indicators.
                Required columns: ['next_day_label', 'next_day_window_start', 
                                  'previous_label', 'ROC', 'RSI'].
                Must use datetime as index.

            tweet_df (pd.DataFrame): DataFrame containing tweet-level data 
                (e.g., tokenized text, timestamps). Must also use datetime as index.

        Returns:
            pd.DataFrame: Merged DataFrame containing:
                - Tweet features from `tweet_df`.
                - Market labels and indicators from `labeled_df`.
            The result includes only rows where all selected label/indicator data is present 
            (i.e., no NaNs after join).

        Notes:
            - Performs an index-based left join (`how="left"`).
            - Drops any rows with missing values after the merge.
            - Prints the distribution of the `next_day_label` column for class balance inspection.
            - Essential step for supervised learning using both textual and price data.
        """
        merged_df = tweet_df.merge(
            labeled_df,
            left_index=True,
            right_index=True,
            how="left"
        )
        merged_df.dropna(inplace=True)
        return merged_df
    
    def undersample_tweets(self, merged_df):
        """
        Balances the tweet dataset by applying undersampling to the majority classes based on 
        the `next_day_label` column.

        This function ensures that each class (bullish, bearish, neutral) contains the same 
        number of samples as the minority class, helping to prevent model bias toward 
        overrepresented trends. It randomly samples tweets from the majority classes while 
        keeping all samples from the minority class.

        Parameters:
            merged_df (pd.DataFrame): DataFrame that contains tweet text features and
                the 'next_day_label' column, indicating the market trend label for the next day.

        Returns:
            pd.DataFrame: A balanced version of the input DataFrame where each class in 
            'next_day_label' has equal representation (same number of rows).

        Steps:
            - Count samples per class in 'next_day_label'.
            - Identify the minority class and its sample count.
            - Keep all rows from the minority class.
            - Randomly sample an equal number of rows from the other classes.
            - Concatenate all subsets into a balanced DataFrame.

        Notes:
            - Random undersampling may discard potentially useful data from majority classes.
            - For reproducibility, consider setting a `random_state` in `.sample()`.
            - Use this method when class imbalance causes poor model generalization.
        """
        trend_counts = merged_df['next_day_label'].value_counts()
        minority_class = trend_counts.idxmin()
        minority_count = trend_counts.min()
        undersampled_df = pd.DataFrame()

        for trend in merged_df['next_day_label'].unique():
            if trend == minority_class:
                undersampled_df = pd.concat([undersampled_df, merged_df[merged_df['next_day_label'] == trend]])
            else:
                subset = merged_df[merged_df['next_day_label'] == trend].sample(minority_count)
                undersampled_df = pd.concat([undersampled_df, subset])
        print("Next day label distribution after undersample: ")
        print(undersampled_df.next_day_label.value_counts())
        return undersampled_df
    
    def generate_tweet_prompts(self, df, include_previous_label=True, include_roc=True, include_rsi=True):
        """
        Generates prompts for a language model (LLM) by combining various financial and sentiment features with tweet text.

        This function constructs a textual prompt for each row in the input DataFrame, which can be used 
        for training or generating predictions with an LLM. The prompt combines the previous label, 
        rate of change (ROC), and Relative Strength Index (RSI) along with the tweet content.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing the columns 'previous_label', 'ROC', 
                'RSI', and 'text'. Each row represents a tweet with corresponding financial features.
            include_previous_label (bool, optional): If True, includes the previous label (bullish, neutral, bearish)
                in the prompt. Default is True.
            include_roc (bool, optional): If True, includes the rate of change (ROC) in the prompt. Default is True.
            include_rsi (bool, optional): If True, includes the Relative Strength Index (RSI) in the prompt. Default is True.

        Returns:
            List[str]: A list of prompts, each corresponding to a row in the input DataFrame. 
            Each prompt is a string structured as: "previous label: [label], roc: [roc_value], rsi: [rsi_value], tweet: [tweet_text]".

        Example:
            df = pd.DataFrame({
                'previous_label': [2, 1],
                'ROC': ['bullish', 'bearish'],
                'RSI': ['neutral', 'bullish'],
                'text': ["Bitcoin is on the rise!", "Ethereum is going down."]
            })
            prompts = generate_tweet_prompts(df)
            # Output: ["previous label: bullish roc: bullish rsi: neutral tweet: Bitcoin is on the rise!",
            #          "previous label: neutral roc: bearish rsi: bullish tweet: Ethereum is going down."]

        Notes:
            - The function constructs the prompt in a format that can be directly used for language model training.
            - The column names in the input DataFrame must match the expected labels ('previous_label', 'ROC', 'RSI', 'text').
            - If any of the optional features (previous_label, roc, rsi) are excluded, they will be omitted from the final prompt.
        """
        def id_to_label(x):
            return "bullish" if x == 2 else "neutral" if x == 1 else "bearish"

        prompts = []
        df["text"] = df["content"]
        for _, row in df.iterrows():
            prompt_parts = []
            if include_previous_label:
                prompt_parts.append(f"previous label: {id_to_label(row['previous_label'])}")
            if include_roc:
                prompt_parts.append(f"roc: {row['ROC']}")
            if include_rsi:
                prompt_parts.append(f"rsi: {row['RSI']}")
            prompt_parts.append(f"tweet: {row['text']}")
            prompts.append(" ".join(prompt_parts))

        return prompts
    
    def tokenize(self, tokenizer, dataset):
        """
        Tokenizes the text data in the provided dataset for use in a BERT-style model such as BertForSequenceClassification.

        This function applies tokenization to each item in the dataset, converting the text into input IDs and attention masks 
        that are required by BERT models. The text is tokenized, padded to a maximum length, and truncated if necessary. 
        The output includes the tokenized text as input IDs, an attention mask, and the original label.

        Parameters:
            tokenizer (PreTrainedTokenizer): A HuggingFace tokenizer object used to tokenize the text data.
            dataset (datasets.Dataset): A HuggingFace dataset containing the text data (in the "text" field) and labels 
                (in the "label" field).

        Returns:
            datasets.Dataset: The tokenized version of the dataset with additional fields:
                - "input_ids": Tokenized and padded input IDs for the model.
                - "attention_mask": Attention mask (1 for real tokens, 0 for padding).
                - "label": The original label for each entry.

        Example:
            # Assuming `tokenizer` is a HuggingFace tokenizer and `dataset` is a HuggingFace Dataset containing 'text' and 'label'.
            tokenized_data = tokenize(tokenizer, dataset)
            # The returned `tokenized_data` will contain input_ids, attention_mask, and label for each data point.

        Notes:
            - This function assumes that the input dataset has a 'text' field for the text data and a 'label' field for labels.
            - The tokenized data will be padded or truncated to the specified maximum length (512 tokens by default).
            - The function uses `partial` to simplify the tokenization process within the `map` function, allowing batch processing.
        """
        # Tokenize the text field in the dataset
        def tokenize_function(tokenizer, item):
            # Tokenize the text and return only the necessary fields
            encoded = tokenizer(item["text"], padding="max_length", truncation=True, max_length=512)
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "label": item["label"]}

        # tokenizing the dataset text to be used in train and test loops
        partial_tokenize_function = partial(tokenize_function, tokenizer)
        tokenized_datasets = dataset.map(partial_tokenize_function, batched=True)

        return tokenized_datasets
    
    def apply_transformations(self, text, transformations):
        if "REMOVE_USERNAMES" in transformations:
            text = re.sub(r"@\w+", "", text)
        if "REMOVE_URLS" in transformations:
            text = re.sub(r"(?:https?://|www\.)\S+\.\S+", "", text)
        if "REMOVE_PUNCTUATION_MARKS" in transformations:
            remove_pun_pattern_1 = r"(?<!\d)\.(?!\d)|[^\w\s.']"
            remove_pun_pattern_2 = r"'"
            remove_pun_pattern_3 = r"\s\s+"
            sub1 = re.sub(remove_pun_pattern_1, " ", text)
            sub2 = re.sub(remove_pun_pattern_2, "", sub1)
            text = re.sub(remove_pun_pattern_3, " ", sub2)
        if "REMOVE_PUNCTUATION_WITH_EXCLUDE" in transformations:
            exclude = set(string.punctuation)
            for char in ['!','?','%','$','&']:
                exclude.remove(char)
            return ''.join(ch for ch in text if ch not in exclude)
        if "TEXT_TO_LOWER" in transformations:
            text = text.lower()
        if "REMOVE_N" in transformations:
            text = text.replace("\\n", "").replace("\n", "")
        if "REMOVE_SHORT_WORDS" in transformations:
            remove_short_pattern = r'\b\w{1,2}\b'
            text = re.sub(remove_short_pattern, '', text)
        if "REMOVE_ENDING_HASHTAGS" in transformations:
            pattern = r'([#$@\$]\w+)(?=(\s[#$@\$]\w+)*\s*$)'
            text = re.sub(pattern, '', text)
        if "REMOVE_HASHTAGS" in transformations:
            text = re.findall(r"#(\w+)", text)
        if "REMOVE_HEX" in transformations:
            text = re.sub(r'\b0x[a-fA-F0-9]{6,}\b', '', text)
        if "REMOVE_EMOTICONS" in transformations:
            emoji_pattern = re.compile(
                "[" 
                u"\U0001F600-\U0001F64F"  # Emoticons
                u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # Transport & map
                u"\U0001F1E0-\U0001F1FF"  # Flags
                u"\U00002700-\U000027BF"  # Dingbats
                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs (includes 🤝)
                u"\U00002600-\U000026FF"  # Misc symbols (e.g. ☀️☂️)
                u"\U00002B00-\U00002BFF"  # Arrows etc.
                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                "]+", flags=re.UNICODE
            )
            text = emoji_pattern.sub(r'', text)
        if "REMOVE_SPACES" in transformations:
            text = re.sub(r"\s{2,}", " ", text).strip()
        if "REPLACE_WITH_BTC" in transformations:
            text = re.sub(r"Bitcoin|bitcoin|btc|BitCoin", "BTC", text)
        if "REPLACE_WITH_NVDA" in transformations:
            text = re.sub(r"nVidia|nvidia|nvda|NVDA|Nvidia", "NVIDIA", text)
        if "REPLACE_WITH_GOOGL" in transformations:
            text = re.sub(r"google|googl|GOOGL|Google", "GOOGLE", text)
        if "REPLACE_WITH_AMZN" in transformations:
            text = re.sub(r"AMZN|amzn|amazon|Amazon", "AMAZON", text)
        if "REPLACE_WITH_AAPL" in transformations:
            text = re.sub(r"AAPL|aapl|apple|Apple", "APPLE", text)
        if "REPLACE_WITH_TSLA" in transformations:
            text = re.sub(r"TSLA|tsla|Tesla", "TESLA", text)
        if "REPLACE_WITH_MSFT" in transformations:
            text = re.sub(r"MSFT|msft|MS|Microsoft|M$", "MICROSOFT", text)
        if "REPLACE_WITH_SPX" in transformations:
            text = re.sub(r"SPX|spx|sp500|s&p500|S&p500", "S&P500", text)
        return text

    def transform_dataset(self, dataset, transformations):
        """
        Apply a series of text cleaning and preprocessing transformations to a dataset of text data.

        This function allows for customizable text preprocessing steps that remove or modify unwanted parts 
        of the text, such as URLs, usernames, punctuation marks, emoticons, and more. The specific transformations 
        to be applied are defined by the user and passed as a list of transformation names.

        Parameters:
            dataset (datasets.Dataset): A HuggingFace dataset containing a "text" field, which consists of the raw text 
                to be preprocessed.
            transformations (list): A list of strings, each representing a transformation to be applied to the text. 
                Available transformations include:
                - "REMOVE_USERNAMES": Removes usernames starting with '@'.
                - "REMOVE_URLS": Removes URLs (http/https).
                - "REMOVE_PUNCTUATION_MARKS": Removes unwanted punctuation marks and extra spaces.
                - "REMOVE_PUNCTUATION_WITH_EXCLUDE": Removes punctuation marks except for specified characters (e.g., !, ?, %, $).
                - "TEXT_TO_LOWER": Converts the text to lowercase.
                - "REMOVE_N": Removes newline characters and escape sequences.
                - "REMOVE_SHORT_WORDS": Removes words with 1 or 2 characters.
                - "REMOVE_ENDING_HASHTAGS": Removes hashtags appearing at the end of the text.
                - "REMOVE_HASHTAGS": Removes hashtags from the text.
                - "REMOVE_HEX": Removes hexadecimal values (e.g., 0x123abc).
                - "REMOVE_EMOTICONS": Removes emoticons and emojis.
                - "REMOVE_SPACES": Removes excess whitespace.
                - "REPLACE_WITH_BTC": Replaces variations of "Bitcoin" with "BTC".

        Returns:
            datasets.Dataset: A new dataset with the text field preprocessed according to the specified transformations.

        Example:
            # Assuming `dataset` is a HuggingFace Dataset with 'text' field and `transformations` contains the desired transformations.
            transformed_data = transform_dataset(dataset, ["REMOVE_USERNAMES", "TEXT_TO_LOWER", "REMOVE_URLS"])
            # The `transformed_data` will contain the cleaned text based on the applied transformations.

        Notes:
            - Each transformation is applied in the order specified in the `transformations` list.
            - The function uses regex and pattern matching to perform the transformations on the text data.
            - The batch transformation is handled via the HuggingFace `map` function to efficiently preprocess large datasets.
        """

        def transform_batch(batch):
            return {"text": [self.apply_transformations(text, transformations) for text in batch["text"]]}

        return dataset.map(transform_batch, batched=True)
    
    def preprocess_and_tokenize(self, labeled_ds, transformations=["TEXT_TO_LOWER", "REMOVE_URLS", "REMOVE_USERNAMES", "REMOVE_PUNCTUATION_WITH_EXCLUDE", "REPLACE_WITH_BTC"], tokenizer_name = "ElKulako/cryptobert"):
        """
        Preprocesses and tokenizes a labeled dataset to prepare it for input into a BERT-style model like FinBERT.

        This function applies text transformations (such as text cleaning, removal of URLs, and punctuation), 
        encodes the labels, and tokenizes the text data using a pre-trained tokenizer (e.g., FinBERT tokenizer).

        Parameters:
            labeled_ds (datasets.Dataset): A HuggingFace dataset containing labeled text data with a "text" field for the 
                raw text and a "label" field for the corresponding labels.
            transformations (list, optional): A list of strings representing the transformations to apply to the text data 
                before tokenization. Default is ["TEXT_TO_LOWER", "REMOVE_URLS", "REMOVE_USERNAMES", "REMOVE_PUNCTUATION_WITH_EXCLUDE", "REPLACE_WITH_BTC"].
                Available transformations include:
                - "REMOVE_USERNAMES": Removes usernames starting with '@'.
                - "REMOVE_URLS": Removes URLs (http/https).
                - "REMOVE_PUNCTUATION_MARKS": Removes unwanted punctuation marks and extra spaces.
                - "REMOVE_PUNCTUATION_WITH_EXCLUDE": Removes punctuation marks except for specified characters (e.g., !, ?, %, $).
                - "TEXT_TO_LOWER": Converts the text to lowercase.
                - "REMOVE_N": Removes newline characters and escape sequences.
                - "REMOVE_SHORT_WORDS": Removes words with 1 or 2 characters.
                - "REMOVE_ENDING_HASHTAGS": Removes hashtags appearing at the end of the text.
                - "REMOVE_HASHTAGS": Removes hashtags from the text.
                - "REMOVE_HEX": Removes hexadecimal values (e.g., 0x123abc).
                - "REMOVE_EMOTICONS": Removes emoticons and emojis.
                - "REMOVE_SPACES": Removes excess whitespace.
                - "REPLACE_WITH_BTC": Replaces variations of "Bitcoin" with "BTC".
            tokenizer_name: 
                Select tokenizer from pretrained model "ElKulako/cryptobert" or "ProsusAI/finbert" or other
        Returns:
            datasets.Dataset: A preprocessed and tokenized version of the input dataset, ready for model training. 
                The returned dataset will contain the following fields:
                - "input_ids": The tokenized input IDs for the model.
                - "attention_mask": The attention mask, indicating the positions of actual tokens vs. padding.
                - "label": The original label encoded as integers.

        Example:
            # Assuming `labeled_ds` is a HuggingFace Dataset with 'text' and 'label' fields.
            preprocessed_and_tokenized_data = preprocess_and_tokenize(labeled_ds)
            # The `preprocessed_and_tokenized_data` will contain cleaned and tokenized text data with corresponding labels.

        Notes:
            - The function first applies the transformations in the `transformations` list to the raw text data.
            - The labels in the dataset are encoded using `class_encode_column`.
            - The function uses the `AutoTokenizer.from_pretrained` method to load a pre-trained tokenizer (CryptoBERT in this case).
            - The preprocessing steps are applied in the order specified in the `transformations` list.
        """
        # Apply the preprocessing transformations
        labeled_ds = self.transform_dataset(labeled_ds, transformations)
        # Encode the labels as integers (class labels)
        labeled_ds = labeled_ds.class_encode_column('label')
        # Load the pre-trained tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        # Tokenize the text data
        labeled_ds = self.tokenize(tokenizer, labeled_ds)

        return labeled_ds
    
    def test(self):
        price_df = self.load_stock_data("BTC-USD", "2020-01-01", "2025-01-01")
        labeled_df = self.add_optimized_labels(price_df)
        labeled_df = self.add_target_label(labeled_df)
        labeled_df = self.add_technical_indicators(labeled_df)
        tweet_df = self.load_tweet_data()
        merged_df = self.merge_data(labeled_df, tweet_df)
        balanced_df = self.undersample_tweets(merged_df)
        balanced_df["text"] = self.generate_tweet_prompts(balanced_df)
        balanced_df = balanced_df.dropna()
        balanced_df = balanced_df.sort_index()
        balanced_df["label"] = balanced_df.next_day_label
        labeled_ds = HuggingfaceDataset.from_pandas(balanced_df[["text", "label"]])
        return self.preprocess_and_tokenize(labeled_ds, tokenizer_name = "ElKulako/cryptobert")

