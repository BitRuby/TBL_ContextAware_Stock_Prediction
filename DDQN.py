#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import random
import time 
import tensorflow as tf
from Data_Preprocessing import Data_Preprocessing
from Calculate_Returns import Calculate_Returns
from Triple_Barrier_Labelel import Triple_Barrier_Labelel
from Model_Train import Model_Train, TextDataset
import torch
from torch.utils.data import DataLoader
from tensorflow import keras
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset as HuggingfaceDataset
import re
import os
import string
import gc

DEBUG = False

dp = Data_Preprocessing()
mt = Model_Train('', '')
encoder = OneHotEncoder(sparse=False)
scaler = MinMaxScaler()
tbl = Triple_Barrier_Labelel()

MEMORY_LENGTH = 100
BATCH_SIZE = 64
MODEL_DESIGN = "64/64"
LEARNING_RATE = 0.0005
FEE = 0.01
INITIAL_CASH = 100000
TARGET_UPDATE = 20
EPISODES = 50
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.95
GAMMA = 0.95
EPSILON = 1.0
# MARKET = "BTC-USD"
STARTING_DATE = "2021-04-01"
ENDING_DATE ="2025-08-01"

# BTC_USD_TWEET_ID = '67082fb60891532d63a3ed67'
TWEETS_RANDOM_SAMPLE = True
TWEETS_ENGAGEMENT_POSTS = False
TWEETS_DAILY_SAMPLE_SIZE = 50
TWEETS_RANDOM_SEED = 42

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[61]:


class Agent():
    def __init__(self, action_size, state_size, gamma, epsilon, epsilon_min, epsilon_decay, model_id = 0):
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.model = self.create_model()
        self.target_model = clone_model(self.model)
        self.optimizer = Adam(learning_rate=LEARNING_RATE) 
        self.step_counter = 0
        
    def generate_model(self, state_size, action_size):
        layers = MODEL_DESIGN.split("/")
        model = Sequential()
        state_size = state_size
        model.add(InputLayer(input_shape=(state_size,), name="InputLayer"))
        for index, units in enumerate(layers):
            model.add(Dense(units=units, activation="relu", name=f"HiddenLayer{index}"))
            model.add(Dropout(0.1))
        model.add(Dense(units=action_size, activation='linear', name="OutputLayer"))
        return model

    def create_model(self):
        return self.generate_model(self.state_size, self.action_size)

    def act(self, state):
        if random.uniform(0,1) < self.epsilon:
            rand_action = random.randrange(self.action_size)
            return rand_action
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict([state], verbose=0)
#         q_values = self.model(np.array(state, dtype=np.float32), training=False).numpy()
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))
        
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def replay(self, step):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, new_states, done = zip(*minibatch)
        states = np.array(states, dtype=np.float32)
        new_states = np.array(new_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
#         states = np.array([sample[0] for sample in minibatch], dtype=np.float32).reshape(BATCH_SIZE, self.state_size)
#         actions = np.array([sample[1] for sample in minibatch], dtype=np.int32)
#         rewards = np.array([sample[2] for sample in minibatch], dtype=np.float32)
#         new_states = np.array([sample[3] for sample in minibatch], dtype=np.float32).reshape(BATCH_SIZE, self.state_size)
#         done = np.array([sample[4] for sample in minibatch])

#         best_action_indices = np.argmax(self.model(new_states, training=False).numpy(), axis=1)
#         target = rewards + (1 - done) * self.gamma * self.target_model(new_states, training=False).numpy()[np.arange(BATCH_SIZE), best_action_indices]
        best_action_indices = np.argmax(self.model.predict([new_states], verbose=0), axis=1)
        target = rewards + (1 - done) * self.gamma * self.target_model.predict([new_states], verbose=0)[np.arange(BATCH_SIZE), best_action_indices]
        with tf.GradientTape() as tape:
            current_Q_values = self.model([states], training=True)
            action_mask = tf.one_hot(actions, current_Q_values.shape[1])
            predicted_Q_values = tf.reduce_sum(current_Q_values * action_mask, axis=1)
            loss = tf.keras.losses.MeanSquaredError()(target, predicted_Q_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        self.step_counter += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.step_counter % TARGET_UPDATE == 0:
            self.step_counter = 0
            self.update_target_model()
        del states, actions, rewards, new_states, done, minibatch
        gc.collect()


# In[62]:


class Environment():
    def __init__(self, data, data_scaled):
        self.action_size = 3
        self.state_size = np.array(data_scaled).shape[1] # + self.action_size + 3 #Context size
        self.data = data
        self.data_scaled = data_scaled
        self.offset = 0
        self.steps = len(data)
        self.returns = Calculate_Returns(INITIAL_CASH, FEE, pd.Series([x[0] for x in data]), pd.Series(dtype="float64"), pd.Series(dtype="float64"), pd.Series(dtype="float64"))
    
    def _append_action_and_position_to_state(self, state, action):
        one_hot_action = np.zeros(self.action_size)
        if action is not None:
            one_hot_action[action] = 1
        position_one_hot = np.zeros(3)
        position_index = int(self.returns.context)
        position_one_hot[position_index] = 1
        return np.concatenate([state, one_hot_action, position_one_hot])
    
    def step(self, action):
        self.offset = self.offset + 1
        new_state = self.data_scaled[self.offset]
#         new_state = self._append_action_and_position_to_state(new_state, action)
        delta = ((self.data[self.offset][0] - self.data[self.offset - 1][0]) / self.data[self.offset - 1][0])
        if action == 0: #Hold
            reward = 0
        elif action == 1: #Short
            reward = -delta
        else: #Long
            reward = delta
#         if action == 0: #Hold
#             if self.returns.context == 0: #Keep holding
#                 reward = -abs(delta)
#             elif self.returns.context == 1: #Keep shorting
#                 reward = -delta
#             elif self.returns.context == 2: #Keep loning
#                 reward = delta
#         elif action == 1: #Short
#             if self.returns.context == 0: #Open short
#                 reward = -delta - FEE
#             elif self.returns.context == 1: #Keep shorting
#                 reward = -delta
#             elif self.returns.context == 2: #Switch to long
#                 reward = delta - 2*FEE
#         elif action == 2: #Long
#             if self.returns.context == 0: #Open long
#                 reward = delta - FEE
#             elif self.returns.context == 1: #Switch to short
#                 reward = - delta - 2*FEE
#             elif self.returns.context == 2: #Keep longing
#                 reward = delta
        self.returns.perform_action(action, self.offset)
#         reward = float(self.returns.returns[-1])
        done = self.offset == len(self.data) - 1
        return new_state, reward, done
    
    def reset(self):
        self.offset = 0
        self.returns = Calculate_Returns(INITIAL_CASH, FEE, pd.Series([x[0] for x in self.data]), pd.Series(dtype="float64"), pd.Series(dtype="float64"), pd.Series(dtype="float64"))
        state_only = self.data_scaled[self.offset]
#         return self._append_action_and_position_to_state(state_only, action=None)
        return state_only


# In[63]:


class DQNAlgorithm():
    def __init__(self, data, data_scaled, episodes, gamma, epsilon, epsilon_min, epsilon_decay):
        self.episodes = episodes
        self.data = data
        self.data_scaled = data_scaled
        self.env = Environment(data, data_scaled)
        self.steps = self.env.steps
        self.agent = Agent(self.env.action_size, self.env.state_size, gamma, epsilon, epsilon_min, epsilon_decay)
        self.best_model_win_rate = 0
        encoder.fit([[0], [1], [2]])
        
    def debug(self, train_data):
        print("Debugging...")
        state = self.env.reset()
        profits = []
        balances = []
        for step in range(self.steps):
            action = self.agent.act(state)
            new_state, reward, done = self.env.step(action)
            print(f"Step: {step}/{self.steps}, Prev Price: {self.data[step][0]}, Previous Context: {self.env.returns.context}, Selected Action: {action}")
            print(f"Next Price: {self.data[step + 1][0]}, Reward: {reward}\n")
            self.agent.remember(state, action, reward, new_state, done)
            profits.append(reward)
            if done == True:
                pos_rewards = (len([p for p in profits if p > 0])/len(profits))*100
                if pos_rewards > self.best_model_win_rate:
                    self.best_model_win_rate = pos_rewards
                    self.best_model = self.agent.model
                print(f"Positive Rewards: {pos_rewards} %")
                if len(self.env.returns.records):
                    print(f"Wins: {(len([e['PnL'] for e in self.env.returns.records if e['PnL'] > 0])/len(self.env.returns.records))*100} %")
                    print(f"Sum PnL: {sum([e['PnL'] for e in self.env.returns.records])}")
                print(f"Sharpe Ratio: {self.env.returns.sharpe()}")
                break
            self.agent.replay(step)
            state = new_state

    def run(self, market):
        print("Training...")
        for episode in range(self.episodes):
            state = self.env.reset()
            profits = []
            balances = []
            actions = 0
            start_time = time.time()
            for step in range(self.steps):
                actions+=1
                action = self.agent.act(state)
                new_state, reward, done = self.env.step(action)
                self.agent.remember(state, action, reward, new_state, done)
                profits.append(reward)
                if done == True:
                    pos_rewards = (len([p for p in profits if p > 0])/len(profits))*100
                    if pos_rewards > self.best_model_win_rate:
                        self.best_model_win_rate = pos_rewards
                        self.best_model = self.agent.model
                    print(f"Episode {episode + 1}/{self.episodes}")
                    print(f"Positive Rewards: {pos_rewards} %")
                    if len(self.env.returns.records):
                        print(f"Wins: {(len([e['PnL'] for e in self.env.returns.records if e['PnL'] > 0])/len(self.env.returns.records))*100} %")
                        print(f"Sum PnL: {sum([e['PnL'] for e in self.env.returns.records])}")
                    days = 365 if market == "BTC" else 252
                    print(f"Sharpe Ratio: {self.env.returns.sharpe(days)}")
                    end_time = time.time()
                    print(f"Elapsed time: {end_time - start_time}\n")
                    break
                self.agent.replay(step)
                state = new_state
        print("Cleaning memory...")
        gc.collect()
        tf.keras.backend.clear_session()
        return self.best_model


# In[64]:


def process_dataset_with_sentiment(labeled_df, tweets_id, market):
    tweet_df = dp.load_tweet_data(tag_id = tweets_id, random_sample = TWEETS_RANDOM_SAMPLE, engagement_posts = TWEETS_ENGAGEMENT_POSTS, daily_sample_size = TWEETS_DAILY_SAMPLE_SIZE, random_seed = TWEETS_RANDOM_SEED)
    merged_df = dp.merge_data(labeled_df, tweet_df)
    merged_df["content"] = merged_df["content"].apply(lambda t: dp.apply_transformations(t, transformations=["TEXT_TO_LOWER", "REMOVE_URLS", "REMOVE_USERNAMES", "REMOVE_PUNCTUATION_WITH_EXCLUDE", "REMOVE_EMOTICONS", f"REPLACE_WITH_{market}"]))
    merged_df["text"] = dp.generate_tweet_prompts(merged_df)
    label_map = {
        0: "Bearish",
        1: "Neutral",
        2: "Bullish"
    }       
    model = AutoModelForSequenceClassification.from_pretrained(
        "btcusd_model",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

    mapped_df = pd.DataFrame(columns=['prediction'])#, 'confidence'])  
    for index, group in merged_df.groupby(merged_df.index):
        preds = mt.predict(model, tokenizer, list(group["text"]))
        counter = Counter(preds.tolist())
        most_common, count = counter.most_common(1)[0]
        confidence = count / len(preds)
        mapped_df.loc[index] = {
            "prediction": most_common,
            "confidence": round(confidence, 2),
        }
    mapped_df['prediction'] = mapped_df['prediction'].map(label_map)
    mapped_df['prediction_t-1'] = mapped_df['prediction'].shift(1)
    mapped_df['prediction_t-2'] = mapped_df['prediction'].shift(2)
    mapped_df['prediction_t-3'] = mapped_df['prediction'].shift(3)
    
    mapped_df = mapped_df.dropna()
    
    # One-hot encode 'Prediction'
    one_hot_prediction = pd.DataFrame(
        encoder.fit_transform(mapped_df[['prediction']]),
        columns=encoder.get_feature_names_out(['prediction']),
        index=mapped_df.index
    )
    # One-hot encode 'Prediction'
    one_hot_prediction_1 = pd.DataFrame(
        encoder.fit_transform(mapped_df[['prediction_t-1']]),
        columns=encoder.get_feature_names_out(['prediction_t-1']),
        index=mapped_df.index
    )
    # One-hot encode 'Prediction'
    one_hot_prediction_2 = pd.DataFrame(
        encoder.fit_transform(mapped_df[['prediction_t-2']]),
        columns=encoder.get_feature_names_out(['prediction_t-2']),
        index=mapped_df.index
    )
        # One-hot encode 'Prediction'
    one_hot_prediction_3 = pd.DataFrame(
        encoder.fit_transform(mapped_df[['prediction_t-3']]),
        columns=encoder.get_feature_names_out(['prediction_t-3']),
        index=mapped_df.index
    )
    # Concatenate one-hot columns
    merged_data = pd.concat([labeled_df, mapped_df, one_hot_prediction, one_hot_prediction_1, one_hot_prediction_2, one_hot_prediction_3], axis=1)
    # Drop the original categorical columns (optional)
    dataset = merged_data.drop(columns=['prediction', 'prediction_t-1', 'prediction_t-2', 'prediction_t-3', 'ROC', 'RSI', 'previous_label'])
    dataset = dataset.dropna()
    return dataset


# In[65]:


def evaluate_model(original_data, scaled_data, mod, log = False, market = "BTC"):
    print("Evaluating...")
    returns = Calculate_Returns(INITIAL_CASH, FEE, pd.Series([x[0] for x in original_data]), pd.Series(dtype="float64"), pd.Series(dtype="float64"), pd.Series(dtype="float64"))
    profits = []
    action = None
    
    def _append_action_and_position_to_state(state, action):
        one_hot_action = np.zeros(3)
        if action is not None:
            one_hot_action[action] = 1
        position_one_hot = np.zeros(3)
        position_index = int(returns.context)
        position_one_hot[position_index] = 1
        return np.concatenate([state, one_hot_action, position_one_hot])

    for step, element in enumerate(original_data[:-1]):
        offset = step
#         state = _append_action_and_position_to_state(scaled_data[offset], action)
#         q_values = mod.predict(np.array(state).reshape(1, -1), verbose=0)
        q_values = mod.predict(np.array(scaled_data[offset]).reshape(1,-1), verbose=0)
        action = np.argmax(q_values[0])
        offset += 1
        if log:
            print(f"Step: {offset}/{len(original_data[:-1])}, Prev Price: {original_data[offset - 1][0]}, Selected Action: {action}")
#         delta = (original_data[offset][0] - original_data[offset - 1][0]) / original_data[offset - 1][0]
#         if action == 0: #Hold
#             if returns.context == 0: #Keep holding
#                 reward = -abs(delta)
#             elif returns.context == 1: #Keep shorting
#                 reward = -delta
#             elif returns.context == 2: #Keep loning
#                 reward = delta
#         elif action == 1: #Short
#             if returns.context == 0: #Open short
#                 reward = -delta - FEE
#             elif returns.context == 1: #Keep shorting
#                 reward = -delta
#             elif returns.context == 2: #Switch to long
#                 reward = delta - 2*FEE
#         elif action == 2: #Long
#             if returns.context == 0: #Open long
#                 reward = delta - FEE
#             elif returns.context == 1: #Switch to short
#                 reward = - delta - 2*FEE
#             elif returns.context == 2: #Keep longing
#                 reward = delta
#         reward = float(returns.returns[-1])
        delta = ((original_data[offset][0] - original_data[offset - 1][0]) / original_data[offset - 1][0])        
        if action == 0: #Hold
            reward = 0
        elif action == 1: #Short
            reward = -delta
        else: #Long
            reward = delta
        returns.perform_action(action, offset)
        if log:
            print(f"Next Price: {original_data[offset][0]}, Reward: {reward}\n")
        profits.append(reward)
    positive_rewards = (len([p for p in profits if p > 0])/len(profits))*100
    if len(returns.records):
        wins = (len([e['PnL'] for e in returns.records if e['PnL'] > 0])/len(returns.records))*100
        sum_pnl = sum([e['PnL'] for e in returns.records])
        days = 365 if market == "BTC" else 252
        sharpe_ratio = returns.sharpe(days)

    if log and len(returns.records):
        print(f"Positive Rewards {positive_rewards}")
        print(f"Wins {wins}")
        print(f"Sum PnL {sum_pnl}")
        print(f"Sharpe Ratio {sharpe_ratio}")
        
    return {
        "Positive Rewards": positive_rewards,
        "Wins": wins if len(returns.records) else 0,
        "Sum PnL": sum_pnl if len(returns.records) else 0,
        "Sharpe Ratio": sharpe_ratio if len(returns.records) else 0
    }


# In[66]:


def get_Tweets_Id(market):
    if market == "BTC-USD":
        return "67082fb60891532d63a3ed67"
    elif market == "NVDA":
        return "67082fb60891532d63a3ed68"
    elif market == "GOOGL":
        return "67082fb70891532d63a3ed69"
    elif market == "AMZN":
        return "67082fb70891532d63a3ed6a"
    elif market == "AAPL":
        return "67082fb70891532d63a3ed6b"
    elif market == "MSFT":
        return "67082fb70891532d63a3ed6c"
    elif market == "TSLA":
        return "67082fb70891532d63a3ed6d"
    elif market == "SPY":
        return "67082fb70891532d63a3ed6e"
    else:
        return "67082fb70891532d63a3ed6f"
    
def extract_text(s):
    # Match only letters at the start until a non-alphabetic character
    match = re.match(r"[A-Za-z]+", s)
    return match.group(0) if match else ""


# In[ ]:


cases = pd.read_csv("DDQN_Cases_2.csv")
for test_id in range(len(cases)):
    path_name = f"{extract_text(cases['Market'][test_id]).lower()}_processed_data.csv"
    if os.path.exists(path_name):
        merged_data = pd.read_csv(path_name)
    else:
        price_df = dp.load_stock_data(cases["Market"][test_id], STARTING_DATE, ENDING_DATE)
        labeled_df = dp.add_optimized_labels(price_df)
        labeled_df = dp.add_target_label(labeled_df)
        labeled_df = dp.add_technical_indicators(labeled_df).dropna()
        labeled_df['RSI_t-1'] = labeled_df['RSI'].shift(1)
        labeled_df['RSI_Value_t-1'] = labeled_df['RSI_Value'].shift(1)
        labeled_df['RSI_t-2'] = labeled_df['RSI'].shift(2)
        labeled_df['RSI_Value_t-2'] = labeled_df['RSI_Value'].shift(2)
        labeled_df['RSI_t-3'] = labeled_df['RSI'].shift(3)
        labeled_df['RSI_Value_t-3'] = labeled_df['RSI_Value'].shift(3)
        labeled_df['ROC_t-1'] = labeled_df['ROC'].shift(1)
        labeled_df['ROC_Value_t-1'] = labeled_df['ROC_Value'].shift(1)
        labeled_df['ROC_t-2'] = labeled_df['ROC'].shift(2)
        labeled_df['ROC_Value_t-2'] = labeled_df['ROC_Value'].shift(2)
        labeled_df['ROC_t-3'] = labeled_df['ROC'].shift(3)
        labeled_df['ROC_Value_t-3'] = labeled_df['ROC_Value'].shift(3)
        labeled_df['volatility_t-1'] = labeled_df['volatility'].shift(1)
        labeled_df['volatility_t-2'] = labeled_df['volatility'].shift(2)
        labeled_df['volatility_t-3'] = labeled_df['volatility'].shift(3)
        labeled_df = labeled_df.dropna()

        # One-hot encode 'ROC'
        one_hot_roc = pd.DataFrame(
            encoder.fit_transform(labeled_df[['ROC']]),
            columns=encoder.get_feature_names_out(['ROC']),
            index=labeled_df.index
        )
        # One-hot encode 'ROC'
        one_hot_roc_1 = pd.DataFrame(
            encoder.fit_transform(labeled_df[['ROC_t-1']]),
            columns=encoder.get_feature_names_out(['ROC_t-1']),
            index=labeled_df.index
        )
        # One-hot encode 'ROC'
        one_hot_roc_2 = pd.DataFrame(
            encoder.fit_transform(labeled_df[['ROC_t-2']]),
            columns=encoder.get_feature_names_out(['ROC_t-2']),
            index=labeled_df.index
        )
        # One-hot encode 'ROC'
        one_hot_roc_3 = pd.DataFrame(
            encoder.fit_transform(labeled_df[['ROC_t-3']]),
            columns=encoder.get_feature_names_out(['ROC_t-3']),
            index=labeled_df.index
        )
        # One-hot encode 'RSI'
        one_hot_rsi = pd.DataFrame(
            encoder.fit_transform(labeled_df[['RSI']]),
            columns=encoder.get_feature_names_out(['RSI']),
            index=labeled_df.index
        )
        # One-hot encode 'RSI'
        one_hot_rsi_1 = pd.DataFrame(
            encoder.fit_transform(labeled_df[['RSI_t-1']]),
            columns=encoder.get_feature_names_out(['RSI_t-1']),
            index=labeled_df.index
        )
        # One-hot encode 'RSI'
        one_hot_rsi_2 = pd.DataFrame(
            encoder.fit_transform(labeled_df[['RSI_t-2']]),
            columns=encoder.get_feature_names_out(['RSI_t-2']),
            index=labeled_df.index
        )
        # One-hot encode 'RSI'
        one_hot_rsi_3 = pd.DataFrame(
            encoder.fit_transform(labeled_df[['RSI_t-3']]),
            columns=encoder.get_feature_names_out(['RSI_t-3']),
            index=labeled_df.index
        )
        labeled_df = labeled_df.drop(columns=['high', 'low', 'open', 'volume', 'lower_barriers', 'upper_barriers', 'window_start', 'signals', 'tp_stop', 'sl_stop', 'next_day_window_start', 'next_day_label'])
        dataset = labeled_df.drop(columns=['ROC_t-1', 'ROC_t-2', 'ROC_t-3', 'RSI_t-1', 'RSI_t-2', 'RSI_t-3'])
        # Concatenate one-hot columns
        merged_data = pd.concat([dataset, one_hot_roc, one_hot_roc_1, one_hot_roc_2,  one_hot_roc_3, one_hot_rsi, one_hot_rsi_1, one_hot_rsi_2, one_hot_rsi_3], axis=1)
        merged_data = merged_data.dropna()
        merged_data = process_dataset_with_sentiment(merged_data, get_Tweets_Id(cases['Market'][test_id]), extract_text(cases['Market'][test_id]))
        merged_data.to_csv(path_name)

    if "TechnicalIndicators" not in cases["Parameters"][test_id]:
        merged_data = merged_data.drop(columns=['ROC_Value', 'ROC_Value_t-1', 'ROC_Value_t-2', 'ROC_Value_t-3', 'ROC_bullish', 'ROC_bearish', 'ROC_neutral', 'ROC_t-1_bullish', 'ROC_t-1_bearish', 'ROC_t-1_neutral', 'ROC_t-2_bullish', 'ROC_t-2_bearish', 'ROC_t-2_neutral', 'ROC_t-3_bullish', 'ROC_t-3_bearish', 'ROC_t-3_neutral', 'RSI_Value', 'RSI_Value_t-1', 'RSI_Value_t-2', 'RSI_Value_t-3', 'RSI_bullish', 'RSI_bearish', 'RSI_neutral', 'RSI_t-1_bullish', 'RSI_t-1_bearish', 'RSI_t-1_neutral', 'RSI_t-2_bullish', 'RSI_t-2_bearish', 'RSI_t-2_neutral', 'RSI_t-3_bullish', 'RSI_t-3_bearish', 'RSI_t-3_neutral'])
    if "Sentiment" not in cases["Parameters"][test_id]:
        merged_data = merged_data.drop(columns=['prediction_Bearish', 'prediction_Bullish', 'prediction_Neutral', 'prediction_t-1_Bearish', 'prediction_t-1_Bullish', 'prediction_t-1_Neutral', 'prediction_t-2_Bearish', 'prediction_t-2_Bullish', 'prediction_t-2_Neutral', 'prediction_t-3_Bearish', 'prediction_t-3_Bullish', 'prediction_t-3_Neutral'])
    if cases["MarketRegime"][test_id] == "WarOnUkraine":
        merged_data.loc[(pd.to_datetime(merged_data[merged_data.columns[0]]) >= "2022-02-01") & (pd.to_datetime(merged_data[merged_data.columns[0]]) <= "2022-03-01")]
        
    merged_data = merged_data.drop(merged_data.columns[0], axis=1)
    merged_data = merged_data.values.tolist()
    split_index = int(len(merged_data) * 0.8)
    train_data = merged_data[:split_index]
    test_data = merged_data[split_index:]
    train_x = np.array(train_data)[:,1:]
    train_y = np.array(train_data)[:,:1]
    test_x = np.array(test_data)[:,1:]
    test_y = np.array(test_data)[:,:1]
    scaler.fit(train_x)
    train_scaled = scaler.transform(train_x).tolist()
    test_scaled = scaler.transform(test_x).tolist()

    dqn = DQNAlgorithm(data=train_y, data_scaled=train_scaled, episodes=EPISODES, gamma=GAMMA, epsilon=EPSILON, epsilon_min=EPSILON_MIN, epsilon_decay = EPSILON_DECAY)
    if DEBUG == True:
        dqn.debug(train_data)
    else:
        best_model = dqn.run(cases['Market'][test_id])

    model_returns_test = evaluate_model(test_y, test_scaled, best_model, True, cases['Market'][test_id])
    columns = ["TestCaseName", "Market", "MarketRegime", "Parameters", "Positive Rewards", "Wins", "Sum PnL", "Sharpe Ratio"]
    test_conc = [cases["TestCaseName"][test_id], cases["Market"][test_id], cases["MarketRegime"][test_id], cases["Parameters"][test_id], model_returns_test["Positive Rewards"], model_returns_test["Wins"], model_returns_test["Sum PnL"], model_returns_test["Sharpe Ratio"]]
    pd.DataFrame([test_conc], columns=columns).to_csv(f"DDQN_Test_Results_2.csv", mode="a", index=False, header=not os.path.exists("DDQN_Test_Results_2.csv"))

