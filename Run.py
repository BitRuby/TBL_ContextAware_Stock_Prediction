#!/usr/bin/env python
# coding: utf-8
 
# In[1]:
 
 
from Model_Train import Model_Train
from Data_Preprocessing import Data_Preprocessing
from Events import Events
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datasets import Dataset as HuggingfaceDataset
import pandas as pd
 
 
# In[2]:
 
 
uri = "mongodb+srv://michalzwierzynski:qbGN9Wf022lslPny@nlp.5exh7.mongodb.net/?retryWrites=true&w=majority&appName=nlpsentanalyze"
client = MongoClient(uri, server_api=ServerApi('1'))
mydb = client["phd"]
paper_results_db = mydb["paper_results"]
 
 
# In[ ]:
 
 
cases = pd.read_csv("TBL_Test_Cases.csv")
ev = Events()
for test_id in range(len(cases)):
    dp = Data_Preprocessing()
    price_df = dp.load_stock_data(cases['Market'][test_id], "2021-05-01", "2025-05-01")
    labeled_df = dp.add_optimized_labels(price_df)
    labeled_df = dp.add_target_label(labeled_df)
    labeled_df = dp.add_technical_indicators(labeled_df)
 
    # labeled_df = ev.add_events(labeled_df)
    # labeled_df = labeled_df[labeled_df["event"] == True]
 
    tweet_df = dp.load_tweet_data(tag_id = cases['TweetsId'][test_id], random_sample = True, engagement_posts = False, daily_sample_size = 50, random_seed = 42)
    merged_df = dp.merge_data(labeled_df, tweet_df)
    balanced_df = dp.undersample_label_data(merged_df)
    balanced_df["text"] = dp.generate_tweet_prompts(balanced_df)
    balanced_df = balanced_df.dropna()
    balanced_df = balanced_df.sort_index()
    balanced_df["label"] = balanced_df.next_day_label
    labeled_ds = HuggingfaceDataset.from_pandas(balanced_df[["text", "label"]])
    labeled_ds = dp.preprocess_and_tokenize(labeled_ds, tokenizer_name="ProsusAI/finbert")
    mt = Model_Train(name=f"2_{cases['Market'][test_id]}_GenericInvestingPosts_", db=paper_results_db)
    mt.start(balanced_df, labeled_ds)
