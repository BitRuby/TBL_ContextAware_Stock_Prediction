#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset as torchDS
import pandas as pd
import numpy as np
from datasets import Dataset as HuggingfaceDataset
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from datetime import datetime
import re

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[2]:


class TextDataset(torchDS):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['label'])
        }


# In[ ]:


class R1ForSequenceClassification(nn.Module):
    def __init__(self, base_model, hidden_size=4096, num_labels=3):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # Use first token (similar to [CLS])
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch, hidden]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# In[ ]:


class Model_Train_Ds():
    def __init__(self, name, db):
        self.name = name
        self.db = db
    
    def create_grouped_folds(self, dataframe_with_day: pd.DataFrame, hf_dataset: 'datasets.Dataset', n_splits = 5):
        """
        Splits the HuggingFace dataset into training and testing folds using GroupKFold,
        grouping by the 'day' column to prevent temporal leakage.

        Parameters:
            dataframe_with_day (pd.DataFrame): Original DataFrame that includes 'day' column for grouping.
            hf_dataset (datasets.Dataset): HuggingFace Dataset version of the data (with 'text' and 'label' columns).

        Returns:
            train_folds (List[datasets.Dataset]): List of training Dataset folds.
            test_folds (List[datasets.Dataset]): List of test Dataset folds.
        """
        groups = dataframe_with_day['day']
        labels = hf_dataset['label']

        train_folds = []
        test_folds = []

        group_kfold = GroupKFold(n_splits=n_splits)

        for train_idx, test_idx in group_kfold.split(X=hf_dataset, y=labels, groups=groups):
            # Check for group leakage (sanity check)
            train_days = set(dataframe_with_day.iloc[train_idx]['day'])
            test_days = set(dataframe_with_day.iloc[test_idx]['day'])
            assert train_days.isdisjoint(test_days), "Train and test sets share the same day!"

            # Create folds
            train_folds.append(hf_dataset.select(train_idx).shuffle())
            test_folds.append(hf_dataset.select(test_idx))

        return train_folds, test_folds
    
    def evaluate(self, model, dataloader, device, model_name="base"):
        """
        Evaluate the model on the given data and labels.

        Args:
        dataloader (DataLoader): The DataLoader for the evaluation data.
        device (torch.device): The device to evaluate the model on.

        Returns:
        Tuple[List, List, List, list]: The labels, predictions, probabilities, losses for each batch.
        """
        # Evaluation loop
        all_labels = []
        all_preds = []
        all_probs = []
        all_losses = []

        for batch in tqdm(dataloader, desc="Evaluating Progress...", leave=False, dynamic_ncols=True):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Get the predicted probabilities from the model's outputs
                preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # Convert the probabilities to class labels
                class_preds = torch.argmax(preds, dim=-1)

                all_probs.append(preds.cpu().numpy())  # Store probabilities
                all_preds.append(class_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_losses.append(loss.item())

        return all_labels, all_preds, all_probs, all_losses
    
    def compute_metrics_classification(self, labels, preds, probs, metrics_to_return=None):
        """
        Compute classification metrics based on the model's predictions and the true labels.

        Args:
        labels (any): The true labels.
        preds (any): The model's predictions.
        probs (any): The model's probabilities
        metrics_to_return (list): List of metric names to compute and return.

        Returns:
        dict: The computed classification metrics.
        """
        if metrics_to_return is None:
            metrics_to_return = ["accuracy", "f1", "precision", "recall", "roc_score", "confusion_matrix"]

        metrics = {}

        if "precision" in metrics_to_return or "recall" in metrics_to_return or "f1" in metrics_to_return:
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
            if "precision" in metrics_to_return:
                metrics["precision"] = precision
            if "recall" in metrics_to_return:
                metrics["recall"] = recall
            if "f1" in metrics_to_return:
                metrics["f1"] = f1

        if "accuracy" in metrics_to_return:
            metrics["accuracy"] = accuracy_score(labels, preds)

        if "roc_score" in metrics_to_return:
            metrics["roc_score"] = roc_auc_score(labels, probs, multi_class='ovr')

        if "confusion_matrix" in metrics_to_return:
            metrics["confusion_matrix"] = confusion_matrix(labels, preds)

        return metrics
    
    def train(self, model, dataloader, device, optimizer, scheduler, learning_rate=2e-5, model_name="train"):
        """
        Train the model on the given data and labels.

        Args:
        dataloader (DataLoader): The DataLoader for the training data.
        device (torch.device): The device to train the model on.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): The number of epochs for training.
        num_folds (int): The number of folds for cross-validation.

        Returns:
        Tuple[List, List, List, List]: The labels, predictions, probabilities, and losses for each batch.
        """
        all_labels = []
        all_preds = []
        all_probs = []
        all_losses = []

        for batch in tqdm(dataloader, desc=f"Training Progress...", leave=False, dynamic_ncols=True):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Store labels, predictions and probabilities for metrics calculation
            preds = torch.nn.functional.softmax(outputs.logits, dim=-1)
            class_preds = torch.argmax(preds, dim=-1)

            all_probs.append(preds.detach().cpu().numpy())  # Store probabilities
            all_preds.append(class_preds.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())
            all_losses.append(loss.item())

        return all_labels, all_preds, all_probs, all_losses
    
    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)
    
    def save_results(self, model, result):
        date = datetime.now()
        name = f"{self.name}_{date}"
        model.save_pretrained(f"{re.sub(r'[^A-Za-z0-9]', '', name)}_model.h5")
        self.db.insert_one({
            "name": name,
            "date": date,
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "accuracy": result["accuracy"],
            "roc_score": result["roc_score"],
            "confusion": result["confusion_matrix"].tolist(),
            "max_drawdown": None,
            "profit_factor": None,
            "final_balance": None,
            "profit_per_trade": None,
            "win_rate": None,
            "total_profits": None,
            "cumulative_returns": None
        })

    def start(self, dataframe_with_day: pd.DataFrame, hf_dataset: 'datasets.Dataset', n_splits = 5, num_epochs = 2):
        # Split the dataset into training and evaluation sets
        train_folds, test_folds = self.create_grouped_folds(dataframe_with_day, hf_dataset, n_splits)
        for index, _ in tqdm(enumerate(train_folds), desc="Fold Progress..."):
            fold_num = index + 1
            config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True, config=config)
            for name, param in base_model.named_parameters():
                if name.startswith("transformer.h.0") or name.startswith("transformer.h.1"):
                    param.requires_grad = False
            
            model = R1ForSequenceClassification(base_model, hidden_size=config.hidden_size, num_labels=3)
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            num_training_steps = (num_epochs * len(train_folds[0])) // 8
            num_warmup_steps = int(0.1 * num_training_steps)
            scheduler = self.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            for epoch in tqdm(range(num_epochs), desc="Epoch Progress..."):
                epoch_num = epoch + 1
                train_dataset = TextDataset(train_folds[index].shuffle())
                train_dataloader = DataLoader(train_dataset, batch_size=8)
                model.to(device)
                labels, preds, probs, losses = self.train(model=model, dataloader=train_dataloader, device=device, optimizer=optimizer, scheduler=scheduler, model_name=f"train")
                results = self.compute_metrics_classification(np.concatenate(labels), np.concatenate(preds), np.concatenate(probs))
                print(results)
                test_dataset = TextDataset(test_folds[index])
                test_dataloader = DataLoader(test_dataset, batch_size=8)
                test_labels, test_preds, test_probs, test_losses = self.evaluate(model=model, dataloader=test_dataloader, device=device, model_name=f"eval")
                test_results = self.compute_metrics_classification(np.concatenate(test_labels), np.concatenate(test_preds), np.concatenate(test_probs))
            self.save_results(model, test_results)

