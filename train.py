# Helibrunna - A HuggingFace compatible xLSTM trainer.
# Copyright (c) 2024 Dr. Tristan Behrens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import datetime
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from accelerate import Accelerator
from dacite import from_dict
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
import fire
import hashlib
import json
from omegaconf import OmegaConf
import multiprocessing
import shutil
import sys
import tempfile
import time
import urllib.request
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer, BpeTrainer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import PreTrainedTokenizerFast
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from source.utilities import display_logo, human_readable_number, load_configs, validate_config, is_torch_compile_ready, model_from_config, save_model

# Try to import h5py for H5 file support
try:
    import h5py
    import pandas as pd
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import torch
# torch.autograd.set_detect_anomaly(True)  # DISABLED for performance

# Import the LinearWarmupCosineAnnealing scheduler from the experiments module.
# Source: https://github.com/NX-AI/xlstm/tree/main
if not os.path.exists("experiments/lr_scheduler.py"):
    url = "https://raw.githubusercontent.com/NX-AI/xlstm/main/experiments/lr_scheduler.py"
    os.makedirs("experiments", exist_ok=True)
    urllib.request.urlretrieve(url, "experiments/lr_scheduler.py")
from experiments.lr_scheduler import LinearWarmupCosineAnnealing

# 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class H5SequenceDataset(TorchDataset):
    """
    A PyTorch Dataset for reading sequences from an HDF5 file with a specific structure.

    The HDF5 file is expected to contain multiple datasets named 'raw_data_X',
    where X is an integer. Each of these datasets contains tuples of
    (protein_id, sequence).

    This class reads data in a streaming fashion and is memory-efficient.
    It returns raw strings that will be tokenized by the data collator.
    """
    def __init__(self, h5_path: str, accelerator):
        self.h5_path = h5_path
        self.accelerator = accelerator
        self.dataset_keys = []
        self.cumulative_lengths = []
        self.total_length = 0

        try:
            with h5py.File(self.h5_path, 'r') as f:
                # Get all dataset keys and sort them numerically
                keys = [key for key in f.keys() if key.startswith('raw_data_')]
                keys.sort(key=lambda x: int(x.split('_')[-1]))
                self.dataset_keys = keys

                # Calculate cumulative lengths for efficient indexing
                lengths = [len(f[key]) for key in self.dataset_keys]
                self.cumulative_lengths = np.cumsum(lengths)
                self.total_length = self.cumulative_lengths[-1] if self.cumulative_lengths.size > 0 else 0

            self.accelerator.print(f"Initialized H5SequenceDataset from {h5_path}. Found {len(self.dataset_keys)} data chunks with a total of {self.total_length} sequences.")

        except Exception as e:
            self.accelerator.print(f"Error initializing H5SequenceDataset: {e}")
            raise

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx >= self.total_length:
            raise IndexError("Index out of range")

        # Find which dataset the index falls into
        dataset_index = np.searchsorted(self.cumulative_lengths, idx, side='right')
        
        # Calculate the local index within that dataset
        if dataset_index == 0:
            local_index = idx
        else:
            local_index = idx - self.cumulative_lengths[dataset_index - 1]

        with h5py.File(self.h5_path, 'r') as f:
            dataset = f[self.dataset_keys[dataset_index]]
            entry = dataset[local_index]

            # Handle the (protein_id, sequence) tuple format
            if hasattr(entry, 'item') and isinstance(entry.item(), tuple):
                _, sequence = entry.item()
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                _, sequence = entry[0], entry[1]
            else:
                # Fallback for simple string/byte entries
                sequence = entry

            # Decode bytes to string if necessary
            if isinstance(sequence, bytes):
                sequence = sequence.decode('utf-8')

            # Return raw string - tokenization will be handled by the data collator
            return str(sequence)


class H5MultilabelClassificationDataset(TorchDataset):
    """
    A PyTorch Dataset for multilabel classification that reads sequences from a CSV
    and corresponding multi-hot encoded labels from an HDF5 file where labels are
    stored as pandas DataFrames.
    """
    def __init__(self, csv_path: str, h5_path: str, sequence_column: str, id_column: str, accelerator, separator: str = '\t'):
        self.accelerator = accelerator
        self.sequence_column = sequence_column
        self.id_column = id_column

        # Load sequences from CSV and set index
        self.accelerator.print(f"Loading sequences from {csv_path}...")
        self.dataframe = pd.read_csv(csv_path, sep=separator).set_index(id_column)
        
        # Load and combine label DataFrames from HDF5
        self.accelerator.print(f"Loading labels from {h5_path}...")
        label_dfs = []
        try:
            with h5py.File(h5_path, 'r') as f:
                # Find keys that correspond to pandas DataFrames (which are HDF5 groups)
                # A common pattern is having 'axis1' inside the group.
                df_keys = [key for key in f.keys() if isinstance(f[key], h5py.Group) and 'axis1' in f[key]]

            if not df_keys:
                raise ValueError(f"Could not find any pandas DataFrame objects in H5 file: {h5_path}")

            self.accelerator.print(f"Found label DataFrames in H5 keys: {df_keys}")

            for key in df_keys:
                df = pd.read_hdf(h5_path, key=key)
                label_dfs.append(df)
        
        except Exception as e:
            self.accelerator.print(f"Error reading HDF5 file: {e}")
            raise

        # Concatenate all label dataframes
        combined_labels_df = pd.concat(label_dfs, axis=1)

        # Align labels with the sequence dataframe
        # This ensures that we have the same entries in the same order
        self.accelerator.print("Aligning sequences and labels...")
        self.dataframe, self.aligned_labels = self.dataframe.align(combined_labels_df, join='inner', axis=0)
        
        # Fill any potential missing values in labels with 0
        self.aligned_labels.fillna(0, inplace=True)

        # Convert to torch tensor for performance
        self.labels_tensor = torch.tensor(self.aligned_labels.values, dtype=torch.float)
        self.num_classes = self.labels_tensor.shape[1]

        # Calculate pos_weight for handling class imbalance
        self.accelerator.print("Calculating pos_weight for class imbalance...")
        num_samples = len(self.labels_tensor)
        num_positives = torch.sum(self.labels_tensor, dim=0)
        num_negatives = num_samples - num_positives
        
        # Avoid division by zero for classes with no positive samples
        # A weight of 1.0 is neutral in this case.
        self.pos_weight = torch.ones_like(num_positives)
        has_positives = num_positives > 0
        self.pos_weight[has_positives] = num_negatives[has_positives] / num_positives[has_positives]
        self.accelerator.print(f"Calculated pos_weight for {self.num_classes} classes.")

        # Reset index to allow for integer-based indexing in __getitem__
        self.dataframe.reset_index(inplace=True)

        self.accelerator.print(f"Initialized H5MultilabelClassificationDataset with {len(self.dataframe)} aligned samples and {self.num_classes} classes.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Data is pre-loaded and aligned, so just retrieve by index
        row = self.dataframe.iloc[idx]
        sequence = row[self.sequence_column]
        labels = self.labels_tensor[idx]

        return {"sequence": str(sequence), "labels": labels}


class MultilabelDataCollator:
    """
    Data collator for multilabel classification.
    Tokenizes sequences and pads them, and stacks labels.
    BERT-compatible tokenization with special tokens.
    """
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        sequences = [example["sequence"] for example in examples]
        labels = [example["labels"] for example in examples]

        # Use BERT-style batch encoding for consistency
        batch = self.tokenizer.batch_encode_plus(
            sequences,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            add_special_tokens=True,  # Add [CLS] and [SEP] like BERT
            return_attention_mask=True,  # For better sequence handling
            return_token_type_ids=False,  # Not needed for single sequences
            return_tensors="pt"
        )
        
        batch["labels"] = torch.stack(labels)
        
        return batch


class ProteinSequenceDataCollator:
    """
    Data collator for protein sequences that handles tokenization of raw strings.
    Similar to DataCollatorForLanguageModeling but for protein sequences.
    """
    def __init__(self, tokenizer, max_length, mlm=False, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        # Handle different input formats
        if isinstance(examples[0], str):
            # Direct string inputs from H5SequenceDataset
            sequences = examples
        elif isinstance(examples[0], dict):
            if "text" in examples[0]:
                # Dict with "text" field
                sequences = [example["text"] for example in examples]
            elif "input_ids" in examples[0]:
                # Already tokenized - this shouldn't happen with our H5 datasets
                # But if it does, we need to handle it gracefully
                # This is likely from a cached HuggingFace dataset
                return self._handle_pretokenized_batch(examples)
            else:
                # Unknown dict format - convert to strings
                sequences = [str(example) for example in examples]
        else:
            # Unknown format - convert to strings
            sequences = [str(example) for example in examples]

        # Tokenize the batch
        batch = self.tokenizer(
            sequences,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # For causal LM, shift the inputs to create labels
        if not self.mlm:
            # Causal language modeling
            input_ids = batch["input_ids"]
            labels = input_ids.clone()
            
            # Shift labels: labels[i] = input_ids[i+1]
            # The last token gets a special "fill" token or is ignored
            labels = torch.roll(labels, -1, dims=1)
            # Set the last position to ignore index or a special fill token
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                labels[:, -1] = -100  # Ignore the last position
            
            batch["labels"] = labels

        return batch
    
    def _handle_pretokenized_batch(self, examples):
        """Handle pre-tokenized examples from cached HuggingFace datasets."""
        # Extract input_ids from the examples
        input_ids_list = [example["input_ids"] for example in examples]
        
        # Find the maximum length and pad
        max_len = max(len(ids) for ids in input_ids_list)
        max_len = min(max_len, self.max_length)  # Respect max_length
        
        padded_input_ids = []
        for ids in input_ids_list:
            # Truncate if necessary
            if len(ids) > max_len:
                ids = ids[:max_len]
            # Pad if necessary
            while len(ids) < max_len:
                ids.append(self.tokenizer.pad_token_id)
            padded_input_ids.append(ids)
        
        # Convert to tensor
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long)
        }
        
        # For causal LM, create labels by shifting
        if not self.mlm:
            input_ids = batch["input_ids"]
            labels = input_ids.clone()
            labels = torch.roll(labels, -1, dims=1)
            labels[:, -1] = -100  # Ignore the last position
            batch["labels"] = labels
        
        return batch


def main(*config_paths, preprocess=False):
    """
    Main function to run the training process, compatible with python-fire.

    Args:
        *config_paths (str): The paths to the configuration files.
        preprocess (bool): Whether to only preprocess the dataset and tokenizer.
    """
    # Convert tuple to list
    config_paths = list(config_paths)

    # Check if any configuration files are provided.
    if len(config_paths) == 0:
        print("No configuration files provided.")
        sys.exit(1)

    # Run preprocessing or training.
    if preprocess:
        preprocess_only(config_paths)
    else:
        run_training(config_paths)


def run_training(config_paths: list[str]):
    """
    Run the training process based on the provided configuration file.
    Args:
        config_path (str): The path to the configuration file.
    Raises:
        FileNotFoundError: If the configuration file is not found.
    Returns:
        None
    """

    # Load the configuration.
    config = load_configs(config_paths)
    #validate_config(config)

    # Specify the output_dir.
    run_dir = "run_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = os.path.join(config.training.output_dir, run_dir)

    # Initialize the loggers.
    loggers = []
    if "wandb_project" in config.training and config.training.wandb_project is not None and config.training.wandb_project != "":
        loggers.append("wandb")

    # Get gradient accumulation steps.
    gradient_accumulation_steps = config.training.get("gradient_accumulation_steps", 1)
    #config.training.batch_size = config.training.batch_size * gradient_accumulation_steps
    mixed_precision = config.training.get("mixed_precision", None)

    # Initialize the accelerator.
    accelerator = Accelerator(
        log_with=loggers,
        project_dir=output_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision
    )

    # Display the logo.
    if accelerator.is_local_main_process:
        display_logo()

    # Display mixed precision.
    if accelerator.is_local_main_process:
        if mixed_precision is not None:
            print(f"Mixed precision enabled. Precision: {mixed_precision}")
        else:
            print("Mixed precision disabled.")

    # Create the output directory.
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.print(f"Output directory: {output_dir}")

    # Set log every step to save every step.
    if "log_every_step" not in config.training:
        config.training.log_every_step = 1
    if "save_every_step" not in config.training:
        config.training.save_every_step = -1

    # Preprocess based on task type and prepare task-specific components.
    task_type = config.training.get("task_type", "lm")
    model_type = config.model.get("type", config.training.get("model_name", "")).lower()

    if task_type == "classification":
        tokenized_datasets, tokenizer, label_encoder = preprocess(config, accelerator)
        num_classes = len(label_encoder)
        config.model.vocab_size = tokenizer.vocab_size
        config.model.num_classes = num_classes
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    elif task_type == "multilabel_classification":
        tokenized_datasets, tokenizer, num_classes = preprocess(config, accelerator)
        config.model.vocab_size = tokenizer.vocab_size
        config.model.num_classes = num_classes
        data_collator = MultilabelDataCollator(
            tokenizer=tokenizer,
            max_length=config.model.context_length
        )
    else: # Language Modeling
        tokenized_datasets, tokenizer = preprocess(config, accelerator)
        fill_token = config.tokenizer.fill_token
        if fill_token is None:
            raise Exception("Fill token is missing for language modeling task.")
        fill_token_id = tokenizer.convert_tokens_to_ids(fill_token)
        vocab_size = tokenizer.vocab_size
        config.model.vocab_size = vocab_size
        if hasattr(config.model, 'num_classes'):
            config.model.num_classes = None
        
        if model_type == "bert":
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
        else:
            # Use our custom data collator for protein sequences
            data_collator = ProteinSequenceDataCollator(
                tokenizer=tokenizer,
                max_length=config.model.context_length,
                mlm=False
            )

    # Create the model.
    accelerator.print("Creating model...")
    model = model_from_config(config.model, device=accelerator.device)

    # If the task is multilabel classification, swap the head of the model.
    if task_type == "multilabel_classification":
        accelerator.print("Swapping model head for multilabel classification...")
        # Get the hidden size from the model's configuration.
        # The lm_head of the xLSTM model has shape (hidden_size, vocab_size).
        hidden_size = model.lm_head.in_features
        
        # Create a new head for classification.
        new_head = torch.nn.Linear(hidden_size, num_classes)
        
        # Initialize the classification head properly for multilabel classification
        # Use Xavier initialization and initialize bias to small negative value
        torch.nn.init.xavier_uniform_(new_head.weight)
        torch.nn.init.constant_(new_head.bias, -2.0)  # Small negative bias for better initial performance
        
        # Replace the old head with the new one.
        model.lm_head = new_head
        accelerator.print(f"Model head swapped. New head: {model.lm_head}")
        accelerator.print("Classification head initialized with Xavier weights and negative bias")

    #model = model.to(device=accelerator.device)
    #model.reset_parameters()

    # Apply precision. Let the accelerator handle it.
    # training_dtype = get_torch_dtype(config.training.weight_precision)
    # model = model.to(dtype=training_dtype)
    # accelerator.print(f"Training dtype: {training_dtype}")

    # Attempt torch compile.
    if config.training.get("torch_compile", True):
        if not is_torch_compile_ready():
            accelerator.print("WARNING: GPU is not torch compile ready. Training may be slower.")
        model = torch.compile(model)
        print("Model compiled.")

    # Print the model.
    accelerator.print(model)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_human = human_readable_number(num_params)
    accelerator.print(f"Number of parameters: {num_params:_} ({num_params_human})")

    # Prepare the DataLoader from the tokenized dataset.
    # Each batch will be padded to the maximum length in the batch.
    accelerator.print("Preparing DataLoader...")
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    # Create validation dataloader if validation data exists
    eval_dataloader = None
    if "validation" in tokenized_datasets:
        accelerator.print("Preparing validation DataLoader...")
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"],
            batch_size=config.training.batch_size,
            shuffle=False,  # Don't shuffle validation data
            collate_fn=data_collator
        )

    # Estimate the number of steps.
    num_steps = config.training.num_epochs * len(tokenized_datasets["train"]) // config.training.batch_size
    num_steps = num_steps // accelerator.num_processes
    accelerator.print(f"Estimated number of steps: {num_steps:_}")

    # If the lr_decay_until_steps is set to "auto", set it to the number of steps.
    if config.training.lr_decay_until_steps == "auto":
        config.training.lr_decay_until_steps = num_steps

    # If the lr_warmup_steps is a percentage, convert it to a number of steps.
    if isinstance(config.training.lr_warmup_steps, str):
        percentage = config.training.lr_warmup_steps
        if not percentage.endswith("%"):
            raise ValueError(f"Invalid percentage: {percentage}")
        percentage = float(percentage[:-1]) / 100
        config.training.lr_warmup_steps = int(num_steps * percentage)

    # Prepare the optimizer and learning rate scheduler.
    optimizer_groups = create_weight_decay_optim_groups(model)
    optimizer = torch.optim.AdamW(
        (
            {"weight_decay": config.training.weight_decay, "params": optimizer_groups[0]},
            {"weight_decay": 0.0, "params": optimizer_groups[1]},
        ),
        lr=config.training.lr,
        betas=(0.9, 0.999),  # Same as BERT default
        eps=1e-8,            # Same as BERT default
    )
    
    # Use different schedulers based on config
    total_steps = config.training.lr_decay_until_steps if config.training.lr_decay_until_steps != "auto" else num_steps
    
    # Check if cosine scheduler is requested
    scheduler_type = config.training.get("lr_scheduler_type", "linear")
    
    if scheduler_type == "cosine":
        accelerator.print("Using cosine annealing scheduler")
        lr_scheduler = LinearWarmupCosineAnnealing(
            optimizer,
            warmup_steps=config.training.lr_warmup_steps,
            decay_until_step=total_steps,
            max_lr=config.training.lr,
            min_lr=config.training.lr * config.training.lr_decay_factor
        )
    else:
        accelerator.print("Using linear scheduler with warmup")
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.training.lr_warmup_steps,
            num_training_steps=total_steps
        )

    # Prepare model, optimizer, and dataloader for accelerator.
    if eval_dataloader is not None:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
    else:
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Get some parameters.
    save_every_step = config.training.save_every_step
    log_every_step = config.training.log_every_step
    eval_every_step = config.training.get("eval_every_step", 0)  # Add eval_every_step
    num_epochs = config.training.num_epochs
    enable_mixed_precision = config.training.enable_mixed_precision
    wandb_project = config.training.get("wandb_project", None)
    max_steps = config.training.get("max_steps", None)  # Get max_steps before deleting config  

    # Get a subset of the config that includes only the model.
    model_config = OmegaConf.select(config, "model")

    # Create the readme.
    create_readme(output_dir, config)

    # Get the model name.
    model_name = config.training.model_name

    # Save the config as yaml and delete it.
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)
    del config

    # Save the tokenizer.
    tokenizer.save_pretrained(output_dir)

    # Enable trackers.
    if wandb_project is not None:
        accelerator.print(f"Enabling wandb logging for project: {wandb_project}")
        config_dict = OmegaConf.to_container(model_config)
        # Add num_params to the config.
        config_dict["num_params"] = num_params
        config_dict["num_params_human"] = num_params_human
        wandb_run = run_dir + "-" + model_name
        accelerator.init_trackers(
            project_name=wandb_project, 
            config=config_dict,
            init_kwargs={"wandb": {"name": wandb_run}}
        )

    # Training loop.
    step = 0
    running_loss = []
    # For multilabel metrics collection
    running_predictions = []
    running_labels = []
    history = {
        "loss": [],
        "lr": [],
        "epoch": [],
        "step": [],
    }
    average_loss = 0.0
    # Add a green progress bar.
    progress_bar = tqdm(total=num_steps, desc="Training", unit="step", colour="GREEN")

    # Ignore tokens during loss calculation.
    ignore_index = -1
    if tokenizer.pad_token is not None:
        ignore_index = tokenizer.pad_token_id
    accelerator.print(f"Ignore index: {ignore_index}")

    # Define validation evaluation function for multilabel classification
    def evaluate_validation():
        if eval_dataloader is None or task_type != "multilabel_classification":
            return {}
        
        accelerator.print("Running validation evaluation...")
        model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader):
                # Limit validation to first 100 batches for faster evaluation
                if batch_idx >= 100:
                    break
                    
                inputs = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
                labels = batch["labels"]
                
                # Forward pass
                outputs = model(inputs)
                
                # Pool the outputs (same as training)
                if attention_mask is not None:
                    last_token_indices = attention_mask.sum(dim=1) - 1
                    batch_indices = torch.arange(outputs.size(0), device=outputs.device)
                    pooled_outputs = outputs[batch_indices, last_token_indices]
                else:
                    pooled_outputs = outputs[:, -1, :]
                
                # Compute loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pooled_outputs, 
                    labels,
                    pos_weight=pos_weight
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions and labels
                all_predictions.append(pooled_outputs.detach())
                all_labels.append(labels.detach())
        
        model.train()  # Switch back to training mode
        
        if len(all_predictions) > 0:
            # Concatenate all predictions and labels
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute metrics
            metrics = compute_multilabel_metrics(all_predictions, all_labels)
            metrics["val_loss"] = total_loss / num_batches if num_batches > 0 else 0.0
            
            accelerator.print(f"Validation metrics: {metrics}")
            return metrics
        
        return {"val_loss": total_loss / num_batches if num_batches > 0 else 0.0}

    # Do the training.
    model.train()

    # Get pos_weight for multilabel classification if available
    pos_weight = None
    if task_type == "multilabel_classification":
        train_dataset = tokenized_datasets["train"]
        if hasattr(train_dataset, 'pos_weight'):
            accelerator.print("Using pos_weight for loss calculation.")
            pos_weight = train_dataset.pos_weight.to(accelerator.device)
    
    for epoch in range(num_epochs):
        accelerator.print(f"Starting epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx == 0:
                accelerator.print("Processing first batch...")
            
            if batch_idx % 100 == 0:
                accelerator.print(f"Processing batch {batch_idx}")
                
                # Add detailed debugging every 1000 batches
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    accelerator.print("=== DEBUGGING INFO ===")
                    accelerator.print(f"Current step: {step}")
                    accelerator.print(f"Current loss: {average_loss:.6f}")
                    accelerator.print(f"Learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")
                    
                    # Check gradient norms
                    total_grad_norm = 0
                    param_count = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                            param_count += 1
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    accelerator.print(f"Total gradient norm: {total_grad_norm:.6f}")
                    accelerator.print(f"Parameters with gradients: {param_count}")
                    
                    # Check model output statistics
                    if task_type == "multilabel_classification":
                        with torch.no_grad():
                            sample_outputs = outputs[:1]  # First sample
                            sample_pooled = pooled_outputs[:1]
                            accelerator.print(f"Sample model output range: [{sample_outputs.min().item():.4f}, {sample_outputs.max().item():.4f}]")
                            accelerator.print(f"Sample pooled output range: [{sample_pooled.min().item():.4f}, {sample_pooled.max().item():.4f}]")
                            accelerator.print(f"Sample sigmoid range: [{torch.sigmoid(sample_pooled).min().item():.4f}, {torch.sigmoid(sample_pooled).max().item():.4f}]")
                    
                    accelerator.print("=== END DEBUG ===")

            if task_type == "classification":
                # For classification, the batch already contains input_ids and labels
                inputs = batch['input_ids'].to(accelerator.device)
                labels = batch['labels'].to(accelerator.device)
                with accelerator.accumulate(model):
                    outputs = model(inputs)
                    # For classification, use only the last token's output
                    # outputs shape: [batch_size, seq_len, num_classes]
                    # We want: [batch_size, num_classes]
                    outputs = outputs[:, -1, :]  # Take last token output
                    loss = torch.nn.functional.cross_entropy(outputs, labels)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    running_loss.append(loss.item())
                    average_loss = sum(running_loss) / len(running_loss)

            elif task_type == "multilabel_classification":
                if batch_idx == 0:
                    accelerator.print("Setting up multilabel classification batch...")
                
                inputs = batch['input_ids'].to(accelerator.device)
                labels = batch['labels'].to(accelerator.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(accelerator.device)
                
                if batch_idx == 0:
                    accelerator.print(f"Input shape: {inputs.shape}, Labels shape: {labels.shape}")
                    accelerator.print("Starting forward pass...")
                
                with accelerator.accumulate(model):
                    # xLSTM doesn't support attention_mask parameter, only inputs
                    outputs = model(inputs)
                    
                    if batch_idx == 0:
                        accelerator.print(f"Model output shape: {outputs.shape}")
                        accelerator.print("Computing pooled outputs...")
                    
                    # Pool the outputs across the sequence length dimension
                    # For xLSTM, use the LAST non-padded token instead of mean pooling
                    # This is more appropriate for sequence classification with autoregressive models
                    if attention_mask is not None:
                        # Find the last non-padded position for each sequence
                        last_token_indices = attention_mask.sum(dim=1) - 1  # Get index of last non-padded token
                        batch_indices = torch.arange(outputs.size(0), device=outputs.device)
                        pooled_outputs = outputs[batch_indices, last_token_indices]
                    else:
                        # If no attention mask, use the last token
                        pooled_outputs = outputs[:, -1, :]
                    
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pooled_outputs, 
                        labels,
                        pos_weight=pos_weight
                    )
                    accelerator.backward(loss)
                    
                    # Add gradient clipping to prevent NaN values
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    running_loss.append(loss.item())
                    average_loss = sum(running_loss) / len(running_loss)
                    
                    # Collect predictions and labels for metrics (only from main process)
                    if accelerator.is_local_main_process:
                        running_predictions.append(pooled_outputs.detach())
                        running_labels.append(labels.detach())

            else: # Language Modeling
                inputs = batch['input_ids'].to(accelerator.device)
                if model_type == "bert":
                    # For BERT/MLM, labels are the same as inputs, masking is handled by the data collator
                    with accelerator.accumulate(model):
                        outputs = model(inputs, labels=inputs)
                        loss = outputs.loss
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        running_loss.append(loss.item())
                        average_loss = sum(running_loss) / len(running_loss)
                else:
                    # Causal LM training
                    # Get the labels by shifting the inputs. Remove the first token. Fill the last token.
                    labels = torch.roll(inputs, -1, dims=1)
                    labels[:, -1] = fill_token_id
                    # Forward pass.
                    with accelerator.accumulate(model):
                        outputs = model(inputs)
                        loss = torch.nn.functional.cross_entropy(
                            outputs.view(-1, vocab_size),
                            labels.view(-1),
                            ignore_index=ignore_index,
                        )
                        accelerator.backward(loss)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        running_loss.append(loss.item())
                        average_loss = sum(running_loss) / len(running_loss)

            # Next step.
            step += 1

            # Check if we've reached max_steps (like BERT training)
            if max_steps is not None and step >= max_steps:
                accelerator.print(f"Reached max_steps={max_steps}. Stopping training.")
                break

            # Compute epoch with fraction.
            epoch_fraction = num_epochs * step / num_steps

            # Save every step.
            if step % save_every_step == 0 and step > 0 and save_every_step > 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    save_model(accelerator.unwrap_model(model), model_config, checkpoint_dir)

            # Log every step.
            if step % log_every_step == 0 and step > 0 and log_every_step > 0 and accelerator.is_local_main_process:
                # Update the log.
                last_lr = lr_scheduler.get_last_lr()[0]
                history["loss"].append(average_loss)
                history["lr"].append(last_lr)
                history["step"].append(step)
                history["epoch"].append(epoch_fraction)
                
                # Prepare log data
                log_data = {"loss": average_loss, "lr": last_lr, "epoch": epoch_fraction}
                
                # Compute multilabel metrics if we have collected predictions
                if task_type == "multilabel_classification" and len(running_predictions) > 0:
                    # Concatenate all collected predictions and labels
                    all_predictions = torch.cat(running_predictions, dim=0)
                    all_labels = torch.cat(running_labels, dim=0)
                    
                    # Compute metrics
                    metrics = compute_multilabel_metrics(all_predictions, all_labels)
                    
                    # Add metrics to log data
                    for metric_name, metric_value in metrics.items():
                        log_data[f"train_{metric_name}"] = metric_value
                    
                    # Clear collected predictions and labels
                    running_predictions = []
                    running_labels = []
                
                running_loss = []

                # Log to wandb.
                if wandb_project is not None:
                    accelerator.log(log_data, step=step)
                # Update the progressbar. Use the step as the total. Also display the loss and lr.
                progress_bar.set_postfix({"loss": average_loss, "lr": last_lr, "epoch": epoch_fraction})
                progress_bar.update(log_every_step)
            
            # Evaluate on validation set every eval_every_step
            if step % eval_every_step == 0 and step > 0 and eval_every_step > 0 and accelerator.is_local_main_process:
                val_metrics = evaluate_validation()
                if val_metrics and wandb_project is not None:
                    # Log validation metrics to wandb
                    accelerator.log(val_metrics, step=step)
        
        # Break outer epoch loop if max_steps reached
        if max_steps is not None and step >= max_steps:
            break

    # End training.
    progress_bar.close()
    accelerator.wait_for_everyone()
    accelerator.end_training()

    # Print some information.
    accelerator.print(f"Training completed. Epochs: {epoch}, Steps: {step}")

    # Save the last model.
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}-last")
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        save_model(accelerator.unwrap_model(model), model_config, checkpoint_dir)

    # Save the history as JSON.
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)


def create_weight_decay_optim_groups(model):

    # If the model has the methor _create_weight_decay_optim_groups, use it. Likely only for xLSTM.
    if hasattr(model, "_create_weight_decay_optim_groups"):
        return model._create_weight_decay_optim_groups()
    
    # Following the implementation of xLSTM, we split the parameters into two groups: decay and no_decay.
    # The decay group contains all parameters except the ones with a shape of 1.
    # The no_decay group contains all parameters with a shape of 1.
    else:
        decay = set()
        no_decay = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.ndim > 1:
                    decay.add(param)
                elif param.ndim == 1:
                    no_decay.add(param)
                else:
                    raise ValueError(f"Unsupported parameter shape: {param.shape}")
        return tuple(decay), tuple(no_decay)


def get_torch_dtype(dtype: str) -> torch.dtype:
    """
    Returns the corresponding torch.dtype for the given dtype string.

    Args:
        dtype (str): The dtype string.

    Returns:
        torch.dtype: The corresponding torch.dtype.

    Raises:
        ValueError: If the dtype is unknown.
    """

    if dtype == "float32":
        return torch.float32
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def compute_multilabel_metrics(predictions, labels):
    """
    Compute multilabel classification metrics.
    
    Args:
        predictions (torch.Tensor): Raw logits from the model [batch_size, num_classes]
        labels (torch.Tensor): Ground truth binary labels [batch_size, num_classes]
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    # Convert to numpy arrays
    preds_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    # Apply sigmoid to get probabilities for ROC-AUC
    preds_prob = torch.sigmoid(predictions).detach().cpu().numpy()
    
    # Apply threshold of 0.5 for binary classification metrics
    preds_binary = (preds_prob > 0.5).astype(int)
    
    # Compute metrics
    try:
        # ROC-AUC using probabilities
        roc_auc = roc_auc_score(labels_np, preds_prob, average="samples")
    except Exception:
        # In case of issues (e.g., all labels are 0 for some samples)
        roc_auc = 0.0
    
    # Precision, Recall, F1 using binary predictions
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np, preds_binary, average='samples', zero_division=0
    )
    
    # Accuracy using binary predictions
    accuracy = accuracy_score(labels_np, preds_binary)
    
    return {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc)
    }
    

def create_readme(output_dir, config):
    """
    Create a README file based on a template and provided configuration.
    Args:
        output_dir (str): The directory where the README file will be saved.
        config (dict): The configuration dictionary containing the necessary information.
    Raises:
        FileNotFoundError: If the template or banner file is not found.
    Returns:
        None
    """

    # Load the template.
    template_path = os.path.join("assets", "readmetemplate.md")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    # Load the template.
    with open(template_path, "r") as f:
        readme_text = f.read()

    # Project name.
    model_name = config.training.model_name

    # Configuration convert the configuration to a yaml string.
    configuration = OmegaConf.to_yaml(config)

    # Base model.
    base_model = "None"
    if "base_model" in config.model:
        base_model = config.model.base_model

    # Tags.
    tags = ["NLP"]
    if "tags" in config.model:
        tags = config.model.tags.split(",")
    tags = "\n".join([f"  - {tag}" for tag in tags])

    # Languages.
    languages = ["en"]
    if "languages" in config.model:
        languages = config.model.languages.split(",")
    languages = "\n".join([f"  - {language}" for language in languages])

    # Datasets.
    if hasattr(config.dataset, 'hugging_face_ids'):
        datasets = config.dataset.hugging_face_ids
        datasets = "\n".join([f"  - {dataset}" for dataset in datasets])
    else:
        # For classification tasks with local files
        datasets = f"  - {config.dataset.path if hasattr(config.dataset, 'path') else 'Local dataset'}"
    
    # License.
    license = "mit"

    # Format the template.
    readme_text = readme_text.format(
        model_name=model_name,
        configuration=configuration,
        base_model=base_model,
        tags=tags,
        languages=languages,
        datasets=datasets,
        license=license,
    )

    # Save the readme.
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_text)

    # Copy the banner.
    banner_path = os.path.join("assets", "trainedwithhelibrunna.jpg")
    if not os.path.exists(banner_path):
        raise FileNotFoundError(f"Banner not found: {banner_path}")
    banner_target_path = os.path.join(output_dir, "banner.jpg")
    shutil.copy(banner_path, banner_target_path)


def preprocess_only(config_paths: list[str]):

    # Load the configuration.
    config = load_configs(config_paths)
    # validate_config(config)  # Commented out for classification task

    # Initialize the accelerator.
    accelerator = Accelerator()

    _ = preprocess(config, accelerator, ask_for_overwrite=True)


def preprocess(config, accelerator=None, ask_for_overwrite=False):
    """
    Preprocess dataset and tokenizer based on the task type (language modeling or classification).
    
    Args:
        config (OmegaConf): The configuration object.
        accelerator (Accelerator): The Accelerator instance.
    
    Returns:
        - For LM: (tokenized_datasets, tokenizer)
        - For Classification: (tokenized_datasets, tokenizer, label_encoder)
    """
    task_type = config.training.get("task_type", "lm")

    if task_type == "classification":
        return preprocess_for_classification(config, accelerator, ask_for_overwrite)
    elif task_type == "multilabel_classification":
        return preprocess_for_multilabel_classification(config, accelerator, ask_for_overwrite)
    else:
        return preprocess_for_lm(config, accelerator, ask_for_overwrite)


def preprocess_for_multilabel_classification(config, accelerator, ask_for_overwrite):
    """Preprocess data for a multilabel classification task."""
    model_name = config.training.model_name
    preprocessed_path = f"./preprocessed/{model_name}"
    tokenizer_path = f"./preprocessed/{model_name}/tokenizer"

    # For multilabel classification, we don't pre-tokenize and save the entire dataset
    # because it's handled by the H5MultilabelClassificationDataset.
    # We just need to ensure the tokenizer is created and available.

    if os.path.exists(tokenizer_path) and not ask_for_overwrite:
        accelerator.print("Loading pre-trained tokenizer...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        if accelerator.is_local_main_process:
            if os.path.exists(preprocessed_path) and ask_for_overwrite:
                overwrite = input(f"Tokenizer for {model_name} already exists. Overwrite? [y/n]: ")
                if overwrite.lower() == "y":
                    accelerator.print("Deleting existing preprocessed tokenizer...")
                    if os.path.exists(tokenizer_path):
                        shutil.rmtree(tokenizer_path)
            
            os.makedirs(preprocessed_path, exist_ok=True)

            # For multilabel, we expect a pre-trained tokenizer from a file,
            # often the one used during pre-training.
            accelerator.print("Creating tokenizer for multilabel classification...")
            tokenizer = create_tokenizer(config.tokenizer)
            tokenizer.save_pretrained(tokenizer_path)
    
    accelerator.wait_for_everyone()
    if not accelerator.is_local_main_process:
        # Ensure other processes load the tokenizer after it's created
        while not os.path.exists(f"{tokenizer_path}/tokenizer.json"):
            time.sleep(1)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Now, create the dataset instance.
    accelerator.print("Creating H5 multilabel dataset...")
    dataset = H5MultilabelClassificationDataset(
        csv_path=config.dataset.path,
        h5_path=config.dataset.train_path,  # Updated to use train_path
        sequence_column=config.dataset.sequence_column,
        id_column=config.dataset.id_column,
        accelerator=accelerator
    )
    
    # The training loop expects a dictionary-like object for datasets.
    # We'll wrap our single dataset in a simple dictionary.
    datasets = {"train": dataset}
    num_classes = dataset.num_classes

    return datasets, tokenizer, num_classes



def preprocess_for_classification(config, accelerator, ask_for_overwrite):
    """Preprocess data for a classification task."""
    model_name = config.training.model_name
    preprocessed_path = f"./preprocessed/{model_name}"
    tokenizer_path = f"./preprocessed/{model_name}/tokenizer"
    label_encoder_path = f"./preprocessed/{model_name}/label_encoder.json"
    tokenized_data_path = f"./preprocessed/{model_name}/tokenized_datasets"

    if os.path.exists(tokenized_data_path) and not ask_for_overwrite:
        accelerator.print("Loading preprocessed classification data...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        with open(label_encoder_path, "r") as f:
            label_encoder = json.load(f)
        tokenized_datasets = load_from_disk(tokenized_data_path)
        return tokenized_datasets, tokenizer, label_encoder

    if accelerator.is_local_main_process:
        if os.path.exists(preprocessed_path) and ask_for_overwrite:
            overwrite = input("Preprocessed data already exists. Overwrite? [y/n]: ")
            if overwrite.lower() == "y":
                accelerator.print("Deleting existing preprocessed data...")
                shutil.rmtree(preprocessed_path)
        
        os.makedirs(preprocessed_path, exist_ok=True)

        # Load raw dataset from CSV
        dataset_path = config.dataset.path
        accelerator.print(f"Loading classification dataset from: {dataset_path}")
        raw_datasets = load_dataset("csv", data_files=dataset_path)

        # Create and train a character-level tokenizer for sequences
        accelerator.print("Training character tokenizer...")
        tokenizer = train_char_tokenizer(raw_datasets, config.dataset.sequence_column, config.tokenizer)
        tokenizer.save_pretrained(tokenizer_path)

        # Create and save a label encoder for GO terms
        accelerator.print("Creating label encoder...")
        label_column = config.dataset.label_column
        unique_labels = sorted(raw_datasets["train"].unique(label_column))
        label_encoder = {label: i for i, label in enumerate(unique_labels)}
        with open(label_encoder_path, "w") as f:
            json.dump(label_encoder, f)
        
        # Tokenize sequences and encode labels
        def tokenize_and_encode(example):
            tokenized_input = tokenizer(
                example[config.dataset.sequence_column],
                truncation=True,
                padding=False,
                max_length=config.model.context_length
            )
            return {
                "input_ids": tokenized_input["input_ids"],
                "labels": label_encoder[example[label_column]]
            }

        accelerator.print("Tokenizing and encoding dataset...")
        tokenized_datasets = raw_datasets.map(
            tokenize_and_encode,
            batched=False, # Process one by one for simplicity with labels
            remove_columns=raw_datasets["train"].column_names,
            num_proc=1 if len(raw_datasets["train"]) < 1000 else multiprocessing.cpu_count()
        )
        tokenized_datasets.save_to_disk(tokenized_data_path)

    accelerator.wait_for_everyone()
    
    # Load for all processes
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    with open(label_encoder_path, "r") as f:
        label_encoder = json.load(f)
    tokenized_datasets = load_from_disk(tokenized_data_path)

    return tokenized_datasets, tokenizer, label_encoder


def preprocess_for_lm(config, accelerator, ask_for_overwrite):
    """Preprocess data for a language modeling task."""
    # Check if we have H5 files or HuggingFace IDs
    if hasattr(config.dataset, 'train_path') and config.dataset.train_path.endswith('.h5'):
        return preprocess_for_lm_h5(config, accelerator, ask_for_overwrite)
    else:
        return preprocess_for_lm_huggingface(config, accelerator, ask_for_overwrite)


def preprocess_for_lm_h5(config, accelerator, ask_for_overwrite):
    """Preprocess H5 data for language modeling task."""
    if not HAS_H5PY:
        raise ImportError("h5py and pandas are required for H5 file support. Please install them: pip install h5py pandas")
    
    model_name = config.training.model_name
    preprocessed_path = f"./preprocessed/{model_name}"
    tokenizer_path = f"./preprocessed/{model_name}/tokenizer"
    tokenized_data_path = f"./preprocessed/{model_name}/tokenized_datasets"
    checksum_path = f"./preprocessed/{model_name}/checksum.txt"

    # Compute the checksum from the configuration
    checksum = compute_checksum_from_config(config)

    # Compare the checksum. If the checksum is different, delete the preprocessed data
    preprocess_anyway = False
    if os.path.exists(checksum_path) and os.path.exists(tokenized_data_path):
        with open(checksum_path, "r") as f:
            checksum_from_file = f.read()
        if checksum_from_file != checksum:
            accelerator.print("Checksum mismatch. Preprocessing anyway...")
            preprocess_anyway = True

    # Check if preprocessed data exists and is valid
    if not preprocess_anyway and os.path.exists(tokenized_data_path) and os.path.exists(tokenizer_path):
        accelerator.print("Preprocessed data found. Loading...")
    else:
        if os.path.exists(preprocessed_path):
            if ask_for_overwrite:
                overwrite = input("Preprocessed data exists. Overwrite? (y/n): ")
                if overwrite.lower() != "y":
                    accelerator.print("Exiting...")
                    return None, None
            accelerator.print("Deleting existing preprocessed data...")
            shutil.rmtree(preprocessed_path)
        
        os.makedirs(preprocessed_path, exist_ok=True)

        # Load H5 datasets for language modeling
        accelerator.print(f"Loading H5 datasets for language modeling...")
        
        # Create tokenizer first
        tokenizer = create_tokenizer(config.tokenizer)
        tokenizer.save_pretrained(tokenizer_path)

        # Create datasets that return raw strings (tokenization handled by data collator)
        tokenized_datasets = {}
        
        accelerator.print("Creating H5 datasets...")
        tokenized_datasets["train"] = H5SequenceDataset(
            config.dataset.train_path, 
            accelerator
        )

        if hasattr(config.dataset, 'valid_path') and config.dataset.valid_path:
            tokenized_datasets["validation"] = H5SequenceDataset(
                config.dataset.valid_path, 
                accelerator
            )
        if hasattr(config.dataset, 'test_path') and config.dataset.test_path:
            tokenized_datasets["test"] = H5SequenceDataset(
                config.dataset.test_path, 
                accelerator
            )
        
        # For compatibility with the main training loop, we need to simulate
        # the HuggingFace dataset structure. We'll create a simple wrapper.
        class DatasetWrapper:
            def __init__(self, datasets_dict):
                self.datasets = datasets_dict
            
            def __getitem__(self, key):
                return self.datasets[key]
            
            def save_to_disk(self, path):
                # We don't actually save the datasets since they're streaming from H5
                # Just create the directory and save a marker file
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "h5_datasets.marker"), "w") as f:
                    f.write("H5 streaming datasets - no serialization needed")

        tokenized_datasets = DatasetWrapper(tokenized_datasets)
        tokenized_datasets.save_to_disk(tokenized_data_path)
        
        # Save checksum
        with open(checksum_path, "w") as f:
            f.write(checksum)

    accelerator.wait_for_everyone()
    
    # Load for all processes
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    
    # Check if we have H5 streaming datasets or regular HF datasets
    if os.path.exists(os.path.join(tokenized_data_path, "h5_datasets.marker")):
        # Recreate the H5 streaming datasets
        tokenized_datasets = {}
        tokenized_datasets["train"] = H5SequenceDataset(
            config.dataset.train_path, 
            accelerator
        )
        
        if hasattr(config.dataset, 'valid_path') and config.dataset.valid_path:
            tokenized_datasets["validation"] = H5SequenceDataset(
                config.dataset.valid_path, 
                accelerator
            )
        if hasattr(config.dataset, 'test_path') and config.dataset.test_path:
            tokenized_datasets["test"] = H5SequenceDataset(
                config.dataset.test_path, 
                accelerator
            )
            
        # Wrap in our simple dataset wrapper for compatibility
        class DatasetWrapper:
            def __init__(self, datasets_dict):
                self.datasets = datasets_dict
            
            def __getitem__(self, key):
                return self.datasets[key]
        
        tokenized_datasets = DatasetWrapper(tokenized_datasets)
    else:
        # Load regular HuggingFace datasets
        tokenized_datasets = load_from_disk(tokenized_data_path)

    return tokenized_datasets, tokenizer


def preprocess_for_lm_huggingface(config, accelerator, ask_for_overwrite):
    """Preprocess HuggingFace data for a language modeling task."""
    # This function is the original 'preprocess' logic
    hugging_face_ids = config.dataset.hugging_face_ids
    if isinstance(hugging_face_ids, str):
        hugging_face_ids = [hugging_face_ids]
    model_name = config.training.model_name
    preprocessed_path = f"./preprocessed/{model_name}"
    data_path = f"./preprocessed/{model_name}/data"
    tokenizer_path = f"./preprocessed/{model_name}/tokenizer"
    tokenized_data_path = f"./preprocessed/{model_name}/tokenized_datasets"
    checksum_path = f"./preprocessed/{model_name}/checksum.txt"

    # Compute the checksum from the configuration
    checksum = compute_checksum_from_config(config)

    # Compare the checksum. If the checksum is different, delete the preprocessed data
    preprocess_anyway = False
    if os.path.exists(checksum_path) and os.path.exists(data_path):
        with open(checksum_path, "r") as f:
            checksum_from_file = f.read()
        if checksum_from_file != checksum:
            accelerator.print("Checksum mismatch. Preprocessing anyway...")
            preprocess_anyway = True

    # If tokenizer and tokenized datasets exist, and ask_for_overwrite is True, ask for overwrite
    if os.path.exists(tokenizer_path) and os.path.exists(tokenized_data_path) and ask_for_overwrite and not preprocess_anyway:
        overwrite = input("Preprocessed data already exists. Overwrite? [y/n]: ")
        if overwrite.lower() == "y":
            accelerator.print("Deleting existing preprocessed data...")
            shutil.rmtree(preprocessed_path)

    # If tokenizer and tokenized datasets exist, load them
    if os.path.exists(tokenizer_path) and os.path.exists(tokenized_data_path) and not preprocess_anyway:
        accelerator.print("Loading preprocessed data...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        tokenized_datasets = load_from_disk(tokenized_data_path)
        return tokenized_datasets, tokenizer
    
    # Write the checksum to a file
    os.makedirs(preprocessed_path, exist_ok=True)
    with open(checksum_path, "w") as f:
        f.write(checksum)

    # Download and combine the datasets
    if accelerator.is_local_main_process:
        accelerator.print(f"Loading datasets: {hugging_face_ids}")
        raw_datasets = None
        for dataset_id in hugging_face_ids:
            accelerator.print(f"Downloading dataset: {dataset_id}")
            if os.path.isdir(dataset_id) or dataset_id.endswith('.txt'):
                current_dataset = load_dataset("text", data_files=dataset_id)
            elif dataset_id.endswith('.json') or dataset_id.endswith('.jsonl'):
                current_dataset = load_dataset("json", data_files=dataset_id)
            elif dataset_id.endswith('.csv'):
                current_dataset = load_dataset("csv", data_files=dataset_id)
            else:
                # Assume HuggingFace Hub dataset (with optional config)
                if ":" in dataset_id:
                    ds, config = dataset_id.split(":", 1)
                    current_dataset = load_dataset(ds, config)
                else:
                    current_dataset = load_dataset(dataset_id)
            if raw_datasets is None:
                raw_datasets = current_dataset
            else:
                for split in raw_datasets:
                    if split in current_dataset:
                        raw_datasets[split] = concatenate_datasets([
                            raw_datasets[split],
                            current_dataset[split]
                        ])
        # Save the combined dataset to a temporary directory, then move it to the final directory.
        with tempfile.TemporaryDirectory() as tempdir:
            temp_data_path = os.path.join(tempdir, "data")
            raw_datasets.save_to_disk(temp_data_path)
            shutil.move(temp_data_path, data_path)
        accelerator.print("Datasets downloaded, combined, and saved.")
    else:
        # Other processes wait for the dataset to be downloaded and saved
        while not os.path.exists(data_path):
            time.sleep(1)
        raw_datasets = load_from_disk(data_path)
    
    accelerator.wait_for_everyone()

    # Tokenizer creation.
    if config.tokenizer.type in ["whitespace", "bpe"]:
        if accelerator.is_local_main_process:
            accelerator.print("Training whitespace tokenizer...")
            tokenizer = train_tokenizer(config.tokenizer, raw_datasets)
            tokenizer.save_pretrained(tokenizer_path)
        else:
            while not os.path.exists(f"{tokenizer_path}/tokenizer.json"):
                time.sleep(1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    elif config.tokenizer.type == "pretrained":
        from transformers import AutoTokenizer
        if accelerator.is_local_main_process:
            tokenizer_id = config.tokenizer.pretrained_id
            accelerator.print(f"Loading pre-trained tokenizer: {tokenizer_id}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            # Accept any PreTrainedTokenizerFast (including BertTokenizerFast)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None else '[PAD]'
            tokenizer.save_pretrained(tokenizer_path)
        else:
            while not os.path.exists(f"{tokenizer_path}/tokenizer_config.json"):
                time.sleep(1)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {config.tokenizer.type}")
    
    accelerator.wait_for_everyone()

    # Assign the vocabulary size to the model configuration.
    #assert vocab_size > 0
    #config.model.vocab_size = vocab_size

    # Tokenize the datasets.
    def tokenize_function(example):
        tokenized_example = tokenizer(
            example["text"],
            truncation=True,
            padding=False,
            max_length=config.model.context_length,
        ) 
        return {
            "input_ids": tokenized_example["input_ids"]
        } 

    if accelerator.is_local_main_process:
        accelerator.print("Tokenizing datasets...")
        tokenized_datasets = raw_datasets.map(
            tokenize_function, 
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            num_proc=1 if len(raw_datasets["train"]) < 1000 else multiprocessing.cpu_count()
        )
        tokenized_datasets.save_to_disk(tokenized_data_path)
    else:
        while not os.path.exists(tokenized_data_path):
            time.sleep(1)
        time.sleep(1)
        tokenized_datasets = load_from_disk(tokenized_data_path)

    accelerator.wait_for_everyone()

    # Check a sample.
    if accelerator.is_local_main_process:
        accelerator.print("Sample tokenized text:")
        sample = raw_datasets["train"][0]
        tokenized = tokenized_datasets["train"][0]
        assert list(tokenized.keys()) == ["input_ids"], list(tokenized.keys())
        accelerator.print(f"Original text: {sample}")
        accelerator.print(f"Tokenized text: {tokenized}")

    return tokenized_datasets, tokenizer


def compute_checksum_from_config(config):

    # Convert the configuration to a dictionary
    config_dict = OmegaConf.to_container(config)

    # Use selective fields for the checksum
    checksum_string = "HeliBrunna - A HuggingFace compatible xLSTM trainer by Dr. Tristan Behrens\n"
    checksum_string += "Configuration:\n"
    checksum_string += f"training batch size: {config_dict['training']['batch_size']}\n"
    
    # Handle different dataset formats
    if 'hugging_face_ids' in config_dict['dataset']:
        checksum_string += f"datasets: {','.join(config_dict['dataset']['hugging_face_ids'])}\n"
    elif 'train_path' in config_dict['dataset']:
        checksum_string += f"train_path: {config_dict['dataset']['train_path']}\n"
    elif 'path' in config_dict['dataset']:
        checksum_string += f"dataset_path: {config_dict['dataset']['path']}\n"
    
    checksum_string += f"Have a pleasant day!\n"

    # Compute the checksum. Use MD5
    checksum = hashlib.md5(checksum_string.encode()).hexdigest()
    return checksum


def train_tokenizer(tokenizer_config, raw_datasets):
    """
    Train a tokenizer based on the given configuration and raw datasets.
    Args:
        tokenizer_config (TokenizerConfig): The configuration for the tokenizer.
        raw_datasets (dict): A dictionary containing the raw datasets.
    Returns:
        PreTrainedTokenizerFast: The trained tokenizer.
    Raises:
        ValueError: If the tokenizer type is unknown.
    """
    
    # Initialize the tokenizer.
    if tokenizer_config.type == "whitespace":
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[EOS]"]
        )
    elif tokenizer_config.type == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[EOS]"]
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_config.tokenizer_type}")
    tokenizer.pre_tokenizer = WhitespaceSplit()

    # Train the tokenizer.
    def get_training_corpus():
        dataset = raw_datasets["train"]
        for start_idx in range(0, len(dataset), 1000):
            samples = dataset[start_idx : start_idx + 1000]
            yield samples["text"]
    training_corpus = get_training_corpus()
    tokenizer.train_from_iterator(training_corpus, trainer=trainer)

    # Convert the tokenizer to a fast tokenizer.
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer_path = os.path.join(tempdir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Return the tokenizer.
    return tokenizer


def create_tokenizer(tokenizer_config):
    """Create tokenizer based on config with BERT-compatible special token handling."""
    if tokenizer_config.type == "file":
        # Load vocabulary from file
        vocab_path = tokenizer_config.path
        with open(vocab_path, 'r') as f:
            vocab = [line.strip() for line in f if line.strip()]
        
        vocab_map = {token: i for i, token in enumerate(vocab)}
        tokenizer = Tokenizer(WordLevel(vocab=vocab_map, unk_token=tokenizer_config.unk_token))
        
        # Convert to fast tokenizer with BERT-compatible settings
        with tempfile.TemporaryDirectory() as tempdir:
            tokenizer_path = os.path.join(tempdir, "tokenizer.json")
            tokenizer.save(tokenizer_path)
            fast_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_path,
                clean_up_tokenization_spaces=False,
                model_max_length=512
            )
            
            # Set special tokens (don't add them as they're already in vocab)
            if hasattr(tokenizer_config, 'pad_token'):
                fast_tokenizer.pad_token = tokenizer_config.pad_token
            if hasattr(tokenizer_config, 'unk_token'):
                fast_tokenizer.unk_token = tokenizer_config.unk_token
            if hasattr(tokenizer_config, 'mask_token'):
                fast_tokenizer.mask_token = tokenizer_config.mask_token
            if hasattr(tokenizer_config, 'cls_token'):
                fast_tokenizer.cls_token = tokenizer_config.cls_token
            if hasattr(tokenizer_config, 'sep_token'):
                fast_tokenizer.sep_token = tokenizer_config.sep_token
        
        return fast_tokenizer
    
    elif tokenizer_config.type == "char":
        # Character-level tokenizer (for backward compatibility)
        return train_char_tokenizer(None, None, tokenizer_config)
    
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_config.type}")


def train_char_tokenizer(raw_datasets, sequence_column, tokenizer_config):
    """Trains a character-level tokenizer for protein sequences."""
    
    # Find all unique characters in the sequence column
    def get_char_corpus():
        dataset = raw_datasets["train"]
        for i in range(len(dataset)):
            yield dataset[i][sequence_column]

    all_text = "".join(list(get_char_corpus()))
    vocab = sorted(list(set(all_text)))
    
    # Create a tokenizer from the vocab
    special_tokens = [tokenizer_config.pad_token, tokenizer_config.unk_token]
    vocab_map = {char: i for i, char in enumerate(special_tokens + vocab)}

    # Use WordLevel tokenizer to treat each character as a token
    tokenizer = Tokenizer(WordLevel(vocab=vocab_map, unk_token=tokenizer_config.unk_token))
    
    # Convert to a fast tokenizer
    with tempfile.TemporaryDirectory() as tempdir:
        tokenizer_path = os.path.join(tempdir, "tokenizer.json")
        tokenizer.save(tokenizer_path)
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        fast_tokenizer.add_special_tokens({'pad_token': tokenizer_config.pad_token, 'unk_token': tokenizer_config.unk_token})

    return fast_tokenizer


if __name__ == "__main__":
    # Get the absolute path of the script.
    script_path = os.path.abspath(__file__)
    # Get the directory of the script.
    script_dir = os.path.dirname(script_path)
    # Change the current working directory to the script's directory.
    os.chdir(script_dir)
    
    # Run the main function.
    fire.Fire(main)
