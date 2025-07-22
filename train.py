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
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import PreTrainedTokenizerFast
from source.utilities import display_logo, human_readable_number, load_configs, validate_config, is_torch_compile_ready, model_from_config, save_model

# Try to import h5py for H5 file support
try:
    import h5py
    import pandas as pd
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

import torch
torch.autograd.set_detect_anomaly(True)

# Import the LinearWarmupCosineAnnealing scheduler from the experiments module.
# Source: https://github.com/NX-AI/xlstm/tree/main
if not os.path.exists("experiments/lr_scheduler.py"):
    url = "https://raw.githubusercontent.com/NX-AI/xlstm/main/experiments/lr_scheduler.py"
    os.makedirs("experiments", exist_ok=True)
    urllib.request.urlretrieve(url, "experiments/lr_scheduler.py")
from experiments.lr_scheduler import LinearWarmupCosineAnnealing

# 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



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
        # For classification, the model output should be num_classes, not vocab_size
        config.model.vocab_size = num_classes  # Override vocab_size for classification
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else: # Language Modeling
        tokenized_datasets, tokenizer = preprocess(config, accelerator)
        fill_token = config.tokenizer.fill_token
        if fill_token is None:
            raise Exception("Fill token is missing for language modeling task.")
        fill_token_id = tokenizer.convert_tokens_to_ids(fill_token)
        vocab_size = tokenizer.vocab_size
        config.model.vocab_size = vocab_size
        
        if model_type == "bert":
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create the model.
    accelerator.print("Creating model...")
    model = model_from_config(config.model, device=accelerator.device)
    #model = model.to(device=accelerator.device)
    #model.reset_parameters()

    # Apply precision.
    training_dtype = get_torch_dtype(config.training.weight_precision)
    model = model.to(dtype=training_dtype)
    accelerator.print(f"Training dtype: {training_dtype}")

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
    )
    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer,
        config.training.lr_warmup_steps,
        config.training.lr_decay_until_steps,
        config.training.lr,
        config.training.lr_decay_factor * config.training.lr,
    )

    # Prepare model, optimizer, and dataloader for accelerator.
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Get some parameters.
    save_every_step = config.training.save_every_step
    log_every_step = config.training.log_every_step
    num_epochs = config.training.num_epochs
    enable_mixed_precision = config.training.enable_mixed_precision
    wandb_project = config.training.get("wandb_project", None)  

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

    # Do the training.
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:

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
                running_loss = []

                # Log to wandb.
                if wandb_project is not None:
                    accelerator.log({"loss": average_loss, "lr": last_lr, "epoch": epoch_fraction}, step=step)
                # Update the progressbar. Use the step as the total. Also display the loss and lr.
                progress_bar.set_postfix({"loss": average_loss, "lr": last_lr, "epoch": epoch_fraction})
                progress_bar.update(log_every_step)

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
    else:
        return preprocess_for_lm(config, accelerator, ask_for_overwrite)


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
        
        def load_h5_as_dataset(h5_path, max_sequences=10000):
            """Load H5 file with raw_data_X structure and convert to HuggingFace Dataset."""
            accelerator.print(f"Loading H5 file: {h5_path}")
            
            # Read H5 file - original structure with raw_data_X datasets
            sequences = []
            with h5py.File(h5_path, 'r') as f:
                # Print H5 structure for debugging
                accelerator.print(f"H5 file keys: {list(f.keys())}")
                
                # Original UniParc format has datasets named raw_data_X
                raw_data_keys = [key for key in f.keys() if key.startswith('raw_data_')]
                accelerator.print(f"Found raw_data datasets: {raw_data_keys}")
                
                total_loaded = 0
                for dataset_name in raw_data_keys:
                    if total_loaded >= max_sequences:
                        break
                        
                    accelerator.print(f"Loading from dataset: {dataset_name}")
                    dataset = f[dataset_name]
                    
                    # Check the structure of this dataset
                    accelerator.print(f"Dataset {dataset_name} shape: {dataset.shape}")
                    accelerator.print(f"Dataset {dataset_name} dtype: {dataset.dtype}")
                    
                    # Sample a few entries to understand the structure
                    sample_size = min(max_sequences - total_loaded, len(dataset), 1000)
                    accelerator.print(f"Sampling {sample_size} sequences from {len(dataset)} in {dataset_name}")
                    
                    for i in range(sample_size):
                        if total_loaded >= max_sequences:
                            break
                            
                        entry = dataset[i]
                        
                        # Handle numpy structured array entries (protein_id, sequence)
                        if hasattr(entry, 'item') and isinstance(entry.item(), tuple):
                            # This is a numpy void object containing (protein_id, sequence)
                            protein_id, sequence = entry.item()
                            
                            # Decode bytes to string if necessary
                            if isinstance(sequence, bytes):
                                sequence = sequence.decode('utf-8')
                            if isinstance(protein_id, bytes):
                                protein_id = protein_id.decode('utf-8')
                                
                            sequences.append(sequence)
                            
                        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                            # Format: (protein_id, sequence) or similar
                            protein_id, sequence = entry[0], entry[1]
                            
                            # Decode bytes to string if necessary
                            if isinstance(sequence, bytes):
                                sequence = sequence.decode('utf-8')
                            if isinstance(protein_id, bytes):
                                protein_id = protein_id.decode('utf-8')
                                
                            sequences.append(sequence)
                            
                        elif hasattr(entry, 'shape'):
                            # Pre-tokenized sequence as array
                            if len(entry.shape) == 1:
                                # 1D array of token IDs - convert back to space-separated string
                                sequences.append(' '.join(map(str, entry)))
                            else:
                                accelerator.print(f"Unexpected entry shape: {entry.shape}")
                                continue
                                
                        else:
                            # Try to treat as string sequence directly
                            if isinstance(entry, bytes):
                                entry = entry.decode('utf-8')
                            sequences.append(str(entry))
                        
                        total_loaded += 1
                
                accelerator.print(f"Loaded {len(sequences)} sequences from {h5_path}")
                
                if not sequences:
                    raise ValueError(f"Could not find sequence data in H5 file: {h5_path}")
                
                # Show a sample for debugging
                if sequences:
                    accelerator.print(f"Sample sequence: {sequences[0][:100]}...")
                
                return Dataset.from_dict({"text": sequences})
        
        # Load train dataset with limited sequences for testing
        train_path = config.dataset.train_path
        raw_datasets = {"train": load_h5_as_dataset(train_path, max_sequences=1000)}
        
        # Load validation and test if provided
        if hasattr(config.dataset, 'valid_path') and config.dataset.valid_path:
            raw_datasets["validation"] = load_h5_as_dataset(config.dataset.valid_path, max_sequences=500)
        if hasattr(config.dataset, 'test_path') and config.dataset.test_path:
            raw_datasets["test"] = load_h5_as_dataset(config.dataset.test_path, max_sequences=500)
        
        # Create tokenizer
        tokenizer = create_tokenizer(config.tokenizer)
        tokenizer.save_pretrained(tokenizer_path)

        # Tokenize datasets
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=False, max_length=config.model.context_length)

        accelerator.print("Tokenizing datasets...")
        tokenized_datasets = {}
        for split_name, dataset in raw_datasets.items():
            accelerator.print(f"Tokenizing {split_name} split...")
            tokenized_datasets[split_name] = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=1 if len(dataset) < 1000 else min(4, multiprocessing.cpu_count())
            )
        
        # Convert to datasets object
        from datasets import DatasetDict
        tokenized_datasets = DatasetDict(tokenized_datasets)
        
        # Save tokenized datasets
        tokenized_datasets.save_to_disk(tokenized_data_path)
        
        # Save checksum
        with open(checksum_path, "w") as f:
            f.write(checksum)

    accelerator.wait_for_everyone()
    
    # Load for all processes
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
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
    """Create tokenizer based on config."""
    if tokenizer_config.type == "file":
        # Load vocabulary from file
        vocab_path = tokenizer_config.path
        with open(vocab_path, 'r') as f:
            vocab = [line.strip() for line in f if line.strip()]
        
        vocab_map = {token: i for i, token in enumerate(vocab)}
        tokenizer = Tokenizer(WordLevel(vocab=vocab_map, unk_token=tokenizer_config.unk_token))
        
        # Convert to fast tokenizer
        with tempfile.TemporaryDirectory() as tempdir:
            tokenizer_path = os.path.join(tempdir, "tokenizer.json")
            tokenizer.save(tokenizer_path)
            fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            
            # Add special tokens
            special_tokens = {}
            if hasattr(tokenizer_config, 'pad_token'):
                special_tokens['pad_token'] = tokenizer_config.pad_token
            if hasattr(tokenizer_config, 'unk_token'):
                special_tokens['unk_token'] = tokenizer_config.unk_token
            if hasattr(tokenizer_config, 'mask_token'):
                special_tokens['mask_token'] = tokenizer_config.mask_token
            
            if special_tokens:
                fast_tokenizer.add_special_tokens(special_tokens)
        
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
