# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from safetensors.torch import save_file

import deepspeed
from transformers.integrations import deepspeed as hfdeepspeed

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
from medusa.model.medusa_model_legacy import MedusaModel, MedusaConfig
from datasets import load_dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def alpaca_chat_template(messages):
    # Format the messages as a chat with alternating "user" and "assistant" turns
    chat = ""
    for message in messages:
        if message['role'] == 'user':
            chat += f"User: {message['content']}\n"
        elif message['role'] == 'assistant':
            chat += f"Assistant: {message['content']}\n"
    return chat.strip()

# Customized for training Medusa heads
class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            medusa = model.module.medusa
        else:
            medusa = model.medusa

        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(2 + i)].contiguous()
            medusa_labels = labels[..., 2 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
        self.log(log)
        return (loss, logits) if return_outputs else loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.1-405B")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = None
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=5,
        metadata={"help": "Number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (transformers.Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess(
    formatted_text,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    if "### Response:" not in formatted_text:
        raise ValueError("formatted_text must contain '### Response:'")

    # Tokenize the formatted text
    encoding = tokenizer(
        formatted_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    # Initialize targets with ignore tokens
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Find the assistant response in the formatted text and mask it
    assistant_response = formatted_text.split("### Response:")[1].strip()
    start = formatted_text.index(assistant_response)
    stop = start + len(assistant_response)

    # Mask the assistant response in the targets
    for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[0]):
        if tok_start >= start and tok_stop <= stop:
            targets[0][tok_index] = input_ids[0][tok_index]

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.input_ids = raw_data["input_ids"]  # Directly use the tensors
        self.labels = raw_data["labels"]
        self.attention_mask = raw_data["attention_mask"]
        # rank0_print("Formatting inputs...")
        # self.tokenizer = tokenizer
        # self.input_ids = []
        # self.labels = []
        # self.attention_mask = []

        # from tqdm import tqdm
        # # Use tqdm to add a progress bar
        # for example in tqdm(raw_data, desc="Preprocessing data"):
        #     formatted_text = example["formatted_text"]
        #     data_dict = preprocess(formatted_text, tokenizer)
        #     self.input_ids.append(data_dict["input_ids"][0])
        #     self.labels.append(data_dict["labels"][0])
        #     self.attention_mask.append(data_dict["attention_mask"][0])

        # # Convert lists to tensors
        # self.input_ids = torch.stack(self.input_ids)
        # self.labels = torch.stack(self.labels)
        # self.attention_mask = torch.stack(self.attention_mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


def load_preprocessed_data(load_path, tokenizer):
    """Load preprocessed dataset from disk."""
    if os.path.exists(load_path):
        print(f"Loading preprocessed data from {load_path}")
        data_dict = torch.load(load_path)
        print(data_dict.keys())
        return SupervisedDataset(data_dict, tokenizer)
    else:
        print(f"No preprocessed data found at {load_path}")
        return None
    
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """

    rank0_print("Loading data...")

    from datasets import load_from_disk
    train_dataset = load_from_disk("/work1/deming/shared/alpaca-gpt4/preprocessed_train")
    half_length = len(train_dataset) // 3
    train_dataset = train_dataset.select(range(half_length))
    # train_dataset = load_preprocessed_data("/work1/deming/shared/alpaca-gpt4/preprocessed_train.pt", tokenizer)
    if train_dataset is None:
        # Preprocess and save training data
        train_dataset = load_dataset("/work1/deming/shared/alpaca-gpt4", split="train")
        def format_dataset(example):
            messages = [
                {"role": "user", "content": example["instruction"] + "\n" + example["input"]},
                {"role": "assistant", "content": example["output"]},
            ]
            example["formatted_text"] = tokenizer.apply_chat_template(messages, tokenize=False)
            return example

        train_dataset = train_dataset.map(format_dataset)

        # Step 3: Tokenize the formatted text
        def tokenize(example):
            tokenized = tokenizer(
                example["formatted_text"],
                truncation=True,
                padding="max_length",
                max_length=8192,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = train_dataset.map(tokenize)

        columns_to_keep = ["input_ids", "attention_mask", "labels"]
        all_columns = tokenized_dataset.column_names
        columns_to_remove = [col for col in all_columns if col not in columns_to_keep]

        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

        train_save_path = os.path.join(data_args.data_path, "/work1/deming/shared/alpaca-gpt4/preprocessed_train")
        tokenized_dataset.save_to_disk(train_save_path)
        print(f"Preprocessed data saved to {train_save_path}")
        train_dataset = tokenized_dataset

    # print(f"train_dataset after dataset_cls: {train_dataset}")

    # # Load eval dataset from Parquet file if provided
    # if data_args.eval_data_path:
    #     eval_dataset = load_dataset("parquet", data_files=data_args.eval_data_path, split="train")
    #     eval_dataset = dataset_cls(eval_dataset, tokenizer=tokenizer)
    # else:
    #     eval_dataset = None


    return dict(train_dataset=train_dataset, eval_dataset=None)




def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names=["labels"]
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token

    alpaca_template = """
    {% if messages[0]['role'] == 'system' %}
        {% set loop_messages = messages[1:] %}
        {% set system_message = messages[0]['content'] %}
    {% else %}
        {% set loop_messages = messages %}
        {% set system_message = false %}
    {% endif %}

    {% for message in loop_messages %}
        {% if message['role'] == 'user' %}
            {{ '### Instruction:\n' + message['content'] + '\n' }}
        {% elif message['role'] == 'assistant' %}
            {{ '### Response:\n' + message['content'] + '\n' }}
        {% endif %}
    {% endfor %}
    """

    tokenizer.chat_template = alpaca_template

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Define a function for Alpaca-style chat formatti

    # distributed setup
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    dschf = hfdeepspeed.HfDeepSpeedConfig("/work1/deming/seliny2/axolotl/deepspeed/zero3-offload.json")  # keep this object alive

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16,
    )

    model_engine = deepspeed.initialize(
        config="/work1/deming/seliny2/axolotl/deepspeed/zero3-offload.json",
        model=model,
        model_parameters=model.parameters(),
        config_params="/work1/deming/seliny2/axolotl/deepspeed/zero3-offload.json")[0]
    model = model_engine.module

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    for  param in model.parameters():
        if param.requires_grad and param.shape[-1] != 0 :
            print(f"Trainable Parameter Shape: {param.shape}")

    # Add Medusa heads
    medusa_lm_head = MedusaModel(
        model,
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )
    # Generate Medusa config for pushing to HF hub
    medusa_config = MedusaConfig(
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
        version="1"
    )

    training_args.output_dir = f"{training_args.output_dir}_medusa_mlp_{model_args.model_name_or_path.split('/')[-1]}_medusa_{training_args.medusa_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.medusa_num_layers}"

    # Save Medusa config
    medusa_config.save_pretrained(training_args.output_dir)

    # Start trainner
    trainer = CustomizedTrainer(
        model=medusa_lm_head, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True

    if hasattr(medusa_lm_head, "module"):
        lm_head = medusa_lm_head.module.medusa_head
    else:
        lm_head = medusa_lm_head.medusa_head

    print(f"Training complete.")

    with deepspeed.zero.GatheredParameters(lm_head.parameters()):
        state_dict = lm_head.state_dict()

    # Save Medusa heads
    if local_rank == 0:
        # Modify the tokenizer internal state before saving.
        tokenizer.encode("Test", truncation=None, padding="do_not_pad")
        tokenizer.save_pretrained(training_args.output_dir)
        save_file(
            state_dict,
            os.path.join(training_args.output_dir, "medusa_lm_head.safetensors"),
        )


if __name__ == "__main__":
    train()
