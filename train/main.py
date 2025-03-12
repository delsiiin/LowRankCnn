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

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os

from transformers import LlamaForCausalLM

import random

from data_utils import make_supervised_data_module

import copy

def seed_torch(seed=42):
 
    random.seed(seed)
 
    np.random.seed(seed)
 
    torch.manual_seed(seed)
 
    torch.cuda.manual_seed(seed)
 
    torch.cuda.manual_seed_all(seed) 
 
    torch.backends.cudnn.benchmark = False
 
    torch.backends.cudnn.deterministic = True

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

seed_torch(42)

# Customized for training Medusa heads

def replace_compute_loss_cross_entropy():
    
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
        
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        loss = outputs[0]

        logits = outputs[1]

        if loss == 0:

            labels = inputs["labels"]

            loss_fct = CrossEntropyLoss()

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss += loss_fct(shift_logits, shift_labels)
        
        return (loss, logits) if return_outputs else loss
    
    transformers.trainer.Trainer.compute_loss = compute_loss

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
   
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
    model_type: str = field(
        default="vicuna",
        metadata={"help": "vicuna/llama2/llama3"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    attn_down_proj_dim: int = field(
        default=1024,
        metadata={"help": "Down projection dimension for attention."},
    )
    num_deep_cnn_layers: int = field(
        default=2,
        metadata={"help": "Number of deep CNN layers."},
    )
    kernel_size: int = field(
        default=63,
        metadata={"help": "Kernel size for deep CNN layers."},
    )
    inner_channel: int = field(
        default=1,
        metadata={"help": "Inner channel for deep CNN layers."},
    )
    loss_type: str = field(
        default="task",
        metadata={"help": "Loss type."},
    )
    vsdebug: bool = field(
        default=False,
        metadata={"help": "Debug mode."},
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

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if training_args.vsdebug:
        import debugpy; debugpy.connect(('127.0.0.1', 5678))

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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    from models.LoRCnn.modeling_llama_cnn import LlamaForCausalLM as BaseLlamaForCausalLM
    config.max_position_embeddings = training_args.model_max_length
    # Load model and tokenizer
    model = BaseLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if model_args.load_in_4bit else None,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
    )

    # Generate Medusa config for pushing to HF hub
    from models.LoRCnn.LoRCnn_config import LoRCnnConfig
    cnn_config = LoRCnnConfig(
        attn_down_proj_dim=training_args.attn_down_proj_dim,
        num_deep_cnn_layers=training_args.num_deep_cnn_layers,
        inner_channel=training_args.inner_channel,
        kernel_size=training_args.kernel_size,
        loss_type=training_args.loss_type,
        **config.to_dict()  # Inherit all parameters from the base config
    )

    cnn_config.max_position_embeddings = training_args.model_max_length

    print(cnn_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    if data_args.model_type == "llama3":
        tokenizer.pad_token_id = 0
    else:
        tokenizer.pad_token = tokenizer.unk_token

    # Check if datasets already exist
    train_dataset_path = os.path.join(training_args.output_dir, f"train_dataset_{data_args.model_type}.pt")
    eval_dataset_path = os.path.join(training_args.output_dir, f"eval_dataset_{data_args.model_type}.pt")

    if os.path.exists(train_dataset_path) and os.path.exists(eval_dataset_path):
        # Load the datasets
        train_dataset = torch.load(train_dataset_path)
        eval_dataset = torch.load(eval_dataset_path)
        data_module = {"train_dataset": train_dataset, "eval_dataset": eval_dataset}
    else:
        # Load data
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        # Save the datasets for future use
        torch.save(data_module["train_dataset"], train_dataset_path)
        torch.save(data_module["eval_dataset"], eval_dataset_path)


    # Format output dir
    training_args.output_dir = os.path.join(training_args.output_dir, f"{model_args.model_name_or_path.split('/')[-1]}_LoRCnn_{training_args.attn_down_proj_dim}_lr_{training_args.learning_rate}_layer_{training_args.num_deep_cnn_layers}")

    # Save Medusa config
    cnn_config.save_pretrained(training_args.output_dir)

    # Add Medusa heads
    from models.LoRCnn.LoRCnn_model import LoRCnnModel
    LoRCnn_model = LoRCnnModel(
        model,
        cnn_config
    )

    # Freeze the base model
    for param in LoRCnn_model.parameters():
        param.requires_grad = False
    
    for name, param in LoRCnn_model.named_parameters():
        if "attn_down_proj" in name or "attention_predictor" in name:
            param.requires_grad = True

    # for name, param in LoRCnn_model.named_parameters():
    #     print(name, param.requires_grad)

    # import pdb; pdb.set_trace()
    # Start trainner

    replace_compute_loss_cross_entropy()

    LoRCnn_model.config.use_cache = False         # required for gradient checkpointing
    LoRCnn_model.enable_input_require_grads()     # required for gradient checkpointing
    # model.gradient_checkpointing_enable()  # enable gradient checkpointing

    trainer = Trainer(
        model=LoRCnn_model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if hasattr(LoRCnn_model, "module"):
        LoRCnn_model = LoRCnn_model.module
    else:
        LoRCnn_model = LoRCnn_model

    # 定义保存路径
    bin_filename = os.path.join(training_args.output_dir, "LoRCNN_weights.bin")

    # 遍历参数并存入字典
    merged_weights = {
        name: param.cpu() for name, param in LoRCnn_model.named_parameters()
        if "attn_down_proj" in name or "attention_predictor" in name
    }

    # 直接保存到 .bin 文件
    torch.save(merged_weights, bin_filename)

    # for name, param in LoRCNN_layers.named_parameters():
    #     if "attn_down_proj" in name or "attention_predictor" in name:
    #         torch.save(
    #             param,
    #             os.path.join(training_args.output_dir, f"{name.replace('.', '_')}.pt"),
    #         )

if __name__ == "__main__":
    train()