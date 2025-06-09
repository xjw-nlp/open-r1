# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name open-r1/OpenR1-Math-220k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from transformers import AutoConfig, AutoModelForCausalLM

from open_r1.configs import SFTConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from sft_trainer_ours import SFTTrainer

from seqrec_utils import load_train_datasets
from collator import Collator


logger = logging.getLogger(__name__)


@dataclass
class SeqRecScriptArguments(ScriptArguments):
    # deepspeed: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Used for deepspeed json file."
    #     }
    # )
    only_train_response: bool = field(
        default=False, 
        metadata={
            "help": "Compute loss on the response only."
        }
    )
    data_path: str = field(
        default="", 
        metadata={
            "help": "Data directory."
        }
    )
    dataset: str = field(
        default="Instruments", 
        metadata={
            "help": "Dataset name."
        }
    )
    max_his_len: int = field(
        default=20, 
        metadata={
            "help": "The max number of items in history sequence, -1 means no limit."
        }
    )
    index_file: str = field(
        default=".index.json", 
        metadata={
            "help": "The item indices file."
        }
    )
    his_sep: str = field(
        default=", ", 
        metadata={
            "help": "The separator used for history."
        }
    )
    add_prefix: bool = field(
        default=False, 
        metadata={
            "help": "Whether add sequential prefix in history."
        }
    )
    tasks: str = field(
        default="seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain", 
        metadata={
            "help": "Downstream tasks, separate by comma."
        }
    )
    train_prompt_sample_num: str = field(
        default="1,1,1,1,1,1", 
        metadata={
            "help": "The number of sampling prompts for each task."
        }
    )
    train_data_sample_num: str = field(
        default="0,0,0,100000,0,0", 
        metadata={
            "help": "The number of sampling prompts for each task."
        }
    )
    valid_prompt_id: int = field(
        default=0, 
        metadata={
            "help": "The prompt used for validation."
        }
    )
    sample_valid: bool = field(
        default=False, 
        metadata={
            "help": "Use sampled prompt for validation."
        }
    )
    valid_prompt_sample_num: int = field(
        default=2, 
        metadata={
            "help": "The number of sampling validation sequential recommendation prompts."
        }
    )
    use_chat_template: bool = field(
        default=False, 
        metadata={
            "help": "Whether to use the chat template in tokenizier."
        }
    )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load config
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    # breakpoint()
    if not hasattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token
    
     ################
    # Load datasets
    ################
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_data, valid_data = load_train_datasets(script_args, tokenizer)
    
    # Add new tokens
    new_tokens = [f'<{i}_{j}>' for i in ['a', 'b', 'c', 'd'] for j in range(256)]
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens(new_tokens))
    if not "<think>" in tokenizer.get_vocab():
        print('Add the token "<think>"')
        add_num += tokenizer.add_tokens(['<think>'])
    if not "</think>" in tokenizer.get_vocab():
        print('Add the token "</think>"')
        add_num += tokenizer.add_tokens(['</think>'])

    config.vocab_size = len(tokenizer)
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print("add {} new token.".format(add_num))
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)
    if hasattr(training_args, 'max_seq_length'):
        tokenizer.model_max_length = training_args.max_seq_length
    collator = Collator(script_args, tokenizer)

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=True,
        **model_kwargs,
    )
    model_args.model_name_or_path = None
    print('DEBUG: load model')
    model.resize_token_embeddings(len(tokenizer))
    print(model)
    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=tokenizer,
        data_collator=collator,
        # data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )
    
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_data)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(valid_data)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((SeqRecScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # breakpoint()
    print(script_args)
    print('<>')
    print(training_args)
    print('<>')
    print(model_args)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    main(script_args, training_args, model_args)
